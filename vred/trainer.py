"""One stage controller; the student model's ordinary forward is unchanged."""

from contextlib import contextmanager
from dataclasses import replace
import math

import torch
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from .effects import (EffectCapture, relative_pairs, response_effect_scores,
                      stratified_write_positions, subset_effects, write_positions)
from .losses import (effect_weighted_kd_sparse, floored_effect_weights, logits_kd,
                     logits_kd_sparse, weighted_kd_sparse, write_loss)


class VREDTrainer:
    def __init__(self, student, teacher, stage, config, seed=1234):
        if stage not in ("dpt", "sft", "dft"):
            raise ValueError(f"Unknown stage: {stage}")
        self.student, self.teacher, self.stage, self.config = student, teacher, stage, config
        self.settings = config["stages"][stage]
        if stage == "dpt" and self.settings["lambda_logits"] != 0:
            raise ValueError("DPT must not include final-logit KD")
        if stage == "sft" and (self.settings["lambda_write"] != 0 or self.settings["lambda_logits"] != 0):
            raise ValueError("SFT is CE only")
        self.generator = torch.Generator().manual_seed(seed)
        self.write_scope = config.get("write_position_scope", "response")
        self.write_effect_sampling = config.get("write_effect_sampling", "uniform")
        self.write_lambda_schedule = config.get("write_lambda_schedule", "constant")
        if self.write_scope not in ("response", "prompt", "prompt_response"):
            raise ValueError(f"Unknown write position scope: {self.write_scope}")
        if self.write_effect_sampling not in ("uniform", "stratified"):
            raise ValueError(f"Unknown write effect sampling: {self.write_effect_sampling}")
        if self.write_lambda_schedule not in ("constant", "warmup_cosine"):
            raise ValueError(f"Unknown write lambda schedule: {self.write_lambda_schedule}")
        if self.write_effect_sampling == "stratified" and config.get("write_position_limit") is None:
            raise ValueError("write_effect_sampling=stratified requires write_position_limit")
        self.write_step, self.write_total_steps = 0, None
        self.sparse = bool(config.get("sparse_logits", True))
        self.effect_weighted_kd = bool(config.get("effect_weighted_kd", False))
        self.lambda_effect_kd = float(config.get("lambda_effect_kd", 1.0))
        if self.lambda_effect_kd < 0 or not math.isfinite(self.lambda_effect_kd):
            raise ValueError("lambda_effect_kd must be finite and nonnegative")
        if self.effect_weighted_kd and stage != "dft":
            raise ValueError("effect_weighted_kd is only supported for DFT")
        if self.effect_weighted_kd and not self.sparse:
            raise ValueError("effect_weighted_kd requires sparse logits")
        self.effect_kd_weight_source = str(config.get("effect_kd_weight_source", "logits"))
        if self.effect_kd_weight_source not in ("logits", "hidden"):
            raise ValueError("effect_kd_weight_source must be 'logits' or 'hidden'")
        if self.effect_kd_weight_source == "hidden":
            if not self.effect_weighted_kd:
                raise ValueError("hidden effect KD requires effect_weighted_kd")
            if self.write_scope != "response":
                raise ValueError("hidden effect KD requires write_position_scope=response")
            if config.get("write_position_limit") is not None:
                raise ValueError("hidden effect KD requires all response positions")
            if self.write_effect_sampling != "uniform":
                raise ValueError("hidden effect KD requires write_effect_sampling=uniform")
            self.effect_kd_floor = float(config.get("effect_kd_floor", 0.1))
            if not math.isfinite(self.effect_kd_floor) or not 0 <= self.effect_kd_floor < 1:
                raise ValueError("effect_kd_floor must be finite and in [0,1)")
        else:
            self.effect_kd_floor = 0.0
        student.requires_grad_(True)
        if stage == "dpt":
            student.language_model.requires_grad_(False)
        student.train()
        if stage == "sft":
            if teacher is not None:
                raise ValueError("SFT must not load a teacher")
            self.pairs = []
        else:
            if teacher is None:
                raise ValueError(f"{stage}: a frozen teacher is required")
            teacher.requires_grad_(False).eval()
            self.pairs = relative_pairs(len(student.language_model.model.layers),
                                        len(teacher.language_model.model.layers), self.settings["pairs"])
        self.selective_checkpointing = bool(config.get("selective_checkpointing", False)) and stage in ("dpt", "dft")
        self.checkpoint_excluded = {student_index for student_index, _ in getattr(self, "pairs", [])}

    @contextmanager
    def _checkpointed_backbone(self):
        """Checkpoint non-selected student blocks only.

        Selected blocks must keep their ordinary forward so the effect-capture
        hooks see real attention probabilities and post-MLP outputs.  The
        non-selected blocks are recomputed during backward; there is no dropout
        in these models, so the forward values and gradients are unchanged.
        The wrappers are restored immediately after the forward; the checkpoint
        graph keeps the original functions for its later recomputation.
        """
        if not self.selective_checkpointing:
            yield
            return
        layers = self.student.language_model.model.layers
        installed = []
        for index, layer in enumerate(layers):
            if index in self.checkpoint_excluded:
                continue
            original = layer.forward

            def wrapped(*args, _original=original, _layer=layer, **kwargs):
                if torch.is_grad_enabled() and _layer.training:
                    return activation_checkpoint(_original, *args, use_reentrant=False, **kwargs)
                return _original(*args, **kwargs)

            layer.forward = wrapped
            installed.append((layer, original))
        try:
            yield
        finally:
            for layer, original in installed:
                layer.forward = original

    def set_write_schedule(self, total_steps):
        """Set the optimizer-step horizon used by warmup_cosine; no-op when constant."""
        self.write_total_steps = max(int(total_steps), 1)

    def set_write_step(self, optimizer_step):
        self.write_step = int(optimizer_step)

    def write_lambda_scale(self):
        if self.write_lambda_schedule == "constant" or self.write_total_steps is None:
            return 1.0
        progress = min(max(self.write_step / self.write_total_steps, 0.0), 1.0)
        warmup = float(self.config.get("warmup_ratio", 0.03))
        if warmup <= 0:
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        if warmup >= 1:
            return progress
        if progress < warmup:
            return progress / warmup
        return 0.5 * (1.0 + math.cos(math.pi * (progress - warmup) / (1.0 - warmup)))

    def _student_forward(self, batch, forward_model):
        """Run the student, optionally through a DeepSpeed engine wrapper.

        Sparse mode passes ``sparse=True`` through the wrapper so the engine's
        ordinary forward path (including any configured hooks) is preserved; the
        returned sparse loss stays connected to all trainable parameters for
        ``engine.backward``.
        """
        if forward_model is None:
            return self.student(batch) if not self.sparse else self.student(
                batch, sparse=True, return_logits=self.stage == "dft",
                ce_chunk_size=self.config["ce_chunk_size"])
        return forward_model(batch) if not self.sparse else forward_model(
            batch, sparse=True, return_logits=self.stage == "dft",
            ce_chunk_size=self.config["ce_chunk_size"])

    def loss(self, batch, forward_model=None, normalization_counts=None):
        """Return a differentiable scalar and detached raw/weighted metrics.

        forward_model may be the DeepSpeed engine wrapping exactly self.student.
        Both models consume the same Batch; coordinates and target IDs cannot drift.
        """
        candidate_limit = None if self.write_effect_sampling == "stratified" else self.config["write_position_limit"]
        candidates = write_positions(batch, candidate_limit, self.generator, scope=self.write_scope)
        positions = candidates
        zero = next(self.student.parameters()).new_zeros((), dtype=torch.float32)
        write, raw_write, final_kd, effect_kd = zero, zero, zero, zero
        teacher_output = None
        teacher_logits = None
        teacher_minus_logits = None
        teacher_effects = {}
        if self.stage != "sft":
            # Stratified sampling needs teacher effects at every candidate
            # before it can choose the write subset. Uniform keeps the original
            # one-pass path: effects are captured only at the selected positions.
            with torch.no_grad(), EffectCapture(self.teacher, [t for _, t in self.pairs], batch,
                                                candidates) as targets:
                if self.stage == "dpt":
                    self.teacher.forward_backbone(batch)
                elif self.sparse:
                    teacher_logits = self.teacher.forward_sparse(batch, compute_loss=False).logits
                else:
                    teacher_output = self.teacher(batch)
                if self.write_effect_sampling == "stratified" and len(candidates):
                    positions, selected = stratified_write_positions(
                        batch, targets.effects, self.pairs, self.config["write_position_limit"],
                        self.generator, scope=self.write_scope)
                    teacher_effects = subset_effects(targets.effects, selected)
                else:
                    teacher_effects = targets.effects
            if self.effect_weighted_kd and self.effect_kd_weight_source == "logits":
                # Counterfactual teacher: same text/positions, visual features zeroed only
                # for the logits-effect variant. Hidden-effect KD reuses teacher_effects.
                # The effect-capture hooks have already been removed, so this second
                # forward cannot overwrite the write targets.
                counterfactual = replace(batch, features=tuple(
                    None if feature is None else torch.zeros_like(feature)
                    for feature in batch.features))
                with torch.no_grad():
                    teacher_minus_logits = self.teacher.forward_sparse(
                        counterfactual, compute_loss=False).logits
            with EffectCapture(self.student, [s for s, _ in self.pairs], batch, positions) as sources:
                with self._checkpointed_backbone():
                    output = self._student_forward(batch, forward_model)
            write, raw_write = write_loss(
                self.student, self.teacher, sources.effects, teacher_effects, self.pairs,
                chunk_size=self.config["query_chunk_size"], temperature=self.config["tau_write"],
                kind=self.config["write_loss_type"], vocab_reduction=self.config["write_vocab_reduction"],
                beta=self.config["smoothl1_beta"], eta=self.config["effect_kl_eta"],
                checkpoint_probes=self.config["checkpoint_probes"],
                vocab_chunk_size=self.config["probe_vocab_chunk_size"])
            if self.stage == "dft":
                if self.sparse:
                    if self.settings["lambda_logits"] > 0:
                        final_kd = logits_kd_sparse(
                            output.logits, teacher_logits,
                            self.config["tau_logits"], self.config["query_chunk_size"])
                    if self.effect_weighted_kd:
                        if self.effect_kd_weight_source == "hidden":
                            scores = response_effect_scores(batch, teacher_effects, self.pairs)
                            weights = floored_effect_weights(scores, self.effect_kd_floor)
                            effect_kd = weighted_kd_sparse(
                                output.logits, teacher_logits, weights,
                                self.config["tau_logits"], self.config["query_chunk_size"])
                        else:
                            effect_kd = effect_weighted_kd_sparse(
                                output.logits, teacher_logits, teacher_minus_logits,
                                self.config["tau_logits"], self.config["query_chunk_size"])
                else:
                    final_kd = logits_kd(output.logits, teacher_output.logits, batch,
                                         self.config["tau_logits"], self.config["query_chunk_size"])
        else:
            output = self._student_forward(batch, forward_model)
        ce = output.loss.float()
        response_count = int(batch.response_mask.sum())
        response_scale, write_scale = 1.0, 1.0
        distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        world = torch.distributed.get_world_size() if distributed else 1
        if normalization_counts is not None:
            responses, writes, group_size = normalization_counts
            response_scale = world * response_count * group_size / max(responses, 1)
            write_scale = world * len(positions) * group_size / max(writes, 1)
        elif distributed:
            counts = torch.tensor([response_count, len(positions)], device=ce.device, dtype=torch.float32)
            torch.distributed.all_reduce(counts)
            # ZeRO averages gradients across ranks. Restore the global valid-token
            # mean when ranks have unequal response/write counts.
            response_scale = world * response_count / counts[0].clamp_min(1)
            write_scale = world * len(positions) / counts[1].clamp_min(1)
        ce = ce * response_scale
        final_kd = final_kd * response_scale
        effect_kd = effect_kd * response_scale
        write, raw_write = write * write_scale, raw_write * write_scale
        write_lambda_scale = self.write_lambda_scale()
        weighted_write = self.settings["lambda_write"] * write_lambda_scale * write
        weighted_kd = self.settings["lambda_logits"] * final_kd
        weighted_effect_kd = self.lambda_effect_kd * effect_kd
        total = ce + weighted_write + weighted_kd + weighted_effect_kd
        if not total.requires_grad:
            # A text-only DPT batch has no path to the trainable connector.
            total = total + sum(p.reshape(-1)[0] * 0 for p in self.student.parameters() if p.requires_grad)
        metrics = {"loss": total.detach(), "ce": ce.detach(), "write_raw": raw_write.detach(),
                   "write": write.detach(), "write_weighted": weighted_write.detach(),
                   "logits_kd": final_kd.detach(), "logits_kd_weighted": weighted_kd.detach(),
                   "write_lambda_scale": write_lambda_scale,
                   "response_tokens": response_count, "write_positions": len(positions)}
        if self.effect_weighted_kd:
            metrics["effect_kd"] = effect_kd.detach()
            metrics["effect_kd_weighted"] = weighted_effect_kd.detach()
        return total, metrics


def accumulation_groups(loader, accumulation, position_limit=None, device="cpu", write_scope="response"):
    """Prefetch CPU batches to normalize by all valid tokens in an optimizer step.

    At the configured microbatch=1/accumulation=32, cached features occupy about
    52 MiB in this queue. No model activations or vocabulary tensors are queued.
    """
    from itertools import islice

    iterator = iter(loader)
    while True:
        group = list(islice(iterator, accumulation))
        if not group:
            return
        responses, writes = 0, 0
        for batch in group:
            responses += int(batch.response_mask.sum())
            coordinates = write_positions(batch, scope=write_scope)
            if position_limit is None:
                writes += len(coordinates)
            else:
                counts = torch.bincount(coordinates[:, 0], minlength=len(batch.record_ids))
                writes += int(counts.clamp_max(position_limit).sum())
        counts = torch.tensor([responses, writes], device=device, dtype=torch.long)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(counts)
        yield group, (int(counts[0]), int(counts[1]), len(group))

"""Language model plus cached-feature connector; never imports a vision implementation."""

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def validate_tokenizers(student_path, teacher_path, student_config, teacher_config):
    student = AutoTokenizer.from_pretrained(student_path, local_files_only=True, use_fast=True)
    teacher = AutoTokenizer.from_pretrained(teacher_path, local_files_only=True, use_fast=True)
    for name in ("get_vocab", "get_added_vocab"):
        if getattr(student, name)() != getattr(teacher, name)():
            raise ValueError(f"Tokenizer {name} mismatch: {student_path} vs {teacher_path}")
    if json.loads(student.backend_tokenizer.to_str()) != json.loads(teacher.backend_tokenizer.to_str()):
        raise ValueError(f"Tokenizer segmentation/normalization mismatch: {student_path} vs {teacher_path}")
    for name in ("bos_token_id", "eos_token_id", "pad_token_id", "chat_template"):
        if getattr(student, name, None) != getattr(teacher, name, None):
            raise ValueError(f"Tokenizer {name} mismatch: {student_path} vs {teacher_path}")
    left, right = dict(student.special_tokens_map), dict(teacher.special_tokens_map)
    # The supplied teacher adds only an UNK alias to the existing EOS/PAD token.
    if left.get("unk_token") is None and right.get("unk_token") == teacher.eos_token:
        right.pop("unk_token")
    if left != right:
        raise ValueError(f"Tokenizer special token mismatch: {student_path} vs {teacher_path}")
    if student_config.vocab_size != teacher_config.vocab_size:
        raise ValueError("Teacher/student model vocabulary widths differ")
    if max(student.get_vocab().values()) >= student_config.vocab_size:
        raise ValueError("Tokenizer IDs exceed the model vocabulary")
    for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
        left, right = getattr(student_config, name, None), getattr(teacher_config, name, None)
        if left != right:
            raise ValueError(f"Model {name} mismatch: {left} vs {right}")
        token = getattr(student, name)
        if name == "bos_token_id" and token is None and left == student.eos_token_id:
            continue  # Qwen base configs alias BOS to EOS; these tokenizers insert no BOS.
        if left is not None and token != left:
            raise ValueError(f"Model/tokenizer {name} mismatch: {left} vs {token}")
    return teacher


@dataclass
class SparseOutput:
    """CE and (optionally) logits restricted to shifted response positions."""

    loss: torch.Tensor
    logits: torch.Tensor = None


class VREDModel(nn.Module):
    def __init__(self, language_model, vision_width=1152, vision_tokens=729):
        super().__init__()
        self.language_model = language_model
        hidden = language_model.config.hidden_size
        self.connector = nn.Sequential(nn.Linear(vision_width, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.vision_tokens = vision_tokens
        self.load_decision = {"format": "base_language_model", "adapter_merges": 0}

    def forward(self, batch, sparse=False, **kwargs):
        if sparse:
            return self.forward_sparse(batch, **kwargs)
        embeddings = self._embeddings(batch)
        return self.language_model(inputs_embeds=embeddings, attention_mask=batch.attention_mask,
                                   labels=batch.labels, use_cache=False, **kwargs)

    def forward_backbone(self, batch):
        """Run decoder blocks for DPT targets without the final vocabulary head or CE."""
        return self.language_model.model(inputs_embeds=self._embeddings(batch),
                                         attention_mask=batch.attention_mask, use_cache=False)

    def forward_sparse(self, batch, compute_loss=True, return_logits=True, ce_chunk_size=0):
        """Ordinary backbone forward; head/CE restricted to shifted response positions.

        Mathematically identical to the dense causal-LM CE because ignored labels
        contribute zero gradient, but avoids materializing [batch, sequence, vocab].
        With ``ce_chunk_size`` > 0 the head/CE are also processed in chunks of
        response positions: the loss is a weighted mean of per-chunk CE sums, so
        no [all_response_positions, vocab] FP32 tensor is required.  ``logits`` is
        only returned (and only fully materialized) when ``return_logits`` is true;
        DFT needs it for final KD, SFT/DPT do not.
        """
        if ce_chunk_size < 0:
            raise ValueError("ce_chunk_size must be nonnegative")
        backbone = self.language_model.model(inputs_embeds=self._embeddings(batch),
                                             attention_mask=batch.attention_mask, use_cache=False)
        hidden = backbone.last_hidden_state  # Qwen2Model already applied its final RMSNorm
        positions = batch.response_positions
        if len(positions):
            rows, columns = positions.unbind(1)
            states = hidden[rows, columns]
        else:
            rows = columns = None
            states = hidden.reshape(-1, hidden.shape[-1])[:0]
        head = self.language_model.get_output_embeddings()
        full_logits = head(states) if (return_logits or (compute_loss and not ce_chunk_size)) else None
        loss = None
        if compute_loss:
            if rows is None:
                loss = hidden.sum() * 0  # keep a differentiable scalar for empty batches
            else:
                targets = batch.labels[rows, columns + 1]
                if ce_chunk_size:
                    total = states.new_zeros((), dtype=torch.float32)
                    for start in range(0, len(states), ce_chunk_size):
                        stop = min(start + ce_chunk_size, len(states))
                        chunk = full_logits[start:stop] if full_logits is not None else head(states[start:stop])
                        total = total + F.cross_entropy(chunk.float(), targets[start:stop],
                                                        ignore_index=-100, reduction="sum")
                    loss = total / len(states)
                else:
                    loss = F.cross_entropy(full_logits.float(), targets, ignore_index=-100)
        return SparseOutput(loss=loss, logits=full_logits if return_logits else None)

    def _embeddings(self, batch):
        embeddings = self.language_model.get_input_embeddings()(batch.input_ids)
        embeddings = embeddings.clone()
        for row, (start, end) in enumerate(batch.visual_spans.tolist()):
            feature = batch.features[row]
            if start < 0:
                if feature is not None:
                    raise ValueError(f"{batch.record_ids[row]}: text-only span has image features")
                continue
            if feature is None or end - start != self.vision_tokens or end > embeddings.shape[1]:
                raise ValueError(f"{batch.record_ids[row]}: inconsistent visual span {(start, end)}")
            if tuple(feature.shape) != (self.vision_tokens, self.connector[0].in_features):
                raise ValueError(f"{batch.record_ids[row]}: invalid feature shape {tuple(feature.shape)}")
            weight = self.connector[0].weight
            embeddings[row, start:end] = self.connector(feature.to(device=weight.device, dtype=weight.dtype))
        return embeddings

    def train(self, mode=True):
        super().train(False if getattr(self, "frozen_teacher", False) else mode)
        return self

    def save_pretrained(self, directory, tokenizer):
        if self.load_decision["format"] == "unmerged_llavakd_lora" and self.load_decision["adapter_merges"] != 1:
            raise ValueError("Merge the teacher adapter before saving VRED weights")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.language_model.save_pretrained(directory / "language_model")
        torch.save(self.connector.state_dict(), directory / "connector.pt")
        tokenizer.save_pretrained(directory)
        (directory / "vred.json").write_text(json.dumps({
            "vision_width": self.connector[0].in_features, "vision_tokens": self.vision_tokens,
            "load_decision": self.load_decision}, indent=2) + "\n")


def _teacher_language(path, dtype, merge, attn="eager"):
    """Read the explicitly pinned legacy format, applying each LoRA update exactly once."""
    adapter_config = json.loads((path / "adapter_config.json").read_text())
    for flag in ("fan_in_fan_out", "use_dora", "use_rslora", "rank_pattern", "alpha_pattern", "modules_to_save"):
        if adapter_config.get(flag):
            raise ValueError(f"{path}: unsupported adapter option {flag}")
    if adapter_config["bias"] != "none" or adapter_config["peft_type"] != "LORA":
        raise ValueError(f"{path}: expected bias-free LoRA format")
    state_path = path / "language_model/pytorch_model.bin"
    state = torch.load(state_path, map_location="cpu", weights_only=True, mmap=True)
    if not any(".base_layer." in key for key in state) or any("lora_" in key for key in state):
        raise ValueError(f"{state_path}: expected saved unmerged base_layer weights")
    state = {key.replace(".base_layer", ""): value for key, value in state.items()}
    config = AutoConfig.from_pretrained(path / "language_model", local_files_only=True)
    if config.tie_word_embeddings and "lm_head.weight" not in state:
        state["lm_head.weight"] = state["model.embed_tokens.weight"]
    lm = AutoModelForCausalLM.from_config(config, torch_dtype=dtype, attn_implementation=attn)
    lm.load_state_dict(state, strict=True, assign=True)
    lm.tie_weights()
    lm.to(dtype=dtype)
    adapter = load_file(str(path / "adapter_model.safetensors"), device="cpu")
    prefix = "base_model.model.language_model."
    expected = set()
    updates = []
    for target in adapter_config["target_modules"]:
        if not target.startswith("language_model."):
            raise ValueError(f"{path}: non-language adapter target {target}")
        name = target.removeprefix("language_model.")
        a_key, b_key = prefix + name + ".lora_A.weight", prefix + name + ".lora_B.weight"
        expected.update((a_key, b_key))
        a, b = adapter[a_key], adapter[b_key]
        module = lm.get_submodule(name)
        rank = adapter_config["r"]
        if a.shape != (rank, module.in_features) or b.shape != (module.out_features, rank):
            raise ValueError(f"{path}: invalid adapter dimensions for {name}")
        updates.append((module, a, b, adapter_config["lora_alpha"] / rank))
    if set(adapter) != expected:
        raise ValueError(f"{path}: unexpected/missing adapter keys: {set(adapter) ^ expected}")
    handles = []
    if not merge:
        for module, a, b, scale in updates:
            def hook(module, inputs, output, a=a.to(dtype), b=b.to(dtype), scale=scale):
                return output + F.linear(F.linear(inputs[0], a), b) * scale
            handles.append(module.register_forward_hook(hook))

    def merge_once():
        if getattr(lm, "_vred_merged", False):
            raise ValueError(f"{path}: attempted to merge teacher adapter twice")
        for handle in handles:
            handle.remove()
        with torch.no_grad():
            for module, a, b, scale in updates:
                module.weight.copy_((module.weight.float() + (b.float() @ a.float()) * scale).to(dtype))
        lm._vred_merged = True
        updates.clear()
        handles.clear()

    if merge:
        merge_once()
    return lm, merge_once


def load_model(config, role="student", device="cpu", dtype=torch.float32, saved=None,
               merge_teacher=True, attn="eager"):
    """Single entry point for a base student, frozen teacher, or saved VRED model.

    ``attn`` defaults to eager, which effect capture requires and which keeps
    training/evaluation on the same kernels. ``sdpa`` is an explicit opt-in for
    memory experiments only: it changes BF16 logits measurably (see README).
    merge_teacher=False is solely for the explicit adapter-vs-merged smoke check;
    call the returned model's merge_teacher_once() before using it for training.
    """
    if role not in ("student", "teacher"):
        raise ValueError(f"Unknown model role: {role}")
    if attn not in ("eager", "sdpa"):
        raise ValueError(f"Unsupported attention implementation: {attn}")
    validate_tokenizers(config["student"], config["teacher"],
                        AutoConfig.from_pretrained(config["student"], local_files_only=True),
                        AutoConfig.from_pretrained(Path(config["teacher"]) / "language_model", local_files_only=True))
    if saved is not None:
        path = Path(saved)
        metadata = json.loads((path / "vred.json").read_text())
        validate_tokenizers(path, config["teacher"],
                            AutoConfig.from_pretrained(path / "language_model", local_files_only=True),
                            AutoConfig.from_pretrained(Path(config["teacher"]) / "language_model", local_files_only=True))
        lm = AutoModelForCausalLM.from_pretrained(path / "language_model", local_files_only=True,
                                                 torch_dtype=dtype, attn_implementation=attn)
        model = VREDModel(lm, metadata["vision_width"], metadata["vision_tokens"])
        model.connector.load_state_dict(torch.load(path / "connector.pt", map_location="cpu", weights_only=True))
        model.load_decision = metadata["load_decision"]
    elif role == "student":
        lm = AutoModelForCausalLM.from_pretrained(config["student"], local_files_only=True,
                                                 torch_dtype=dtype, attn_implementation=attn)
        model = VREDModel(lm, config["vision_width"], config["vision_tokens"])
    else:
        path = Path(config["teacher"])
        if config["teacher_format"] != "unmerged_llavakd_lora":
            raise ValueError(f"{path}: teacher format decision must be unmerged_llavakd_lora")
        lm, merge_once = _teacher_language(path, dtype, merge_teacher, attn=attn)
        model = VREDModel(lm, config["vision_width"], config["vision_tokens"])
        connector = torch.load(path / "connector/pytorch_model.bin", map_location="cpu", weights_only=True)
        model.connector.load_state_dict({k.removeprefix("_connector."): v for k, v in connector.items()}, strict=True)
        model.load_decision = {"format": config["teacher_format"], "adapter_merges": int(merge_teacher),
                               "checkpoint": str(path)}

        def finish_merge():
            merge_once()
            model.load_decision["adapter_merges"] = 1
        model.merge_teacher_once = finish_merge
    model.to(device=device, dtype=dtype)
    if role == "teacher":
        model.frozen_teacher = True
        model.requires_grad_(False).eval()
    return model

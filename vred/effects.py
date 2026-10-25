"""Read-only eager-Qwen hooks for local visual-value interventions."""

from dataclasses import dataclass, fields
import inspect
import math

import torch
from torch.nn import functional as F
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, repeat_kv


def relative_pairs(student_depth, teacher_depth, count):
    if not 1 <= count <= min(student_depth, teacher_depth):
        raise ValueError("Layer-pair count must fit both decoder depths")
    return [(math.ceil(k * student_depth / count) - 1,
             math.ceil(k * teacher_depth / count) - 1) for k in range(1, count + 1)]


def write_positions(batch, limit=None, generator=None, scope="response"):
    """Return explicit write coordinates for one of three position scopes.

    ``response`` is the original behavior: supervised assistant tokens after
    the visual span. ``prompt`` selects only non-response tokens after the
    visual span (the system/question/assistant-prefix context); ``prompt_response``
    selects both prompt and response tokens.  All scopes require a real causal
    successor, which keeps the probe target well-defined at the last token.
    """
    if scope not in ("response", "prompt", "prompt_response"):
        raise ValueError(f"Unknown write position scope: {scope}")
    positions = torch.arange(batch.attention_mask.shape[1], device=batch.attention_mask.device)[None]
    spans = batch.visual_spans
    valid_span = (spans[:, 0] >= 0) & (spans[:, 1] > spans[:, 0]) & (spans[:, 1] <= positions.shape[1])
    after_visual = valid_span[:, None] & (positions >= spans[:, 1:2])
    attention = batch.attention_mask
    response = batch.response_mask & attention
    shifted_any = torch.zeros_like(attention)
    shifted_any[:, :-1] = attention[:, 1:]
    if scope == "response":
        # Keep the original mask composition so default runs are bit-identical.
        mask = batch.write_mask & response & attention
        mask = mask & after_visual
        shifted = torch.zeros_like(mask)
        shifted[:, :-1] = (batch.labels[:, 1:] != -100) & attention[:, 1:]
        mask &= shifted
    elif scope == "prompt":
        mask = (~batch.response_mask) & attention & after_visual & shifted_any
    else:
        prompt = (~batch.response_mask) & attention & after_visual & shifted_any
        mask = prompt | (batch.response_mask & attention & after_visual)
    if limit is not None:
        if limit < 1:
            raise ValueError("write_position_limit must be positive or null")
        for row in range(len(mask)):
            candidates = mask[row].nonzero().flatten()
            if len(candidates) > limit:
                chosen = torch.randperm(len(candidates), generator=generator)[:limit].to(mask.device)
                mask[row].zero_()
                mask[row, candidates[chosen]] = True
    return mask.nonzero()


def teacher_effect_scores(effects, pairs):
    """Cheap per-position ranking signal: mean teacher hidden-effect L2 norm."""
    if not pairs:
        raise ValueError("Effect sampling requires at least one layer pair")
    score = None
    for _, teacher_index in pairs:
        effect = effects[teacher_index]
        delta = (effect.u_plus - effect.u_minus).float()
        value = delta.square().sum(-1).sqrt()
        score = value if score is None else score + value
    return score / len(pairs)


def subset_effects(effects, indices):
    """Select rows from captured effects without re-running teacher hooks."""
    selected = {}
    for index, effect in effects.items():
        values = {}
        for field in fields(effect):
            value = getattr(effect, field.name)
            values[field.name] = value[indices]
        selected[index] = BlockEffect(**values)
    return selected


def stratified_write_positions(batch, effects, pairs, limit, generator=None, scope="response"):
    """Take equal random draws from low/mid/high teacher-effect strata per row.

    The hidden-state effect norm is used only as a ranking proxy; the actual
    write loss still probes the vocabulary head at the selected coordinates.
    ``write_position_limit`` is the per-example budget, so stratified selection
    requires it explicitly.
    """
    if limit is None or limit < 1:
        raise ValueError("stratified write sampling requires write_position_limit")
    candidates = write_positions(batch, None, generator, scope)
    if not len(candidates):
        return candidates, candidates.new_empty((0,), dtype=torch.long)
    if not effects:
        raise ValueError("stratified write sampling requires captured teacher effects")
    scores = teacher_effect_scores(effects, pairs)
    chosen_rows = []
    for row in candidates[:, 0].unique().tolist():
        indices = (candidates[:, 0] == row).nonzero().flatten()
        budget = min(limit, len(indices))
        if budget == len(indices):
            chosen_rows.append(indices)
            continue
        order = indices[torch.argsort(scores[indices], stable=True)]
        if budget < 3:
            perm = torch.randperm(len(order), generator=generator)[:budget].to(order.device)
            chosen_rows.append(order[perm])
            continue
        base, extra = divmod(budget, 3)
        chunks = torch.chunk(order, 3)
        picks = []
        for stratum, chunk in enumerate(chunks):
            count = base + (1 if stratum < extra else 0)
            count = min(count, len(chunk))
            if count:
                perm = torch.randperm(len(chunk), generator=generator)[:count].to(chunk.device)
                picks.append(chunk[perm])
        chosen_rows.append(torch.cat(picks))
    selected = torch.cat(chosen_rows)
    return candidates[selected], selected


def visual_message(attention, probabilities, projected_values, positions, spans):
    """Use the actual all-key softmax, including RoPE, GQA, masks, and dropout.

    The output projection's bias belongs wholly to the nonvisual term.
    No new attention normalization or attention routing is computed here.
    """
    batch, length, _ = projected_values.shape
    values = projected_values.view(batch, length, -1, attention.head_dim).transpose(1, 2)
    values = repeat_kv(values, attention.num_key_value_groups)
    messages = projected_values.new_zeros((len(positions), attention.o_proj.out_features))
    keys = torch.arange(length, device=projected_values.device)
    for row in positions[:, 0].unique().tolist():
        indices = (positions[:, 0] == row).nonzero().flatten()
        queries = positions[indices, 1]
        start, end = spans[row].tolist()
        visual = (keys >= start) & (keys < end) & (start >= 0)
        weights = probabilities[row][:, queries, :] * visual[None, None, :]
        heads = torch.matmul(weights, values[row]).transpose(0, 1).reshape(len(indices), -1)
        message = F.linear(heads, attention.o_proj.weight, bias=None)
        messages = messages.index_copy(0, indices, message)
    return messages


@dataclass
class BlockEffect:
    positions: torch.Tensor
    incoming: torch.Tensor
    o_total: torch.Tensor
    o_text: torch.Tensor
    o_vision: torch.Tensor
    u_plus: torch.Tensor
    u_minus: torch.Tensor


class EffectCapture:
    """Hooks never return replacement outputs. Keep this context around forward only.

    Backbone gradient checkpointing is deliberately unsupported in this eager
    reference: its backward recomputation would outlive the temporary hooks.
    Semantic-probe loss checkpointing is independent and supported.
    """
    def __init__(self, model, blocks, batch, positions=None):
        self.model, self.blocks, self.batch = model, list(blocks), batch
        self.positions = write_positions(batch) if positions is None else positions
        self.effects, self.handles, self.pending = {}, [], {}

    def __enter__(self):
        lm = self.model.language_model
        if getattr(lm.model, "gradient_checkpointing", False):
            raise ValueError("Effect capture requires backbone gradient checkpointing disabled")
        if lm.config._attn_implementation != "eager":
            raise ValueError("Effect capture requires attn_implementation='eager'")
        if getattr(lm.config, "use_sliding_window", False):
            raise ValueError("This correctness path supports full causal attention only")
        if hasattr(self.model, "merge_teacher_once") and self.model.load_decision["adapter_merges"] != 1:
            raise ValueError("Merge teacher adapters before effect extraction")
        for index in self.blocks:
            layer = lm.model.layers[index]
            if not isinstance(layer.self_attn, Qwen2Attention):
                raise ValueError("Effect capture currently supports transformers Qwen2Attention only")
        if not len(self.positions):
            return self
        for index in self.blocks:
            layer = lm.model.layers[index]
            self.handles.extend([
                layer.register_forward_pre_hook(self._before(index), with_kwargs=True),
                layer.self_attn.register_forward_pre_hook(
                    self._request_attention(layer.self_attn), with_kwargs=True),
                layer.self_attn.v_proj.register_forward_hook(self._values(index)),
                layer.self_attn.register_forward_hook(self._attention(index)),
                layer.register_forward_hook(self._after(index)),
            ])
        return self

    @staticmethod
    def _request_attention(attention):
        parameters = list(inspect.signature(attention.forward).parameters)
        flag_index = parameters.index("output_attentions") if "output_attentions" in parameters else None

        def hook(module, args, kwargs):
            if flag_index is not None and len(args) > flag_index:
                args = (*args[:flag_index], True, *args[flag_index + 1:])
            else:
                kwargs["output_attentions"] = True
            return args, kwargs
        return hook

    def _before(self, index):
        def hook(module, args, kwargs):
            hidden = args[0] if args else kwargs["hidden_states"]
            rows, queries = self.positions.unbind(1)
            self.pending[index] = {"incoming": hidden[rows, queries]}
        return hook

    def _values(self, index):
        def hook(module, args, output):
            self.pending[index]["values"] = output
        return hook

    def _attention(self, index):
        def hook(module, args, output):
            if not isinstance(output, tuple) or len(output) not in (2, 3):
                raise ValueError(f"Unsupported Qwen2 attention output at block {index}: expected 2 or 3 items")
            total, probabilities = output[:2]
            if probabilities is None:
                raise ValueError("Actual eager attention probabilities were not returned")
            rows, queries = self.positions.unbind(1)
            data = self.pending[index]
            data["o_vision"] = visual_message(module, probabilities, data.pop("values"),
                                               self.positions, self.batch.visual_spans)
            data["o_total"] = total[rows, queries]
        return hook

    def _after(self, index):
        def hook(layer, args, output):
            data = self.pending.pop(index)
            rows, queries = self.positions.unbind(1)
            text = data["o_total"] - data["o_vision"]
            baseline = data["incoming"] + text
            negative = baseline + layer.mlp(layer.post_attention_layernorm(baseline))
            # Read the real post-MLP output (including the final block before RMSNorm).
            self.effects[index] = BlockEffect(self.positions, data["incoming"], data["o_total"],
                                              text, data["o_vision"], output[0][rows, queries], negative)
        return hook

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.pending.clear()


def response_effect_scores(batch, effects, pairs):
    """Align teacher hidden-effect scores to every response position.

    Write positions are response positions of image examples; text-only rows
    have no visual effect to capture. Build a zero score for those positions
    and scatter the teacher scores onto the matching response coordinates.
    This keeps hidden-effect KD defined over the same positions as ordinary KD.
    """
    expected = batch.response_positions
    scores = expected.new_zeros((len(expected),), dtype=torch.float32)
    if not effects:
        return scores
    values = teacher_effect_scores(effects, pairs)
    positions = next(iter(effects.values())).positions
    if len(positions) == 0:
        return scores
    width = batch.input_ids.shape[1]
    keys = expected[:, 0].long() * width + expected[:, 1]
    query_keys = positions[:, 0].long() * width + positions[:, 1]
    indices = torch.searchsorted(keys, query_keys)
    if len(indices) and int(indices.max()) >= len(keys):
        raise ValueError("Teacher effect positions are not response positions")
    if not torch.equal(keys[indices], query_keys):
        raise ValueError("Teacher effect positions are not response positions")
    scores[indices] = values.to(device=scores.device, dtype=scores.dtype)
    return scores

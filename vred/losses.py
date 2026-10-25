"""Full-vocabulary semantic probes and FP32, valid-position reductions."""

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def _probe_parameters(model):
    """Detached norm and vocabulary head sources; no full FP32 head snapshot.

    The head is converted to FP32 chunk-by-chunk inside ``semantic_probe``, so
    the 151936 x hidden FP32 copy (about 1.67 GiB for the supplied models) is
    never materialized.  The returned head/bias are detached views and are not
    mutated.
    """
    norm, head = model.language_model.model.norm, model.language_model.get_output_embeddings()
    return (norm.weight.detach().float(), norm.variance_epsilon,
            head.weight.detach(), head.bias.detach() if head.bias is not None else None)


def semantic_probe(model, states, temperature=1.0, parameters=None, vocab_chunk_size=16384):
    if temperature <= 0 or vocab_chunk_size < 1:
        raise ValueError("Probe temperature and vocabulary chunk size must be positive")
    norm_weight, epsilon, head_weight, bias = _probe_parameters(model) if parameters is None else parameters
    # Qwen RMSNorm semantics, with detached parameters but differentiable states.
    # Qwen RMSNorm and the vocabulary projection are computed in FP32; the head is
    # converted to FP32 per vocabulary chunk instead of as one full-size copy.
    with torch.autocast(device_type=states.device.type, enabled=False):
        states = states.float()
        normalized = states * torch.rsqrt(states.square().mean(-1, keepdim=True) + epsilon)
        normalized = normalized * norm_weight
        chunks = []
        for start in range(0, head_weight.shape[0], vocab_chunk_size):
            stop = min(start + vocab_chunk_size, head_weight.shape[0])
            weight = head_weight[start:stop].detach().float()
            chunk_bias = bias[start:stop].detach().float() if bias is not None else None
            chunks.append(F.linear(normalized, weight, chunk_bias))
        logits = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=-1)
        return (logits / temperature).softmax(-1)


def signed_distribution(effect, tolerance=2e-5):
    effect = effect.float()
    if not torch.isfinite(effect).all():
        raise ValueError("Nonfinite semantic effect")
    mass = effect.abs().sum(-1, keepdim=True) / 2
    if (mass > 1 + tolerance).any() or (effect.sum(-1).abs() > tolerance).any():
        raise ValueError("Effect must be a difference of normalized distributions")
    remaining = (1 - mass).clamp_min(0)
    encoded = torch.cat((effect.clamp_min(0) / 2, (-effect).clamp_min(0) / 2, remaining), -1)
    if not torch.allclose(encoded.sum(-1), torch.ones_like(mass[..., 0]), atol=tolerance, rtol=0):
        raise ValueError("Signed effect encoding is not normalized")
    return encoded


def _kl_terms(target, prediction):
    # Clamp only log arguments; zero target mass contributes exactly zero and
    # has finite gradients at JS's zero/zero entries.
    tiny = torch.finfo(torch.float32).tiny
    return target * (target.clamp_min(tiny).log() - prediction.clamp_min(tiny).log())


def effect_loss(student, teacher, kind="smoothl1", vocab_reduction="sum", beta=1.0, eta=1e-6):
    """Return (configured loss, raw loss), averaged over positions.

    SmoothL1 raw averages vocabulary; KL/JS raw sum signed-distribution entries.
    Teacher targets are always detached, even when called outside no_grad.
    """
    if kind not in ("smoothl1", "effect_kl", "effect_js", "reverse_kl"):
        raise ValueError(f"Unknown write loss: {kind}")
    if vocab_reduction not in ("sum", "mean") or beta <= 0 or not 0 < eta < 1:
        raise ValueError("Invalid vocabulary reduction, SmoothL1 beta, or KL uniform mixture")
    student, teacher = student.float(), teacher.detach().float()
    if student.shape != teacher.shape or student.ndim != 2 or len(student) == 0:
        raise ValueError("Expected paired nonempty [positions, vocabulary] effects")
    if kind == "smoothl1":
        raw = F.smooth_l1_loss(student, teacher, beta=beta)
        return (raw * student.shape[-1] if vocab_reduction == "sum" else raw), raw
    qs, qt = signed_distribution(student), signed_distribution(teacher)
    if kind in ("effect_kl", "reverse_kl"):
        qs, qt = (1 - eta) * qs + eta / qs.shape[-1], (1 - eta) * qt + eta / qt.shape[-1]
        if kind == "effect_kl":
            raw = _kl_terms(qt, qs).sum(-1).mean()
        else:
            raw = _kl_terms(qs, qt).sum(-1).mean()
    else:
        middle = (qs + qt) / 2
        raw = (0.5 * _kl_terms(qt, middle).sum(-1) + 0.5 * _kl_terms(qs, middle).sum(-1)).mean()
    return raw, raw


def write_loss(student, teacher, student_effects, teacher_effects, pairs, chunk_size=32,
               temperature=1.0, kind="smoothl1", vocab_reduction="sum", beta=1.0,
               eta=1e-6, checkpoint_probes=False, vocab_chunk_size=16384):
    if chunk_size < 1 or vocab_chunk_size < 1 or not pairs:
        raise ValueError("Positive query/vocabulary chunk sizes and layer pairs are required")
    loss = next(student.parameters()).new_zeros((), dtype=torch.float32)
    raw = loss.detach().clone()
    if not student_effects and not teacher_effects:
        return loss, raw
    student_parameters, teacher_parameters = _probe_parameters(student), _probe_parameters(teacher)
    for student_index, teacher_index in pairs:
        se, te = student_effects[student_index], teacher_effects[teacher_index]
        if not torch.equal(se.positions, te.positions):
            raise ValueError("Teacher/student write coordinates differ")
        count = len(se.positions)

        def compute(sp, sm, tp, tm):
            ds = (semantic_probe(student, sp, temperature, student_parameters, vocab_chunk_size)
                  - semantic_probe(student, sm, temperature, student_parameters, vocab_chunk_size))
            with torch.no_grad():
                dt = (semantic_probe(teacher, tp, temperature, teacher_parameters, vocab_chunk_size)
                      - semantic_probe(teacher, tm, temperature, teacher_parameters, vocab_chunk_size))
            configured, unscaled = effect_loss(ds, dt, kind, vocab_reduction, beta, eta)
            return torch.stack((configured, unscaled))

        for start in range(0, count, chunk_size):
            stop = min(start + chunk_size, count)
            tensors = (se.u_plus[start:stop], se.u_minus[start:stop],
                       te.u_plus[start:stop].detach(), te.u_minus[start:stop].detach())
            if checkpoint_probes and torch.is_grad_enabled():
                values = checkpoint(compute, *tensors, use_reentrant=False)
            else:
                values = compute(*tensors)
            weight = (stop - start) / (count * len(pairs))
            loss = loss + values[0] * weight
            raw = raw + values[1].detach() * weight
    return loss, raw


def logits_kd(student_logits, teacher_logits, batch, temperature=1.0, chunk_size=32):
    if temperature <= 0 or chunk_size < 1:
        raise ValueError("Positive logits temperature and query chunk size are required")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("Final logit shapes differ")
    positions = batch.response_positions
    loss = student_logits.new_zeros((), dtype=torch.float32)
    for start in range(0, len(positions), chunk_size):
        rows, queries = positions[start:start + chunk_size].unbind(1)
        log_s = (student_logits[rows, queries].float() / temperature).log_softmax(-1)
        log_t = (teacher_logits[rows, queries].detach().float() / temperature).log_softmax(-1)
        terms = (log_t.exp() * (log_t - log_s)).sum(-1)
        loss = loss + terms.sum() * temperature**2 / len(positions)
    return loss


def logits_kd_sparse(student_logits, teacher_logits, temperature=1.0, chunk_size=32):
    """Final-logits KD over already-selected shifted response positions.

    ``VREDModel.forward_sparse`` returns [positions, vocabulary] logits; this is
    the same mean KL as ``logits_kd`` without a dense [batch, sequence, vocabulary]
    tensor.
    """
    if temperature <= 0 or chunk_size < 1:
        raise ValueError("Positive logits temperature and query chunk size are required")
    if student_logits.shape != teacher_logits.shape or student_logits.ndim != 2:
        raise ValueError("Expected paired [positions, vocabulary] sparse logits")
    if len(student_logits) == 0:
        return student_logits.sum() * 0
    loss = student_logits.new_zeros((), dtype=torch.float32)
    for start in range(0, len(student_logits), chunk_size):
        log_s = (student_logits[start:start + chunk_size].float() / temperature).log_softmax(-1)
        log_t = (teacher_logits[start:start + chunk_size].detach().float() / temperature).log_softmax(-1)
        loss = loss + (log_t.exp() * (log_t - log_s)).sum()
    return loss * temperature**2 / len(student_logits)


def effect_weighted_kd_sparse(student_logits, teacher_logits, teacher_minus_logits,
                              temperature=1.0, chunk_size=32):
    """Effect-weighted final-logit KD over sparse response positions.

    The teacher's token-level visual effect is
    ``w_t = ||softmax(z_t+ / T) - softmax(z_t- / T)||_1``.  This function returns
    ``mean_t(w_t * KL(p_t+ || p_s+))``; the weights are not normalized, so the
    raw term naturally shrinks when the teacher's visual effect is small.
    ``teacher_minus_logits`` must be produced by the same teacher with the
    visual features removed (zero features, same text/positions).
    """
    if temperature <= 0 or chunk_size < 1:
        raise ValueError("Positive logits temperature and query chunk size are required")
    if (student_logits.shape != teacher_logits.shape
            or student_logits.shape != teacher_minus_logits.shape
            or student_logits.ndim != 2):
        raise ValueError("Expected three paired [positions, vocabulary] sparse logits")
    if len(student_logits) == 0:
        return student_logits.sum() * 0
    loss = student_logits.new_zeros((), dtype=torch.float32)
    for start in range(0, len(student_logits), chunk_size):
        log_s = (student_logits[start:start + chunk_size].float() / temperature).log_softmax(-1)
        log_t = (teacher_logits[start:start + chunk_size].detach().float() / temperature).log_softmax(-1)
        log_m = (teacher_minus_logits[start:start + chunk_size].detach().float()
                 / temperature).log_softmax(-1)
        p_t = log_t.exp()
        p_m = log_m.exp()
        # Compute |p_t - p_m| in place on p_m and release the [chunk, vocab]
        # probability difference before building the KL term. This keeps the
        # peak temporary allocation at roughly one chunk instead of three.
        p_m.neg_().add_(p_t).abs_()
        weights = p_m.sum(-1).detach()
        del p_m
        terms = log_t.sub(log_s).mul_(p_t).sum(-1)
        loss = loss + (weights * terms).sum()
    return loss * temperature**2 / len(student_logits)


def floored_effect_weights(scores, floor=0.1):
    """Turn per-position teacher-effect magnitudes into mean-one KD weights.

    ``scores`` are detached nonnegative effect magnitudes aligned with the
    response positions. They are divided by their batch mean and floored:
    ``floor + (1 - floor) * score / mean(score)``. The mean weight is one, so
    the resulting KD term keeps the same overall scale as ordinary KD while
    low-effect tokens still retain ``floor`` of the weight. If every score is
    zero, every weight is the floor (there is no measurable visual effect to
    emphasize).
    """
    if not 0 <= floor < 1:
        raise ValueError("effect KD floor must be in [0, 1)")
    scores = scores.detach().float()
    if scores.ndim != 1:
        raise ValueError("Expected one effect score per response position")
    if len(scores) == 0:
        return scores
    mean = scores.mean().clamp_min(1e-12)
    return (floor + (1.0 - floor) * scores / mean).detach()


def weighted_kd_sparse(student_logits, teacher_logits, weights,
                       temperature=1.0, chunk_size=32):
    """Weighted final-logit KD over sparse response positions.

    Returns ``mean_t(weights[t] * KL(p_t || p_s))``. Weights are expected to
    average approximately one; unlike the logits-effect variant this keeps a
    controlled KD scale even when the teacher's visual effect is tiny.
    """
    if temperature <= 0 or chunk_size < 1:
        raise ValueError("Positive logits temperature and query chunk size are required")
    if student_logits.shape != teacher_logits.shape or student_logits.ndim != 2:
        raise ValueError("Expected paired [positions, vocabulary] sparse logits")
    if weights.ndim != 1 or len(weights) != len(student_logits):
        raise ValueError("Expected one KD weight per response position")
    if len(student_logits) == 0:
        return student_logits.sum() * 0
    weights = weights.to(device=student_logits.device, dtype=torch.float32).detach()
    loss = student_logits.new_zeros((), dtype=torch.float32)
    for start in range(0, len(student_logits), chunk_size):
        log_s = (student_logits[start:start + chunk_size].float() / temperature).log_softmax(-1)
        log_t = (teacher_logits[start:start + chunk_size].detach().float() / temperature).log_softmax(-1)
        terms = (log_t.exp() * (log_t - log_s)).sum(-1)
        loss = loss + (weights[start:start + chunk_size] * terms).sum()
    return loss * temperature**2 / len(student_logits)

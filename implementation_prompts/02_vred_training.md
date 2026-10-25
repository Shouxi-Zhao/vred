# 02 VRED: effects, losses, and stages

Use the foundation from prompt 01. Implement only the method specified in `project_background.md`; keep the ordinary model forward unchanged.

1. At selected decoder blocks, decompose the actual all-key attention output into `o_text` and `o_vision` using the original causal/padding softmax normalization, RoPE, GQA, projection, and bias. Define `b=h+o_text`, `v=o_vision`, `u_plus=b+v+MLP(N(b+v))`, and `u_minus=b+MLP(N(b))`. Continue the real model with `u_plus`; the negative branch is auxiliary. Start with an eager correctness path and add chunking/checkpointing only where measurements require it.
2. Implement the detached-parameter semantic probe `softmax(W N_final(u)/tau)` and `d=pi(u_plus)-pi(u_minus)`, using shifted assistant prediction positions. Implement `smoothl1`, `effect_kl`, and `effect_js` in FP32, with the signed `2V+1` encoding from the background and configurable vocabulary reduction. Teacher targets are detached; student states remain differentiable.
3. Add one trainer/configuration for DPT, SFT, and DFT. Use relative block pairs K=3 and K=6, valid-token reduction, BF16 model computation, and the stated CE/write/final-KL objectives. Freeze only what each stage specifies. Integrate DeepSpeed ZeRO-2 without changing legacy scripts.
4. Add numerical tests for attention decomposition, `u_plus` equality with ordinary forward, `v=0 => d=0`, finite/normalized losses, shifted positions, teacher no-grad, and gradients reaching the DPT connector and DFT decoder.

Avoid full-vocabulary probability tensors for all layers at once. If memory is insufficient, use query chunking or documented valid-position sampling; never top-k approximate the vocabulary. Do not add magnitude, affinity, handoff, CMAD, or CMAED losses.

Acceptance: CPU/tiny-model tests pass; a bounded one-batch DPT and DFT forward/backward completes if hardware permits; status records measured memory and remaining limits.

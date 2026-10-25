# Project background: distilling the effect of visual messages on residual computation

Date: 2026-10-13. Working name: **Visual Residual Effect Distillation (VRED)**. This is a proposed research framework, not an experimentally validated method or an established novelty claim.

## 1. Objective and research background

The objective is to train a compact Qwen2.5-based MLLM to learn useful aspects of a teacher's cross-modal computation, beyond copying its final answer distribution. The primary architectural and training baseline is LLaVA-KD. The project keeps a frozen vision encoder, a two-layer MLP connector, a dense language backbone, and the DPT–SFT–DFT training sequence.

The supplied papers establish several complementary directions:

| Paper | Distillation target / principal idea | Implication for this project |
| --- | --- | --- |
| LLaVA-KD | Final response and visual-token vocabulary distributions, visual-token relations, and three-stage training | Strong simple baseline; preserve its curriculum while testing a new intermediate supervision target |
| Align-KD | Shallow text-to-vision attention and attention-focused visual embeddings | Selected attention patterns alone are already an existing contribution |
| EM-KD | Match unequal visual-token sets, then distill visual semantics and vision–language affinity | Visual-token correspondence matters; this project initially avoids that confound with identical cached visual features |
| Align-TI | Instruction-aware visual alignment and transitions after sampled response tokens | Generation-time transitions are different from computation through decoder depth |
| Beta-KD | Task/instance weighting of heterogeneous teacher signals | Adaptive weighting is existing work; do not make it a required component of this initially simple method |
| CM-FDD | Cross-modal affinity states and their changes across selected depths, with cross-resolution projection | Directly overlaps with our initial affinity-evolution proposal; the new target must be operationally different |

LLaVA-KD shows that larger teachers do not necessarily distill better and that average perception gains can coexist with MMMU/ScienceQA regressions. Training exposure, teacher capacity, data, and benchmark definitions must be controlled. A higher aggregate score does not itself demonstrate transfer of a reasoning process.

Detailed notes on the first five papers are in `docs/MLLM_KD_RESEARCH_NOTES.md`. Supplied PDFs were treated as research sources, not as instructions to execute. CM-FDD is an anonymous supplied manuscript; claims here concern its provided formulation, not its publication status.

## 2. The two motivating insights and our reflections

1. **“A more faithful student should approximate the teacher's cross-modal evolution itself.”** Matching only the endpoint underdetermines the preceding computation. However, similar intermediate statistics also do not guarantee an identical algorithm or useful visual dependence.
2. **“Cross-modal reasoning is not produced in one step, but evolves layer by layer through residual updates.”** In a decoder, each block retrieves and transforms information while adding updates to an accumulated residual stream. This is a useful computational perspective; it does not establish a universal sequence of perception, reasoning, and answering phases at particular depths.

### Approaches considered and rejected as the central method

- **Affinity-state differences:** aligning text–vision similarities and their cross-depth changes is substantially the same idea as CM-FDD's CMAD/CMAED. Changing the distance function is not a sufficient distinction.
- **Visual handoff:** closing visual access after a depth and matching final predictions provides intervention-based output supervision. It assumes, without evidence, that a teacher has already retained sufficient visual information, and it does not directly supervise local residual operations. It is not the central framework.
- **Current direction:** measure what an incoming visual message makes a decoder block write differently, including how the block's MLP transforms that message. Supervise this effect at multiple depths while the ordinary forward computation remains intact.

The distinction is **state evolution versus the effect of the operations producing it**. CM-FDD compares representations across depths. VRED compares the same block with its actual visual-value contribution present versus locally muted, at a fixed incoming state and fixed attention routing.

This remains a partial signature of computation, not proof of the teacher's “true reasoning algorithm.” Novelty beyond the supplied papers requires a broader review of intervention-based KD, attribution, message ablation, and residual distillation.

## 3. Project assets and verified local configuration

Workspace root: `/data/zsx/LLaVA-KD`.

| Component | Project setting | Local inspection |
| --- | --- | --- |
| Teacher | SigLIP2-so400m-patch14-384 + two-layer GELU MLP(mlp2x_gelu) + Qwen2.5-3B | Teacher config specifies 1152-dimensional vision features and 2048-dimensional language states |
| Teacher checkpoint | `checkpoints/ta-qwen25-3b-stage0_2-lora-finetune` | Contains root LoRA adapter, language-model weights, connector weights, and configuration |
| Student | Same SigLIP2 encoder + two-layer GELU MLP(mlp2x_gelu) + Qwen2.5-0.5B | Base config has hidden width 896 |
| Base language models | `pretrained_hg/Qwen2.5-3B`, `pretrained_hg/Qwen2.5-0.5B` | Teacher/student decoder depths are 36/24; vocabulary width is 151936 in both base configs |
| Vision weights | `pretrained_hg/siglip2-so400m-patch14-384` | Vision config has width 1152, image size 384, patch size 14 |
| DPT annotations | `mllm_datasets/text_files/blip_laion_cc_sbu_558k.json` | File exists |
| SFT/DFT annotations | `mllm_datasets/text_files/llava_v1_5_mix665k_clean.json` | File exists; use the clean file, not the older mix filename |
| Cached vision features | `mllm_datasets/vision_cache/siglip2-so400m-patch14-384` | Index has 906803 entries; an inspected `.npy` feature has shape `(729, 1152)` and dtype FP16 |
| Benchmarks | `eval_dataset` | Actual benchmark subdirectories are under `eval_dataset/eval/` |
| Hardware | One node, six RTX 4090 GPUs, DeepSpeed ZeRO-2 | Requested training target; no training-memory feasibility result yet |

### Connector consistency and teacher loading

The local implementation in `llavakd/model/connector/mlp.py` and teacher checkpoint establish the intended connector structure:

`Linear(vision_width, language_width, bias=True) → GELU → Linear(language_width, language_width, bias=True)`.

Verified teacher connector tensors are `(2048, 1152)` and `(2048, 2048)` with biases. The student uses the same structure with `(896, 1152)` and `(896, 896)`. Each model has its own connector; the teacher's connector is frozen, and the student's is trained according to stage.

The teacher directory contains both `language_model/pytorch_model.bin` and a root `adapter_model.safetensors`. Adapter keys address the multimodal wrapper's language modules. The adapter config reports LoRA rank 256 and alpha 512. This does not establish whether the saved language-model weights are base, partially adapted, or already merged. Before use, determine the checkpoint format and apply/merge the adapter exactly once if needed. Compare predictions before/after an explicitly documented merge against the intended teacher. Do not silently omit the adapter or apply it twice. Merge on CPU when practical, then freeze/evaluate the resulting teacher. This also simplifies attention-source decomposition and semantic probes.

Equal base vocabulary sizes do not prove equal token-ID semantics. Validate checkpoint tokenizer files, added tokens, vocabulary mapping, BOS/EOS/PAD, and chat templates on teacher/student before any effect or final-logit KD.

## 4. Core methodology

### 4.1 Exact local visual-message decomposition

At a selected pre-normalization decoder block, let `H` denote all incoming residual states and `h_i` the incoming residual vector at a selected nonvisual prediction position `i`.

Attention computes Q/K/V from `N_attn(H)`, applies RoPE and the normal causal/padding mask, and uses the usual softmax over all allowed keys. For a head, decompose its output by source positions:

`a_i = sum_j A_ij V_j = a_i,nonvisual + a_i,visual`.

After head concatenation and the attention output projection, write:

`o_i = o_i,text + o_i,vision`.

“Text” means all nonvisual positions, including special tokens. If the output projection has a bias, include it exactly once in the nonvisual term. Apply any active LoRA transformation consistently; a correctly merged frozen teacher makes this easier. Qwen's grouped-query attention requires correct repetition/grouping of KV heads before summing head-specific messages.

Use the original attention normalization across all allowed keys. **Do not softmax over visual keys alone**, remove image positions from the sequence, or recompute attention after mutation. Those would measure a different intervention.

Define for each selected position:

`b = h + o_text`, `v = o_vision`.

`u_plus  = b + v + MLP(N_ffn(b + v))`.

`u_minus = b     + MLP(N_ffn(b))`.

`u_plus` is the actual block output before the next block/final normalization. `u_minus` is an auxiliary local measurement. Continue the model using **only `u_plus`**. The local negative branch does not replace states in downstream layers.

The residual effect is:

`g = u_plus - u_minus = v + [MLP(N_ffn(b+v)) - MLP(N_ffn(b))]`.

It includes the direct visual write and the subsequent normalization/MLP-mediated interaction. Incoming states can already contain earlier visual information; this is the incremental contribution of the current direct visual-value message, not all image-dependent computation.

### 4.2 Semantic readout and effect target

For each model `m`, use its existing final RMSNorm and vocabulary head as a fixed-parameter semantic probe:

`pi_m(u) = softmax(W_m N_final,m(u) / tau_write)`.

`d_m = pi_m(u_plus) - pi_m(u_minus)`.

At an intermediate block this is a logit-lens proxy, not a calibrated next-token prediction. Early-layer suitability must be validated. Applying final normalization to each branch here does not change the ordinary decoder forward path.

Teacher parameters and all teacher targets are frozen. For the student's write loss, use functional probes with detached norm/head parameters while preserving gradients with respect to `u_plus` and `u_minus`. Do not detach the student states, put the whole student probe in `no_grad`, or freeze the student's head globally: ordinary CE/final KD must still train the normal language model. Tied embeddings need particular care.

Use both branches with their natural student gradients. Allow gradients through `b`, `v`, attention routing, and the MLP where trainable. Stop-gradient through `b` is not part of the default method; the effect also depends on the context in which the message is processed.

For the numerical example discussed earlier:

| Quantity | Teacher | Student |
| --- | --- | --- |
| `u_plus` | `[2, 2]` | `[2.264911, 1.132456]` |
| `u_minus` | `[2.414214, 0]` | `[2.414214, 0]` |
| `pi(u_plus)` over dog/cat/other | `[0.422319, 0.422319, 0.155362]` | `[0.551405, 0.292953, 0.155642]` |
| `pi(u_minus)` | `[0.672842, 0.163579, 0.163579]` | Same in this toy example |
| `d` | `[-0.250523, +0.258740, -0.008217]` | `[-0.121437, +0.129374, -0.007937]` |

The probe used unit-gain zero-epsilon RMSNorm, a toy ReLU MLP, head rows `(1,0)`, `(0,1)`, `(0,0)`, and temperature 1. This is an illustration, not the real Qwen gated-MLP architecture.

### 4.3 What the effect does and does not mean

`d` is signed and sums to zero. Positive entries are probability increases; negative entries are decreases under the probe. Neither sign establishes semantic correctness or relevance. A teacher can strengthen an incorrect answer, and softmax coupling can reduce other probabilities without reducing their logits. Supervised CE anchors the model to the data, but does not guarantee every teacher effect is beneficial.

The method supervises repeated local cross-modal operations. It does not directly reproduce an affinity trajectory or guarantee the teacher's entire reasoning algorithm. Direct visual-value effects omit routing effects already present in the fixed attention weights and image information relayed through nonvisual positions.

The user has deferred magnitude-loss investigation. **Do not add a residual-norm, effect-strength, or magnitude-restoration loss in this project configuration.** The distributional encodings below preserve the observed probability-effect strength without attempting to recover hidden-state magnitude removed by normalization.

### 4.4 Valid positions and causal alignment

Supervise positions predicting valid assistant-response tokens, not the teacher-forced answer token's own position. For a response label at expanded sequence position `t`, use the hidden state at `t-1`, consistent with causal CE and final logits KD. This includes the prompt-end state predicting the first answer token. Avoid unshifted answer-position supervision that already sees the token being predicted.

Build masks after visual-placeholder expansion, padding, and truncation. Exclude padding, ignored labels, invalid/absent image spans, and positions with no causal access to relevant preceding visual tokens. Both models use the same textual prefix. Pair corresponding response predictions by explicit metadata, not merely flattened tensor shapes. Text-only examples still contribute CE/final KD; they contribute no write loss.

Do not force a visual-prefix token to encode a later instruction: the causal mask forbids this. Instruction-conditioned visual integration is observed on the nonvisual side.

## 5. Three selectable write losses

Expose `write_loss_type = smoothl1 | effect_kl | effect_js`. Keep all three options. The initial reference uses SmoothL1 to establish continuity with the proposal; select the final research loss using controlled validation experiments.

For all modes, compute losses in FP32, detach teacher targets, and average over selected layer pairs and valid prediction positions. Vocabulary reductions differ as specified below. Log raw and weighted values separately.

### 5.1 SmoothL1

For each signed component error `e_j = d_S,j - d_T,j`,

`smooth_beta(e) = e^2/(2 beta)` if `abs(e)<beta`, otherwise `abs(e)-beta/2`.

Retain the previously discussed raw definition:

`L_write_raw = mean_layers,positions,vocabulary smooth_beta(d_S-d_T)`.

Start with `beta=1`, configurable. In the toy example the raw mean is approximately `0.00556644`. At this threshold most probability errors fall in the quadratic region, so do not claim a strong robustness advantage over MSE without evidence.

The real vocabulary has 151936 entries. To avoid silently diluting the loss across that axis, the recommended training convention is `L_write = V * L_write_raw`, equivalent to summing the per-vocabulary-component losses before averaging positions/layers. Keep an explicit `write_vocab_reduction` option (`sum` recommended, `mean` retained for reproducibility). This is a fixed vocabulary reduction, not a new magnitude-loss term. Record the convention with every coefficient and result; changing it changes the meaning of `lambda_write`.

### 5.2 Effect-KL and Effect-JS through a signed distribution

Ordinary KL cannot be applied directly to `d`. Construct a probability vector with `2V+1` entries:

`q_up,j = max(d_j, 0)/2`.

`q_down,j = max(-d_j, 0)/2`.

`q_remaining = 1 - sum_j abs(d_j)/2`.

`q(d) = concat(q_up, q_down, q_remaining)`.

Since `d` is a difference of distributions, `sum(d)=0` and `||d||_1/2 <= 1`. The representation is valid, and `d_j = 2(q_up,j-q_down,j)`. Do not independently normalize positive/negative effects: that would erase their probability-redistribution strength.

Effect-KL is teacher-to-student:

`L_effect-KL = mean_layers,positions KL(q(d_T) || q(d_S))`.

Effect-JS is:

`M = (q(d_T)+q(d_S))/2`.

`L_effect-JS = mean_layers,positions [0.5 KL(q(d_T)||M) + 0.5 KL(q(d_S)||M)]`.

Sum over the signed-distribution dimension; do not divide a divergence by `2V+1`. In the toy example, raw KL is approximately `0.061279` and JS `0.013812` nats.

For KL, zero student mass at a teacher-positive component requires smoothing. Use a configurable uniform mixture `q_eta=(1-eta)*q+eta/(2V+1)`, initially `eta=1e-6`, for both sides; compute logs in FP32. This fixes total smoothing mass rather than adding an epsilon per coordinate that grows with vocabulary size. For JS, use safe zero-mass handling (`0 log 0 = 0`); the divergence itself needs no smoothing to remain finite. Clamp only small negative remaining masses caused by roundoff and verify normalization. Report smoothing in results.

The positive/negative split is piecewise differentiable and has a kink at zero. Check its zero-region gradients and sign-flip behavior in numerical validation; finite JS does not by itself ensure useful learning when student effects collapse to zero.

### 5.3 Final logits KD

`L_logits = tau_logits^2 * mean_valid_positions KL(softmax(z_T/tau_logits) || softmax(z_S/tau_logits))`.

Use ordinary final response-prediction logits, matching shifted labels and token identities. Sum vocabulary components before averaging valid positions. Start with `tau_logits=1`; initialize `tau_write=1` independently. The visual-effect probe is not a replacement for this normal-output anchor.

## 6. Recommended stage composition

Adopt the user's proposed structure, with explicit coefficients rather than assuming heterogeneous losses have equal scales:

| Stage | Objective | Student trainable components | Write supervision |
| --- | --- | --- | --- |
| DPT | `L_CE + lambda_write_DPT L_write` | Connector only; LLM frozen | `K=3` matched block effects |
| SFT | `L_CE` | Connector + full student LLM | None; no teacher needed |
| DFT | `L_CE + lambda_write_DFT L_write + lambda_logits L_logits + lambda_effect_kd L_ewkd` (last term optional) | Connector + full student LLM | Initially `K=6`, configurable |

Teacher remains frozen/eval throughout. Frozen vision features are used throughout. DPT follows the frozen student's differentiable operations so that CE/write gradients reach the connector; “frozen LLM” must not mean running the student's LLM under `no_grad`.

**Why this choice:** DPT trains the input interface to elicit appropriate visual-message effects through a fixed language backbone. SFT establishes ordinary instruction-following competence. DFT can then modify the decoder's local operations while final logits KD anchors its output behavior. DPT cannot itself train the frozen decoder to implement new operations, so gains there should be interpreted as interface alignment.

Do not add final logits KD, LLaVA-KD visual-logit KD, RDist, CMAD, or CMAED to the default DPT. Do not add those visual/relation losses to default DFT. They would confound attribution. The original LLaVA-KD losses remain in the comparator, not in the new default objective.

Initial coefficients: `lambda_write_DPT=1`, `lambda_write_DFT=1`, `lambda_logits=1`, with the vocabulary-sum SmoothL1 convention above. These are provisional starting values, not validated optima. On a fixed small training/validation pilot, compare `lambda_write` in `{0.1,1,10}` for each loss family and inspect connector/decoder gradient norms. Keep CE and final KD coefficients fixed initially. Do not choose weights by equalizing total scalar losses alone, and do not evaluate hyperparameters on benchmark test sets. Adopt constant coefficients first; a curriculum is not a required additional mechanism.

The optional DFT effect-weighted KD adds
`L_ewkd = mean_t(w_t * KL(p_t+ || p_s+))` with
`w_t = ||p_t+ - p_t-||_1`, where `p_t+`/`p_t-` are the image-conditioned and
zero-visual-feature teacher distributions and `p_s+` is the student's
image-conditioned distribution. It is disabled by default (`--effect-weighted-kd`
enables it). When enabled, the ordinary KD should remain a small anchor rather
than being kept at the same scale: start with `lambda_logits=0.2` and
`lambda_effect_kd=1.0` (1:5). The effect weights are not normalized and are
usually below 1, so a coefficient of 1 already makes the effect term primary;
`lambda_effect_kd=2` is not the default recommendation and is appropriate only
when the raw `effect_kd / logits_kd` ratio is very small. Compare weighted
contributions (`0.2 * logits_kd` versus `1.0 * effect_kd`) rather than raw losses
alone.

A second, cheaper weighting variant is available with
`--effect-weighted-kd --effect-kd-weight-source hidden --effect-kd-floor 0.15`.
It reuses the teacher's existing plus effect capture instead of running a second
teacher forward. For response token `t`, define
`s_t = mean_pair ||u_(l,t)+ - u_(l,t)-||_2` and
`w_t = floor + (1-floor) * s_t / mean(s)`; then
`L_ewkd = mean_t(w_t * KL(p_t+ || p_s+))`. Text-only positions receive the
floor. For nonzero effects the weights average one, so the coefficient has a
controlled scale; start with `lambda_logits=0` and
`lambda_effect_kd=0.5` rather than reusing the unnormalized logits-mode value of
1.0. Hidden mode requires full response write positions (response scope, no
position limit, uniform sampling) so its weights align with every KD position.
It removes the teacher-minus forward and therefore reduces the dominant L_ewkd
training overhead.

Use one epoch per stage initially. SFT and DFT reuse the same clean instruction dataset; log actual example/token exposures and optimizer steps. Save and explicitly reload DPT → SFT → DFT checkpoints, with descriptive paths rather than hard-coded previous experiment names.

### Relative-depth pairing

Define `K` relative depths `r_k=k/K`, for `k=1,...,K`. For model depth `L_m`, select zero-based block index `ell_m,k=ceil(r_k L_m)-1`. These refer to post-MLP block outputs before final normalization, not the potentially final-normalized entry in a library's `hidden_states` tuple. Hook actual decoder blocks.

With verified depths `L_T=36`, `L_S=24`:

| Stage | Relative depths | Teacher indices | Student indices |
| --- | --- | --- | --- |
| DPT, K=3 | 1/3, 2/3, 1 | 11, 23, 35 | 7, 15, 23 |
| DFT, K=6 | 1/6, 2/6, 3/6, 4/6, 5/6, 1 | 5, 11, 17, 23, 29, 35 | 3, 7, 11, 15, 19, 23 |

Averaging over K prevents simply multiplying total supervision strength when adding blocks. Pairing equal relative depths is a baseline heuristic, not proof of equal functional roles. DFT sensitivity experiments use `K=3,6,12`; K=6 is a practical first choice for 24-GB devices, not a known optimum.

## 7. Offline feature dataset requirements

**Training must not instantiate or load either vision encoder on GPU.** Supply frozen cached encoder features directly to each model's own connector. Do not cache connector outputs: the student's connector changes during training.

The inspected cache manifest reports FP16, feature layer `-1`, `.npy` storage, expected patch count 729, 558128 existing index entries, and 348675 newly requested instruction-image entries, totaling 906803. Its `data_paths` lists only the latest instruction-caching run. The cache script merges indexes but overwrites the manifest for each run, so the manifest alone does not prove that both annotation files have complete coverage.

Dataset contract for future implementation:

1. Read annotation conversations and retain explicit image identifiers. Load the shared cache index once per dataset/worker strategy; avoid uncontrolled copying of its large mapping.
2. Normalize keys exactly as `cache_vision_features.py`: replace backslashes with slashes and apply `.lstrip("./")`. Resolve the mapped file relative to the cache root. Check path normalization collisions and prohibit mappings escaping the root.
3. Load `.npy` features on CPU with `allow_pickle=False`; optional memory mapping/LRU caching must be bounded. Collate variable visual spans explicitly and use pinned-memory transfer where supported.
4. Validate shape `(729,1152)`, finite values, FP16 storage, encoder identity, feature layer, and preprocessing provenance. Share the same raw feature values with teacher and student; project separately. Cast at connector execution to the model's compute dtype.
5. Preserve all 729 patch tokens. Do not inherit the older unconditional `[:,1:]` behavior or a hard-coded 728-token slice. SigLIP2 patch outputs here do not require removal of a CLIP-style CLS token.
6. Return `vision_features`, visual-span metadata, expanded attention/label masks, and image-present flags. Handle text-only samples without an encoder or fabricated visual spans. Unsupported multiple-image records must be detected explicitly rather than treated as single images.
7. Check coverage against both annotation files before a long run, with counts for missing keys, missing files, duplicate image IDs, and malformed features. Do not silently fall back to GPU vision encoding or silently skip failed records.
8. Use the exact preprocessing/template and sequence truncation for both models. Recompute positions and labels after visual insertion, and ensure there are valid response targets left after inserting 729 visual tokens.

The existing `llavakd/data/dataset.py` is an image-loading pipeline and must not be assumed to meet this new contract. This document specifies future implementation; it does not modify dataset code. The sample feature was inspected, but a full cache/file-integrity audit has not been run.

Training-cache availability also does not establish evaluation-cache availability. For evaluation, either build matching caches ahead of time or load the vision encoder in a separate evaluation process. Image perturbation tests require freshly encoded/cached altered images or explicitly labeled feature-space interventions; do not reuse an unchanged feature cache after changing pixels.

## 8. One-node, six-4090 ZeRO-2 execution design

Use one process per GPU and DeepSpeed ZeRO stage 2 for the trainable student. Replicate one frozen BF16 teacher language model and connector per rank, without an optimizer or teacher gradient state. ZeRO-2 shards student optimizer state/gradients, not model parameters or activation memory; six devices do not provide a single shared 144-GB memory pool. Frozen vision weights remain off GPU.

Provisional launch/configuration parameters:

| Setting | DPT | SFT | DFT |
| --- | --- | --- | --- |
| Precision | BF16; FP32 probability/loss calculations | BF16 | BF16; FP32 probability/loss calculations |
| Student microbatch per GPU | 1 | 1 | 1 |
| Gradient accumulation | 32 | 16 | 16 |
| Effective global batch | 192 | 96 | 96 |
| Initial learning rate | 1e-3 | 2e-5 | 2e-5 |
| Epochs | 1 | 1 | 1 |
| Optimizer | AdamW, weight decay 0 | Same | Same |
| Schedule / warmup | Cosine / 0.03 | Same | Same |
| Max expanded sequence length | 2048 | 2048 | 2048 |
| Student gradient checkpointing | Enabled, including frozen-LLM path where needed | Enabled | Enabled |
| Teacher mode | Eval, no gradients, use_cache=False | Not loaded | Eval, no gradients, use_cache=False |

Global batch is `world_size × microbatch × accumulation`; these proposed values intentionally differ from the paper's 256/128 batches because six GPUs with microbatch 1 cannot realize those exactly with integer accumulation. Controls must use the same realized batch. Do not change the encoder feature resolution to resolve memory pressure.

Use the ZeRO-2 configuration in `scripts/zero2.json` as a starting point, explicitly confirming BF16 is enabled, FP16 disabled, stage 2 selected, and resolved batch fields logged. An example future launcher is `deepspeed --include localhost:0,1,2,3,4,5 ...`; no new runnable training entry exists yet.

### Attention extraction and activation memory

Do not request and retain full attention matrices for every layer. Ordinary FlashAttention/SDPA often does not expose source-separated messages directly. Exact visual-message extraction at selected blocks needs the original full-key softmax denominator, correct RoPE, GQA, padding/causal masks, and matching dropout behavior.

A reasonable implementation uses fused ordinary attention plus chunked recomputation of selected query rows over all allowed keys to obtain `o_vision`, then `o_text=o_total-o_vision`. Recomputing over all allowed keys is essential: softmax on just visual keys changes the method. It must remain differentiable for the student. Alternatively, selected-block eager attention is a correctness reference, not an assumed production solution.

Reuse the actual `u_plus`; evaluate the negative FFN branch only for supervised positions. Process vocabulary-head outputs in small query chunks, computing the full-vocabulary normalizer and exact loss. Do not store K pairs of full `(batch,sequence,vocabulary)` probabilities. Gradient checkpointing/custom recomputation is needed to limit retained student autograd activations; forward chunking alone does not guarantee that backward memory is small. The signed-distribution losses can be accumulated without materializing a persistent `2V+1` tensor.

Initial resource target is K=6 with all valid prediction positions. If exact processing still exceeds memory, uniformly sample at most 32 valid prediction positions per example for write supervision (same positions in teacher/student), retaining full CE/final logits KD. Record this as a sampled-estimator configuration and test against all-position supervision on short sequences. Do not silently use a top-vocabulary approximation.

Measure peak allocated/reserved VRAM, iteration time, valid tokens/sec, and GPU-hours at max-length batches before a full run. Lower write query-chunk size, use recomputation, or reduce a documented sampling budget before changing scientific settings. CPU optimizer offload is a separately reported ZeRO-2 fallback if needed. Six-4090 feasibility remains unverified.

## 9. Experiments to test the core method

Experiments are planned, not executed. Use increasing cost tiers so failure of a core premise is discovered before full training.

### E0. Numerical and implementation correctness

Use tiny tensors and short real-model batches to establish:

- `o_total ≈ o_text+o_vision`, with bias/LoRA applied once and the original attention denominator preserved.
- Ordinary block output equals `u_plus`; auxiliary branches do not change the standard forward output.
- Setting `v=0` gives `u_plus=u_minus`, `d=0`, and a correctly normalized `q` with all mass in its remaining entry.
- Correct causal label shift, padding/image masks, GQA/rotary handling, and mixed text/image records. No flattened shape-only alignment.
- Teacher has no gradients; student's write gradients reach the connector in DPT and decoder/connector in DFT. Detached probe weights still permit state gradients, including tied-head configurations.
- KL smoothing/sign reversals, JS zeros, and SmoothL1 produce finite values and nontrivial gradients for relevant cases.
- Exact cached-vs-online features agree within the documented storage precision on a small image subset. No vision encoder is resident during the training-mode check.

Use FP32 as a reference and documented BF16 tolerances. Stop implementation work on the training objective until these invariants pass.

### E1. Is the local semantic effect meaningful?

Before student training, use a fixed held-out set of about 512 image–question examples, stratified across scene understanding, OCR, and reasoning, outside the optimization split. At the selected teacher blocks:

1. Measure `d_T`, its signed redistribution, and its relationship to answer-content tokens versus formatting tokens. Report block-wise variability and degenerate effects.
2. Propagate the same local message mutation through the remainder of the teacher for a diagnostic-only run. Compare downstream answer log-odds/NLL changes with local probe effects. The training method still uses a local branch only.
3. Compare visual-message interventions with norm-matched random message changes and matched nonvisual-message changes. Random controls must be labeled as controls, not valid visual-counterfactual data.
4. Repeat on matched questions about different objects in the same image and on valid altered-image/question pairs with known answers. Measure whether the effect changes with relevant evidence.

Evidence sought: useful block-specific semantic effects, not merely confidence changes or arbitrary perturbation sensitivity. If intermediate readouts are uninformative or do not predict downstream consequences, do not claim process transfer; reconsider the probe before scaling.

### E2. Does the proposed DPT improve the trainable interface?

Train two DPT variants from the same base student and connector initialization:

| Variant | DPT | Subsequent SFT |
| --- | --- | --- |
| P0 | CE only | Identical CE-only SFT |
| P1 | CE + write, K=3 | Identical CE-only SFT |

Use the same caption exposure, batch, and seed. First screen on a fixed training subset; assess final candidates using the full DPT set. Measure post-DPT held-out caption CE, effect agreement, and nonzero connector gradient contributions, then evaluate both post-SFT checkpoints. DPT improvement must survive SFT to justify this component. Include longer CE-only pretraining as a compute-budget control for a promising P1.

### E3. Does local write supervision add value beyond final logits KD?

From an identical saved SFT checkpoint, run a 2×2 DFT factorial:

| Variant | CE | Final logits KD | Write loss, K=6 |
| --- | --- | --- | --- |
| F0 | Yes | No | No |
| F1 | Yes | Yes | No |
| F2 | Yes | No | Yes |
| F3 | Yes | Yes | Yes |

F0 has the same additional instruction-data pass as the other variants. F3 versus F1 isolates write supervision; F2 versus F0 tests standalone contribution; F3 versus F2 tests the final-output anchor. Measure final benchmark performance, held-out effect agreement, and compute cost. Improved training-target agreement without a held-out behavioral gain is insufficient.

After selecting a DFT configuration, compare complete CE-DPT→SFT→F3 and write-DPT→SFT→F3 pipelines. Combined with E2, this separates initial interface benefits from decoder fine-tuning benefits.

### E4. Which part of the visual effect matters?

Fix SFT initialization, paired depths, query positions, and budgets. Compare:

| Variant | Target | Question |
| --- | --- | --- |
| Full local effect | `pi(u_plus)-pi(u_minus)` | Proposed target |
| Direct message only | `pi(b+v)-pi(b)` | Is the FFN-mediated transformation necessary? |
| Positive branch snapshot | `pi(u_plus)` matched by forward KL | Is benefit just intermediate prediction matching? |
| Both branch snapshots | KL on both plus/minus distributions | Is subtracting baseline context beneficial? |
| Shuffled teacher effect | Effects paired to different examples, preserving approximate depth/task distribution | Does input-specific teacher computation matter? |
| Shuffled depth correspondence | Teacher block effects permuted across pairs | Does depth placement matter beyond generic regularization? |

These controls should match auxiliary computation or separately report its cost. Loss scales must receive comparable validation tuning; a weaker arbitrary coefficient is not a fair control. Shuffled controls are expected failure tests, not presumed results.

### E5. Select the write loss and supervision density

At fixed K=6, compare SmoothL1, effect-KL, and effect-JS under E3's F3 objective. Give each the same small validation tuning budget, using its declared reduction/smoothing. Track sign agreement on non-negligible effects, held-out JS in a common diagnostic representation, final task scores, gradient norms, nonfinite rates, and runtime. Do not rank different modes by incomparable raw training-loss values.

With the selected loss, compare DFT K=3,6,12 and final-block-only K=1. Keep per-layer averaging and constant exposure. Multi-depth benefit should exceed final-block-only and survive a compute-matched control. If sampling is needed, test 8/32/all write positions on short batches before choosing a resource budget.

### E6. End-to-end comparators and process-sensitive evaluation

Compare the selected method against:

- Three-stage CE-only training and CE + final-logits KD controls.
- LLaVA-KD with its actual MDist/RDist losses, using the same SigLIP2/Qwen2.5 assets and clean data. Label this as an adapted controlled baseline, not a reproduction of its original published numbers.
- CM-FDD reimplemented/loaded with the same teacher/student, resolution, dataset, exposure, and evaluation protocol after its reference source is accessible. Existing 1.5B checkpoints do not substitute for the requested 0.5B comparison.

Report both matched data exposure and, for the strongest baselines, approximately matched total GPU-hours. Longer baseline SFT/KD controls must not access additional examples unavailable to the proposed method. Include extraction/negative-branch/probe overhead and preprocessing costs.

Beyond normal accuracy, evaluate image/question dependence, correct evidence changes, wrong-image controls, and local-message intervention response on held-out examples. A reduction in visual sensitivity alone is ambiguous: it can reflect useful retention, ignored images, or reliance on language priors. Read it together with grounded accuracy and matched input changes. Full autoregressive generation diagnostics must not receive gold answer prefixes.

## 10. Benchmark protocol, statistics, and success criteria

All ten benchmark directories are present under `eval_dataset/eval/`: VQAv2, GQA, VizWiz, ScienceQA, TextVQA, MME, MMBench, MMBench-CN, POPE, and MMMU. Verify files/splits and external scoring requirements before claiming they are ready to run; directory presence alone is not a completeness audit. For GQA and VQAv2, find local files to compute the metircs.

Pilot locally scoreable tasks may use GQA, ScienceQA-IMG, TextVQA, MME-P, POPE, and MMMU validation. Final results must cover all ten using the same harness, template, split/version, decoding settings, OCR-input policy, and scoring method. MMBench legacy evaluation and server-scored tasks require their documented submission/evaluation paths. 

Report per-task scores, a clearly defined ten-task average, and perception/OCR/reasoning/hallucination breakdowns. Normalize MME-P by dividing raw score by 20 where constructing the comparable average; define every other component explicitly. Also optionally report LLaVA-KD's Avg7 excluding VQAv2, POPE, and MMMU. Do not compare six-task, seven-task, and ten-task means as if they were identical.

Hold out a fixed source-aware validation split from training data for loss weights, density, and checkpoint selection. Deduplicate/group images across train/validation to reduce leakage and keep that split fixed for controls. Never select hyperparameters using benchmark test results. Report any source overlap audit and exact clean-data record/image counts.

Use one seed for feasibility screening and at least three seeds for the selected method and key controls, provisionally 1234/2345/3456. Report mean/std and paired image/question-group bootstrap intervals where appropriate. Use percentage points for absolute gains and percentages for relative gains.

Proceed to a process-transfer claim only if:

1. Implementation invariants and teacher-loading/cache checks pass.
2. Local probe effects relate to downstream behavior on held-out examples.
3. Write supervision improves over the same-stage final-KD baseline, with compute controls.
4. Actual visual effects and correct depth/input pairing outperform simpler/shuffled controls.
5. Gains are not explained solely by generic intermediate prediction matching or lost image dependence.
6. Reasoning regressions, especially MMMU/ScienceQA, are explicitly assessed rather than hidden in an average.

## 11. Current decision and remaining implementation work

The chosen initial research configuration is shared offline SigLIP2 features, independent two-layer GELU connectors, Qwen2.5-3B teacher → Qwen2.5-0.5B student, DPT CE+write at K=3, CE-only SFT, and DFT CE+write+final forward-KL at K=6. SmoothL1 is the initial reference; signed-effect KL and JS are retained selectable alternatives. No magnitude-restoration term is included.

Before implementing a full run, verify the unavailable CM-FDD connector source, teacher LoRA merge state, tokenizer mapping, complete cache coverage, and local semantic-probe usefulness. Implement an offline-feature model/dataset path and exact selected-block message extraction; the current legacy image-loading trainer does not already implement this framework. Validate six-4090 memory/runtime with ZeRO-2 before finalizing density and query-chunk sizes.

This document records agreed scope, provisional defaults, and experiments. It does not claim that any training experiment, full benchmark evaluation, or resource feasibility test has already passed.

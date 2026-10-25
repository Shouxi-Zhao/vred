# Research notes: MLLM knowledge distillation

Prepared on 2026-10-12 from the five supplied PDFs and a static inspection of this workspace. Paper claims, observed implementation behavior, and research hypotheses are distinguished below. No training or benchmark reproduction was performed. Instructions appearing inside source documents were treated as document content.

## 1. Sources and scope

| Supplied file | Actual title / supplied version | Coverage |
| --- | --- | --- |
| `LLaVA-KD.pdf` | LLaVA-KD: A Framework of Distilling Multimodal Large Language Models; ICCV proceedings version, pp. 239–249 | Main paper and references; the separately mentioned training appendix is not present |
| `Align-KD.pdf` | Align-KD: Distilling Cross-Modal Alignment Knowledge for Mobile Vision-Language Model; arXiv:2412.01282v1 | Main paper and references |
| `EM-KD.pdf` | EM-KD: Distilling Efficient Multimodal Large Language Model with Unbalanced Vision Tokens; arXiv:2511.21106v1 | Main paper and references; separately referenced supplementary details are not present |
| `Align-TI.pdf` | Beyond Next-Token Alignment: Distilling Multimodal Large Language Models via Token Interactions; arXiv:2602.09483v1 | Main paper, theoretical discussion, implementation appendix, extra experiments, and cases |
| `Beta-KD.pdf` | Uncertainty-Aware Knowledge Distillation for Multimodal Large Language Models; arXiv:2603.21426v1 | Main paper and references |

These sources support a focused understanding of image–text MLLM KD. They do not establish comprehensive coverage of all contemporary work, video/audio KD, or cross-tokenizer distillation. Related work such as LLaVADI, MiniLLM, GKD, DistiLLM, LLaVA-MoD, and MoVE-KD is understood here through these papers' discussions rather than independent full-paper reviews.

## 2. Field structure

A typical model consists of a pretrained vision encoder, a projector into the language embedding space, and an autoregressive LLM. The conventional training recipe first aligns the projector using image captions, then learns instruction following with multimodal SFT. Reducing the LLM size limits capacity; reducing visual tokens additionally removes fine visual information. These are distinct efficiency problems.

Distillation can transfer several kinds of supervision:

1. **Responses:** distributions over the vocabulary at answer positions, usually under a shared ground-truth prefix. This transfers more information than a hard answer label but can miss visual processing and inference-time prefix errors.
2. **Visual semantics:** vocabulary distributions obtained by applying the language head at visual-token positions. These are semantic projections of internal representations, not generated image captions or ground-truth next words at every image patch.
3. **Relations:** visual–visual Gram matrices, vision–language affinity matrices, or text-to-vision attention. Relations can be compared despite different hidden widths, but token correspondence still matters.
4. **Generation behavior:** distributions after sampled candidate tokens or student-generated prefixes, to reduce the mismatch between teacher forcing and inference.
5. **Reliability and allocation:** weights deciding how strongly to imitate each teacher signal, sample, or visual region.

The five papers address different coordinates of this design space. A better divergence alone is not a universal solution: LLaVA-KD and Align-TI favor forward KL in their tested settings, EM-KD uses reverse KL successfully, and Beta-KD finds strong results with cosine distance on probabilities in its particular experiments.

For clarity, forward KL means `KL(p_teacher || p_student)` and reverse KL means `KL(p_student || p_teacher)`. The former encourages coverage of teacher probability mass; the latter tends to avoid regions the teacher assigns low probability. Align-KD's prose reverses the usual mean-seeking/mode-seeking terminology; its equation specifies the direction unambiguously.

## 3. Model and data comparison

| Method | Teacher → student | Student architecture / vision | Training data and KD placement |
| --- | --- | --- | --- |
| LLaVA-KD | Qwen1.5-4B → 0.5B/1.8B; Qwen2.5-3B → 0.5B/1.5B; larger-teacher ablations | Shared frozen SigLIP; two-layer GELU MLP; compact dense LLM | 558K captions for DPT; 665K instructions for SFT and again for DFT |
| Align-KD | MobileVLM V2 7B, with Vicuna-7B → MobileVLM V2 1.7B, with MobileLLaMA-1.4B | MobileVLM V2 and its lightweight downsampling projector | ShareGPT4V-PT plus mixed caption/VQA/conversation data; KD accompanies two-stage training; Short/Long subsets |
| EM-KD | LLaVA-OneVision-SI 0.5B/7B, total approximately 0.9B/8B → efficient students approximately 0.6B/8B | LLaVA-NeXT-style AnyRes; Qwen2; SigLIP variants; adaptive pooling plus two-layer MLP; up to 144 tokens per image patch | 558K alignment, then 779K instruction mixture; KD only in phase 2 |
| Align-TI | Qwen2-7B → 0.5B/1.5B; Qwen3-8B → 0.6B/1.7B; Vicuna/MobileLLaMA validation | SigLIP and two-layer GELU MLP | Teacher: 558K + 665K; student: 1.2M ShareGPT4V-PT + 2.4M mixed data; KD only in fine-tuning |
| Beta-KD | MobileVLM V2 7B → 1.7B; additionally Qwen2.5-3B → 0.5B/1.5B in LLaVA-style validation | Reuses existing distillation architectures; adds scalar weights or a small weighting MLP | ScienceQA transfer experiments; broader approximately 2.2M transfer mixture; larger comparison tables label 3.6M samples |

The LLM size is not the total MLLM parameter count. LLaVA-KD's “1B” model has a 0.5B language backbone plus vision/projector parameters. EM-KD also studies approximately equal total-size teacher/student pairs: its primary compression target is the visual sequence.

“1.2M samples” for LLaVA-KD describes approximately 558K + 665K distinct training examples, not total exposure. Its extra DFT stage revisits the 665K set. Exact compute accounting must include teacher forward passes and the additional student stage.

## 4. LLaVA-KD as the baseline

### Architecture and objective

Teacher and student follow a LLaVA-style design with a shared vision encoder and matching language vocabulary. Each has its own projector and language model. The final language head produces vocabulary logits at both response and visual positions.

**MDist** comprises forward KL on response-token distributions and forward KL on visual-token distributions. Prompt tokens are excluded in the best reported setting. **RDist** constructs a visual-token self-correlation matrix and minimizes one minus cosine similarity between teacher and student relation matrices. In the local implementation this Gram matrix is formed directly from visual-position vocabulary logits.

The paper's training loss is:

`L = L_CE + alpha * L_response + beta * L_visual + gamma * L_relation`

Both DPT and DFT use reported weights `(alpha, beta, gamma) = (1, 1, 0.5)`. The paper's surrounding prose sometimes swaps the verbal description of alpha/beta, but the equations and equal numerical values make the intended tested objective clear.

| Stage | Data | Student trainable components | Purpose |
| --- | --- | --- | --- |
| DPT | 558K image–caption pairs | Projector only; vision encoder and LLM frozen | Improve the visual input space using teacher semantics and relations |
| SFT | 665K multimodal instructions | Projector and LLM; vision encoder frozen | Acquire basic understanding and instruction following using CE |
| DFT | The 665K instruction set again | Projector and LLM; vision encoder frozen | Refine understanding using CE, MDist, and RDist |

The teacher itself is trained through ordinary PT–SFT. This makes LLaVA-KD a useful experimental baseline: architecture and distinct data remain simple while the training curriculum and multimodal loss are the main interventions.

### Evidence and limits

For Qwen1.5-0.5B, Table 2 gives PT–SFT = 54.7, DPT–SFT = 55.6, PT–DFT = 55.8, DPT–DFT = 55.9, PT–SFT–DFT = 56.6, and DPT–SFT–DFT = 57.9 on Avg10. This supports a benefit from DPT and an additional distillation pass. It does not isolate all gains from extra training exposure.

An especially relevant counterexample is DPT–DFT–DFT = 58.0. Therefore the results support the selected recipe's performance/compute tradeoff, but do not establish that a CE-only middle stage is universally necessary. Stage timing and KD strength remain open variables.

Table 3 shows DPT with only RDist = 55.5, only MDist = 55.1, and both = 55.6 after SFT. With both in DPT, DFT using only RDist = 57.0, only MDist = 57.7, and both = 57.9. The relative utility of semantic and relational supervision depends on the stage.

Table 4 shows that adding visual positions helps relative to response-only KD, whereas adding prompt positions can hurt. This argues against indiscriminate all-token imitation.

On the main table's Avg10, Qwen1.5-0.5B improves 54.7 → 57.9, Qwen2.5-0.5B 56.5 → 58.9, Qwen1.5-1.8B 56.8 → 62.1, and Qwen2.5-1.5B 64.0 → 65.0. Gains are heterogeneous across tasks: Qwen2.5-0.5B MMMU falls 33.6 → 28.3 despite the improved average. Qwen1.5-0.5B also loses slightly on ScienceQA and MMMU. Preserving reasoning must be an explicit research target.

Table 8 finds Qwen1.5-4B teacher → 0.5B student = 57.9 versus a 7B teacher = 57.4; Qwen2.5-3B teacher → 0.5B student = 58.9 versus a 7B teacher = 58.3. A stronger teacher does not necessarily produce a stronger student. Table 8 also contains a likely backbone-label typo: one baseline row says Qwen2.5-0.5B while repeating the Qwen1.5 baseline score 54.7.

The MobileVLM validation gives 48.8 → 53.4 for a smaller student, supporting some portability. Direct vocabulary KD nevertheless assumes compatible token identities and output spaces; equal vocabulary dimensions alone are insufficient.

## 5. Align-KD

**Question:** Can a mobile student learn cross-modal alignment more directly than through answer distributions alone?

The method distills the first decoder layer's text-query/vision-key attention submatrix using MSE and a learned 1×1 projection. It also aligns projector-produced visual embeddings, with another learned projection for hidden-width differences. Teacher attention summed over text positions selects the top 16 visual tokens. The visual loss includes all-token MSE plus `0.1 * focused-token MSE`, so low-attention tokens continue receiving supervision. Reverse KL on output distributions and ordinary supervised loss complete the objective.

Its shallow-layer motivation comes from feature-change analysis and the interpretation that initial modality alignment occurs early. This is a model-specific hypothesis. Align-TI later finds that instruction-relevant visual focus peaks deeper; the two papers optimize different signals and do not establish a universal best layer.

Training follows MobileVLM V2 on eight V100 GPUs, with global batches 256/128 for pretraining/fine-tuning. The vision encoder stays frozen. The paper reports training the rest of the network, including the LLM, during its pretraining procedure; this differs from projector-only LLaVA pretraining. Reported pretraining learning rates are 1e-3 for the projector and 2e-5 for other components, and fine-tuning uses 4e-5.

The data mixture comprises ShareGPT4V-PT, COCO, SBU, Visual Dialog, ShareGPT4V, SQA, IConQA, TextVQA, VSR, and VIGC. Overlength examples are removed rather than truncated. Thus Short (512 embedded tokens, approximately 3.61M total) versus Long (2048, approximately 3.67M) changes both context length and data composition. Short contains fewer SQA examples in particular. SBU is recollected and filtered.

Six benchmarks are GQA, ScienceQA-IMG, TextVQA, MME perception, MMBench-dev, and POPE. Table 1 reports Short 62.4 → 64.4 (+2.0 points) and Long 63.7 → 65.1 (+1.4 points). The abstract's wording suggests +2.0 for both; the table is more precise. Component and attention-target ablations favor the proposed text-to-vision block, but severely poor visual-to-visual attention KD is specific to that experiment and does not refute LLaVA-KD's final-logit relations.

Its light loss design does not make training free: reported GPU-hours increase 176 → 296 for Short and 228 → 676 for Long. Materializing attention also interacts with optimized attention kernels.

## 6. EM-KD

**Question:** How can a student with fewer visual tokens imitate a teacher retaining more fine detail?

The student uses adaptive average pooling followed by a two-layer MLP. Under AnyRes, the limit is 144 tokens per patch, not necessarily 144 tokens for the whole image. Teacher visual states and student visual states are projected through their language heads. Pairwise Manhattan distances between vocabulary logits define a rectangular assignment problem, solved with a Hungarian algorithm under no gradient. Each student token is assigned a teacher token; excess teacher tokens remain unmatched when the teacher has more tokens.

Three KD signals follow:

- **VSD:** reverse KL between matched visual-position vocabulary distributions.
- **VLAD:** Smooth L1 loss between vision–language affinity matrices, formed using cosine similarity between matched visual hidden states and language hidden states inside each model. This yields comparable relation matrices despite different hidden widths.
- **RLD:** reverse KL on response distributions.

The objective is `alpha * CE + (1-alpha) * RLD + beta * VSD + gamma * VLAD`, with reported `(alpha, beta, gamma) = (0.5, 0.25, 25)`. These scales are not directly comparable to LLaVA-KD without checking reduction conventions.

The 558K first phase trains the projector without KD. The 779K second phase trains the student with KD; the main text says the whole student is trained. Both phases last one epoch. Experiments use eight H20 GPUs. Full student encoder/configuration details are deferred to an absent supplement, so they should not be invented from the model-size labels.

Eleven benchmarks cover general QA (GQA, MME-P, RealWorldQA), knowledge/reasoning (ScienceQA-IMG, AI2D, MMMU validation, MMStar), and OCR/charts (ChartQA, DocVQA, TextVQA, OCRBench). TTFT is measured on RealWorldQA on a single A100.

Reported small-student mean scores are efficient baseline 47.7, MiniLLM 49.5, LLaVA-KD with the same matching mechanism 49.7, and EM-KD 50.4. The full-token LLaVA-NeXT reference is 49.4 with 103.3 ms TTFT; EM-KD is 50.4 with 54.9 ms TTFT. The pooled DeCo student also has 54.9 ms TTFT, so the architecture delivers the speed benefit while KD recovers accuracy.

Matching ablations give average pooling 48.6, hidden-state Hungarian matching 49.4, and logit-based Hungarian matching 50.4. Visual-logit KD also exceeds visual-hidden-state KD. One-to-one matching is a useful baseline but can discard distributed teacher evidence: a pooled student token may summarize several teacher patches. Matching costs scale with both token counts and vocabulary width.

Some table entries and prose differ slightly across configurations, and the large-model MiniLLM row appears malformed. Exact per-task large-model claims require checking the source table or original results before reuse.

## 7. Align-TI

**Question:** Can KD transfer both instruction-relevant visual extraction and the local dynamics of autoregressive generation?

**IVA** weights per-visual-token forward KL by teacher instruction-to-vision attention. The source layer is selected using an Instruction-Relevance Score: one minus expected cosine similarity between vectorized attention patterns for input queries. The intuition is that useful visual grounding changes with the instruction. The reported best tested layer is index 21 of 28, approximately three-quarters of the depth. Teacher-derived weights outperform student-derived weights, and the selected weighting beats uniform visual alignment.

IRS needs methodological care: changing images as well as instructions can change attention without proving instruction sensitivity. A stronger diagnostic holds each image fixed and varies its question. Attention itself is also a proxy for relevance, not a causal guarantee.

**TPA** augments ordinary forward KL under ground-truth prefixes. At each response position, it samples candidate next tokens from the student and aligns teacher/student distributions for the following token conditioned on each candidate plus the shared ground-truth prefix. This is local branching around data prefixes, rather than complete student rollouts. Four samples are used by default; nucleus sampling generally exceeds greedy top-d selection.

The full conceptual transition matrix is vocabulary-by-vocabulary, but actual training estimates selected rows by Monte Carlo sampling. The paper's O(|V|²) path-coverage description should not be interpreted as exhaustive coverage in the implementation.

An augmented sequence and ribbon attention mask prevent sibling candidate tokens from attending to one another. Parallelization requires two student forward passes and one teacher pass per batch, with longer sequences. Forward-pass counts do not capture all memory/FLOP overhead; position indexing and kernel compatibility are important implementation concerns.

The final objective is `CE + visual IVA + response KD + TPA`. KD is restricted to fine-tuning. The student uses 1.2M caption samples plus 2.4M mixed examples, while teachers are trained on 558K + 665K. Appendix hyperparameters are one epoch per stage, batches 256/128, learning rates 1e-3/2e-5, AdamW, cosine schedule, warmup 0.03, and max length 2048. Student experiments use 16 H20 GPUs; teacher training uses eight A100 GPUs.

The six benchmarks are GQA, ScienceQA, TextVQA, POPE, MME, and MMBench. For Qwen2-0.5B, SFT = 64.3, ordinary response forward-KL KD = 65.0, IVA alone = 65.1, TPA with response KD = 66.4, and the full method = 66.7. Qwen3-0.6B reaches 69.7; Qwen3-1.7B reaches 73.6. Gains over LLaVA-KD's published numbers are confounded by different backbones and substantially more student data.

The compute-matched appendix is valuable: ordinary KD for 1.5 epochs gives 65.3 in 533 reported training hours, versus full Align-TI 66.7 in 509. Ordinary KD for one epoch takes 355 hours. The method therefore has evidence beyond simply spending more training time. Exposure-bias analysis, teacher/data scaling, architectural validation, and inference measurements support the proposed mechanisms, but do not prove exposure bias is eliminated in all settings.

## 8. Beta-KD

**Question:** How should a student balance supervised data against heterogeneous, imperfect teacher signals?

Teacher supervision is modeled as a Gibbs prior over student activations: `p(a_student | a_teacher, beta) ∝ exp(-beta * discrepancy)`. MAP estimation combines data negative log likelihood with distillation. Under a local Laplace approximation and a zero-energy minimum, the partition function contributes a log-precision term:

`L = CE + sum_k [beta_k(x) * loss_k - (d_k / 2) * log(beta_k(x))]`

Task-level beta values are shared learned positive scalars. Instance-level beta values come from a small feature extractor and positive-output weighting network, allowing different examples to receive different strengths of teacher supervision. The instance network is reported as 524K parameters, approximately 0.03% of the MobileVLM backbone.

For a fixed positive discrepancy, the stationary precision is `beta* = d / (2 * loss)`. This connects the method to uncertainty-based loss balancing. It also exposes limitations: the negative log term prevents beta collapsing to zero, but does not by itself bound beta when the discrepancy approaches zero. Practical reproduction needs to establish bounds, numerical epsilon, effective dimensionality, reductions, feature inputs, and initialization.

The Bayesian approximation assumes a suitable local Hessian and activation space. Softmax invariance and simplex constraints complicate a blanket claim of applicability to every divergence/activation. Learned beta is an allocation mechanism; identifying it with calibrated teacher correctness requires additional evidence. High teacher–student disagreement can reflect a difficult useful example as well as an unreliable teacher.

The main architecture is MobileVLM V2 7B → 1.7B. Sections 4.2–4.4 transfer ScienceQA training data and evaluate its test set. The paper states that the encoder and tokenizer are frozen and only the language backbone is fine-tuned during distillation. Generalization uses a larger mixed transfer set and six benchmarks matching Align-KD. The claimed 2.2M transfer samples and the table's 3.6M sample labels need reconciliation, potentially separating earlier pretraining from transfer exposure.

Experiments vary loss type, logits versus probabilities, two-loss versus three-loss objectives, and task versus instance weighting. Table 1 reports three-run mean/std on ScienceQA. Cosine-probability KD is the best listed KD-only objective, but CE alone scores 48.4 versus cosine-probability KD 47.2, so it does not beat all objectives in that table.

Beta weighting often helps, but the detailed tables contradict universal improvement: instance weighting slightly worsens skew-forward-KL in Table 2 and lowers several three-loss combinations in Table 3. “Instance always beats task” is likewise unsupported. A learned total loss changing scale over time also cannot alone establish faster student convergence; inspect validation scores and individual unweighted losses.

The most directly relevant experiment is Table 5: with Qwen2.5-3B teaching a LLaVA-style 0.5B student, TextVQA improves 52.0 → 54.9 and ScienceQA 60.6 → 64.4 using instance weighting. The 1.5B student also improves. These are two-benchmark validations, not a full ten-benchmark reproduction. Its 1.5B ScienceQA baseline differs slightly from the supplied LLaVA-KD table.

## 9. Benchmark and experimental-design implications

| Evaluation axis | Useful benchmarks from these papers | What the score does not settle |
| --- | --- | --- |
| General understanding | VQAv2, GQA, VizWiz, RealWorldQA | Whether answers depend on the correct visual evidence |
| OCR / detail | TextVQA, DocVQA, ChartQA, OCRBench | General reasoning or robustness to unseen layouts |
| Reasoning / knowledge | ScienceQA-IMG, AI2D, MMMU, MMStar | Whether gains arise from visual reasoning or language priors |
| Broad capability | MME-P, MMBench EN/CN | A single standardized average across papers |
| Hallucination | POPE | All kinds of hallucination beyond the tested object-presence settings |
| Deployment efficiency | TTFT, decoding throughput, memory | Training cost, end-to-end service latency, or different hardware performance |

LLaVA-KD's supplied proceedings version reports Avg7 and Avg10. Its Avg7 excludes VQAv2, POPE, and MMMU. The workspace README instead describes an average excluding MMMU; these are different conventions. Align-KD, Align-TI, and Beta-KD use six-benchmark means; EM-KD uses eleven. MME perception is scaled by dividing its raw score by 20 in the relevant comparisons. OCRBench scaling and all other normalizations must also be fixed explicitly.

Use percentage points for absolute score differences and percentages only for relative differences. Track benchmark version/split, prompt template, decoding settings, OCR inputs for TextVQA, MMBench evaluation protocol, and POPE metric/subsets. Averaging differently selected metrics is descriptive, not a universal measure of model quality.

For credible new experiments:

1. Keep student initialization, teacher checkpoint, vision encoder, tokenizer mapping, data, image resolution, and evaluation harness fixed in loss comparisons.
2. Compare PT–SFT, longer SFT, response-only KD, LLaVA-KD, and the proposed method at both matched data exposure and matched GPU-hours where feasible.
3. Separate stage changes, additional passes, token selection, reliability weighting, and new supervision with factorial ablations.
4. Report per-task scores, category means, training cost, inference cost, parameter counts, and visual-token counts. Highlight regressions.
5. Repeat key results over multiple seeds and estimate uncertainty using question/image-aware resampling rather than treating every item as independent indiscriminately.
6. Use validation data for hyperparameter/layer selection. Audit image/question overlap and benchmark leakage; shared dataset names do not establish test leakage, but neither do they rule it out.

## 10. Local repository map and reproduction observations

The workspace uses a TinyLLaVA-style factory design. The main flow is:

`JSON conversations + images → text/image preprocessing → image-placeholder expansion → vision encoder → connector → LLM → CE and KD trainer`

| File / directory | Responsibility |
| --- | --- |
| `llavakd/model/modeling_LLaVA_KD.py` | `LLaVAKD`; factory-built model; inserts visual embeddings and expands labels; forward and generation |
| `llavakd/model/llm/`, `vision_tower/`, `connector/` | Registered interchangeable backbones and connectors |
| `llavakd/model/connector/mlp.py` | Two-layer GELU projector when configured as `mlp2x_gelu` |
| `llavakd/data/dataset.py` and preprocessing/templates | Lazy conversation/image loading; padding; supervised labels; modality lengths |
| `llavakd/training_recipe/base.py` | Freeze/full/partial training choices, checkpoint loading/saving; LoRA/QLoRA recipes extend this |
| `llavakd/train/train_distill_qwen2.py` | Distilled pretraining entry; teacher and student initialization |
| `llavakd/train/train_sft.py` | Supervised training entry |
| `llavakd/train/train_distill_after_qwen2_sft.py` | Distillation after SFT initialization |
| `llavakd/train/tinyllava_distill_trainer.py` | Active response KL, visual KL, and visual-logit relation objective |
| `llavakd/utils/distill_loss_utils.py` | Alternative divergence utilities; available functions are not all active in the trainer |
| `scripts/train/llava_kd_qwen2/` | Stage launch scripts and experiment wrappers |
| `scripts/eval/`, `llavakd/eval/`, `docs/Evaluation.md` | Benchmark generation, conversions, and evaluation setup |

Static observations that matter before reproduction:

- **Relation weight differs:** trainer line 346 adds the relation loss with coefficient 1.0; the paper specifies 0.5. Response and visual KL are also directly added. There is no active generic loss-weight configuration in this computation.
- **Visual length is fixed:** trainer lines 333–334 slice 728 visual positions. Shared vision-tower code drops the first patch when configured as `patch`; a 384/14 SigLIP grid ordinarily has 729 patches and no CLIP-style CLS token, making this behavior a concrete point to inspect. It helps explain 728 but needs checkpoint/runtime confirmation. Do not silently change it and still call the result an exact baseline reproduction.
- **Batch assumptions:** visual slicing uses the first example's split offset, and the returned `num_images` comes from the last example processed. Mixed image/text or differing-offset batches need validation. The shipped launchers use per-device batch size 1, which avoids some of these cases. More than one image per sample is not supported by the active trainer branch.
- **Response mismatches:** if selected teacher/student response-logit shapes differ, the trainer prints the mismatch and omits response KD for that batch. Equal shapes do not prove aligned token identities; tokenizer and special-token compatibility must be verified independently.
- **Checkpoint paths are embedded:** both KD Python entries hard-code a Qwen2.5-3B teacher; the DFT entry also hard-codes its student SFT checkpoint. Shell teacher arguments therefore do not by themselves select the loaded teacher in these entries.
- **Stage orchestration needs checking:** `train_all_stage.sh` is empty. The SFT launcher expects a `-pretrain` checkpoint path whereas DPT writes a `-distill-pretrain` path. The DFT shell and Python paths also reflect different experiment variants.
- **Vision identity differs in naming:** paper Section 4.1 says SigLIP-B/14@384, while local scripts/README point to `siglip-so400m-patch14-384`. These variants should not be treated as interchangeable for parameter counts or faithful reproduction.
- **Unused relation outputs:** multimodal preparation computes projector-level visual/text relations, but the active KD trainer's RDist uses language-head visual logits. Returning a tensor does not mean that tensor is a distillation target.

These are static implementation observations, not demonstrated training failures. No source code was changed. Git history was unavailable in the exposed workspace, so no provenance claim is made about local modifications.

## 11. Research hypotheses worth testing

The following are hypotheses, not established novel contributions. Their novelty must be checked against additional literature before selecting a project.

### A. Distill the visual evidence that changes an answer

Attention-weighted visual KD already exists in Align-KD/Align-TI. A more discriminating question is whether teacher attention actually identifies answer-essential evidence. Measure teacher response changes after controlled masking or image perturbations, then distill useful response effects or response-conditioned representations. Compare against random regions, attention weighting, and ordinary LLaVA-KD under matched extra teacher compute. Evaluate image dependence, hallucination, and retention of unselected details. Masking can introduce out-of-distribution artifacts, so perturbation validity must be tested.

Causal masking matters architecturally: visual positions before an instruction cannot attend to that later instruction in a decoder. Weighting their loss by instruction attention changes supervision allocation, but does not make those earlier visual representations instruction-conditioned. This distinction should guide the choice of representation or response-side target.

### B. Model transferable difficulty separately from teacher reliability

Beta-KD already supplies adaptive weights. An additional contribution would need to distinguish teacher errors, student capacity mismatch, and hard but valuable samples. Compare disagreement-only weighting with reliability estimates from teacher consistency, label agreement where available, and controlled corruptions. Test whether adaptive supervision retains MMMU/ScienceQA capability while improving perception. Avoid automatically downweighting every large teacher–student gap.

### C. Transfer distributed evidence under token compression

EM-KD establishes hard matching as a baseline. A compressed token may represent multiple teacher regions, motivating a spatially constrained soft or many-to-one transport relation. Test multiple token budgets, coverage of teacher evidence, matching stability, OCR/detail retention, and latency. Compare to pooled matching and Hungarian logit matching using the same student. Soft transport and many-to-one KD are established concepts elsewhere, so their mere addition is insufficient for a novelty claim.

### D. Explain stage-dependent KD through measured student readiness

LLaVA-KD shows stage effects and a larger-teacher transfer bottleneck. Instead of assuming fixed stages are optimal, measure gradients, loss conflicts, and task-specific teacher gaps over training. Compare a readiness-based curriculum against fixed DPT–SFT–DFT, continuous KD, repeated DFT, longer SFT, and learned Beta weighting at a fixed budget. A gain must survive these controls to justify a new curriculum claim.

### E. Connect local generation correction to visual grounding

Align-TI already branches one step around data prefixes. A further hypothesis is that correction should concentrate on visually sensitive answer decisions rather than generic response positions. Compare uniform TPA, entropy-based selection, and visually sensitive selection at the same sampled-token budget. Check whether gains occur on grounded answers and reasoning, and whether local branching remains sufficient under longer student-generated prefixes.

Simply combining attention weighting, Hungarian matching, TPA, and Beta weighting would overlap heavily with these papers and make attribution difficult. A stronger project identifies one measurable failure mode of LLaVA-KD, proposes a mechanism tied to that failure, and demonstrates benefits under controlled data and compute.

## 12. Practical starting point for our next phase

Retain LLaVA-KD as the primary baseline, initially using the workspace's Qwen2.5-3B teacher and Qwen2.5-0.5B student family. First document the exact checkpoint, tokenizer, vision configuration, patch selection, loss reductions/weights, and stage transitions. Reproduce both paper-faithful settings and local-code settings if they differ, labeling them separately.

Then diagnose per-task regressions and teacher transfer quality before choosing a new loss. Start exploratory evaluation with GQA, TextVQA, ScienceQA-IMG, MME-P, MMBench, and POPE, but retain MMMU as a required reasoning check and eventually reproduce all ten LLaVA-KD benchmarks. Extend to DocVQA/ChartQA and latency measurements when the research hypothesis targets token compression. Hardware, data availability, and checkpoint access should determine the scale of the first controlled experiment.

## 13. Implemented VRED option: effect-weighted KD

The VRED DFT path now has an opt-in token-level effect-weighted KD term
(`--effect-weighted-kd`, off by default):

`L_ewkd = mean_t(w_t * KL(p_t+ || p_s+))`, with
`w_t = ||p_t+ - p_t-||_1`, `p_t+ = softmax(z_t+ / T)`,
`p_t- = softmax(z_t- / T)`, and `p_s+ = softmax(z_s+ / T)`.
The plus teacher sees the image; the minus teacher uses the same text and visual
positions with pre-connector visual features zeroed. This is an input-level
counterfactual, not the layer-local `u_minus` used by `L_write`; it is implemented
only for DFT with sparse logits and costs one extra frozen-teacher forward.

When enabled, ordinary final-logit KD is kept as a small anchor rather than at
the same scale as the effect term. The documented starting point is
`lambda_logits=0.2` and `lambda_effect_kd=1.0` (1:5). The effect weights are not
normalized and are usually below 1, so `lambda_effect_kd=1.0` already makes the
effect term primary; `lambda_effect_kd=2` is not the default recommendation and
should be used only when the raw `effect_kd / logits_kd` ratio is below about
0.1. Compare weighted contributions (`0.2 * logits_kd` vs `1.0 * effect_kd`),
not raw loss magnitudes. This term is complementary to prompt-scope
`L_write`: `L_write` is an intermediate-block context regularizer, while
`L_ewkd` is a response-token distributional term. The combined objective should
be ablated against `+L_write`, `+L_ewkd`, and baseline; it is not established as
a strict improvement over ordinary KD.

### Hidden-effect variant with floor

A cheaper variant (`--effect-kd-weight-source hidden`) avoids the second teacher
forward by reusing the plus effect capture. For each response token it computes
`s_t = mean_pair ||u_(l,t)+ - u_(l,t)-||_2`, then
`w_t = floor + (1-floor) * s_t / mean(s)` and
`L_ewkd = mean_t(w_t * KL(p_t+ || p_s+))`. Text-only positions get the floor.
For nonzero effects the weights have mean one, which makes the effective KL scale
much larger than the unnormalized logits-difference formulation; the starting
coefficient is therefore `lambda_effect_kd=0.5` with `lambda_logits=0` and
`--effect-kd-floor 0.15`, not 1.0. The hidden variant requires response-scope
write positions with no limit and uniform sampling, so its per-token scores align
with `batch.response_positions`; the code rejects other combinations. It is an
ablation candidate, not evidence that the extra teacher-minus forward is
unnecessary in the logits formulation.

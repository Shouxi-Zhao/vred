# Implementation status (2026-10-16)

Prompts 01-03: offline foundation, local effects/losses, DPT/SFT/DFT training,
launch/resume profiles, cached inference, legacy evaluation adapter and bounded
diagnostics. Prompt-01/02 asset measurements below are historical; those assets
are absent in the current prompt-03 filesystem snapshot. Final verification and
external prerequisites are recorded in the prompt-03 section at the end.

## Verified assets and contracts

- Read `project_background.md`, `docs/coding_principles.md`, and the existing
  dataset, Qwen/pretrain templates, connector, training scripts, LoRA save recipe,
  and checkpoint loading code before implementation. Legacy code was not modified.
- DPT annotations: 558128 records in `blip_laion_cc_sbu_558k.json`.
- SFT/DFT annotations: 664939 records in `llava_v1_5_mix665k_clean.json`.
- Cache index: 906803 entries. Actual DPT and SFT image features load as finite
  CPU FP16 arrays of shape `(729, 1152)`. This is a sampled loading check, not a
  complete cache-coverage or online-feature-equivalence audit.
- Mixed smoke batch: text record at clean JSON row 624251 (`i6IyJda_0`) and
  image record at row 0 (`000000033471`), padded shape `(2, 853)`, with 112
  shifted response prediction positions and 39 image-valid write positions.
- All 729 visual positions remain intact. Text-only records have no fabricated
  features. Mask generation occurs after expansion/right truncation/padding;
  real EOS tokens remain attended when EOS equals PAD. Visual spans are half-open.
- Response prediction position `p` targets label `p+1`; the prompt-end position
  predicting the first answer is included. A two-turn boundary probe verifies
  that Qwen's leading-space BPE token is supervised. Write positions additionally
  exclude visual/padding positions, text-only rows, and positions before the image.
- Missing keys/files, wrong feature shapes, NaNs, image lists, duplicate/missing
  placeholders, and truncation through the image span fail with record/path details.
- A tiny two-block Qwen passes shifted-CE equality and connector backward through
  a frozen LM, with finite nonzero connector gradients and no LM gradients.
- Supplied 0.5B student passes a mixed text/image forward and exact save/reload
  prediction agreement. Its connector has biased weights `(896, 1152)` and
  `(896, 896)` separated by GELU.
- Default training-mode checks enumerate module types, reject vision factory
  use, and verify no `vision_tower` attribute or legacy vision implementation is
  loaded. Only the language model and connector are constructed in `VREDModel`;
  the only encoder is the separate frozen tower loaded by the opt-in
  `--online-vision` training path.

## Teacher format decision

Pinned decision: `unmerged_llavakd_lora` in `configs/foundation.json`.
The supplied checkpoint is `checkpoints/ta-qwen25-3b-stage0_2-lora-finetune`.
Its language file has 434 non-LoRA parameter tensors, including `.base_layer`
keys; the root adapter has 504 tensors for 252 linear modules, rank 256,
alpha 512, bias `none`, with wrapper-prefixed language-module names.
All 252 adapter-targeted saved base weights exactly equal the corresponding
supplied `pretrained_hg/Qwen2.5-3B` weights. The legacy save recipe separately
saves non-LoRA base parameters and root adapters, corroborating this decision.

The loader strips `.base_layer`, strictly loads the language state, restores
the tied vocabulary head, and applies `W + (512/256) * B @ A` once on CPU.
The default result has no LoRA branches, is frozen, and stays in eval mode.
Loading provenance is saved with VRED checkpoints, which reload directly without
applying adapters. An attempted second merge is an error.

Teacher numerical checks passed with the actual 3B checkpoint. The independent
FP32 reference runs the frozen base linears plus separate `B(A(x)) * 2` branches
on a short text sequence. Merging produces full-vocabulary logits with maximum
absolute difference `0.00018024444580078125`; `torch.testing.assert_close` passes
with `atol=0.0002`, `rtol=0.0002`. A second merge is rejected. The teacher connector
has biased weights `(2048, 1152)` and `(2048, 2048)` separated by GELU. Its mixed
text/image BF16 forward has finite outputs and CE `0.7314658761024475`, with no
parameter gradients. Teacher remains in eval after a training-mode request.

## Tokenizer decision

Teacher/student full token-to-ID mappings, added tokens, parsed tokenizer
backends (including BPE merges/normalization), and chat templates agree.
The tokenizer has 151665 entries (maximum ID 151664), including 22 added tokens;
vocabulary heads have 151936 rows. Teacher's only special-token-map difference
is an UNK alias to the existing `<|endoftext|>` EOS/PAD token; validation permits
that explicit difference and returns the teacher tokenizer for shared encoding.
Both tokenizer BOS values are absent, while both model configs alias BOS to EOS;
encoding explicitly disables automatic special-token insertion. Model EOS/BOS/PAD
configuration compatibility is checked too, including on saved-model reload.
The on-disk tokenizer JSON serialization differs, so byte equality is not used.

## Commands and limits

Available verified environment: Python 3.11, PyTorch 2.10.0+cu128,
Transformers 4.51.0, NumPy and safetensors, under
`/home/inspur/miniconda3/envs/dino`. The default shell Python lacks transformers.
CUDA/driver access is unavailable in this session; no GPU feasibility claim is made.
The complete CPU smoke passed in 227.49 seconds using four CPU threads, BF16
student/teacher mixed forwards, and an FP32 teacher merge reference. Student CE
was `1.368418574333191`, and save/reload maximum prediction error was exactly zero.
These are bounded implementation checks, not trained-student performance results.

```bash
cd /data/zsx/LLaVA-KD
/home/inspur/miniconda3/envs/dino/bin/python -m vred.smoke
# Fast rerun with real data/cache and a tiny language model:
/home/inspur/miniconda3/envs/dino/bin/python -m vred.smoke --tiny-only
```

The Transformers sliding-window warning comes from these Qwen configs; their
`use_sliding_window` flag is false. The foundation uses ordinary eager attention.
No dependencies were installed and no supplied asset was modified.

Prompt 02 is `implementation_prompts/02_vred_training.md`. From the repository root,
the command to start that next implementation task in a Codex CLI session is:

```bash
codex exec -C /data/zsx/LLaVA-KD - < implementation_prompts/02_vred_training.md
```

Prompt 02 is now implemented below. `Batch` prediction coordinates and the ordinary
`VREDModel.forward(batch)` implementation are preserved. Online/cache equivalence,
full cache coverage, source-aware validation splitting, and GPU memory/runtime
remain unverified.

## Prompt 02 implementation

- `effects.py`: temporary eager-Qwen hooks read actual all-key attention
  probabilities and V projections. Actual RoPE, GQA, masks, scaling and dropout
  are reused. The visual projection excludes output bias; subtraction from the
  actual attention output leaves bias exactly once in the nonvisual term.
  Negative FFN branches run only at shared shifted write coordinates. The real
  decoder always continues from its original post-MLP output.
- `losses.py`: detached-parameter functional Qwen RMSNorm/head probes, FP32
  probabilities and losses, full vocabulary, all three specified write losses,
  signed `2V+1` normalization checks, uniform-mixture KL smoothing, safe JS zeros,
  and shifted final teacher-to-student KL with temperature squared.
- `trainer.py`/`train.py`: DPT connector-only CE+write (K=3), SFT connector+LM CE
  without teacher, DFT connector+LM CE+write+final KL (K=6). Frozen teacher is
  outside the student DeepSpeed engine. The default is BF16 model computation
  and FP32 probes/losses; no magnitude/affinity/handoff/CMAD/CMAED terms exist.
- `configs/training.json` contains the shared defaults; `configs/zero2.json`
  explicitly selects stage 2, BF16 on, FP16 off. Resolved six-rank effective
  batches are 192 (DPT) and 96 (SFT/DFT). Defaults use one epoch, cosine schedule,
  warmup 0.03, AdamW, weight decay zero, and the background's learning rates.
- Reductions use actual valid tokens across all accumulation microbatches and
  ranks, including partial groups. Shared coordinate sampling is optional and
  disabled by default. Exact full-vocabulary probe-loss chunks are checkpointed
  to avoid retaining K vocabulary graphs. The default query chunk size is 32.
- CLI exports explicitly reloadable model/tokenizer checkpoints, engine optimizer
  checkpoints when using ZeRO-2, resolved run configuration, raw/weighted losses,
  local/global exposure counts, timings, and CPU/GPU memory fields. SFT/DFT
  require a predecessor checkpoint. Prompt 03 adds in-stage optimizer/RNG/cursor
  resume and periodic exports, verified below.

## Prompt 02 verification

The CPU numerical suite passes 12 tests, including:

- Independent Q/K RoPE + GQA + causal/padding softmax reference; direct
  nonvisual/visual projected sums, including an injected output bias.
- Bit-identical ordinary logits/loss with and without hooks, including dropout.
  FP32 reconstructed local `u_plus` agrees with the real block output at
  `atol=1e-7, rtol=1e-5`; `v=0` gives exactly equal branches and zero effects.
- Shared explicit shifted positions, valid EOS=PAD handling, relative-depth
  pairs, text-only DPT, no teacher gradients, and hook cleanup on exceptions.
- Detached norm/head gradients with tied embeddings, differentiable plus/minus
  states, write-only DPT connector and DFT decoder gradients, and ordinary CE
  gradients reaching the trainable tied head.
- SmoothL1 toy/reduction reference, KL/JS toy values, normalization, smoothing,
  sign reversals, finite/nonzero gradients at zero student effect, and finite
  zero/zero gradients. Both DPT and DFT backward pass in BF16 for all three modes.
- Checkpointed versus ordinary probe values/gradients at a fixed query size;
  changing query size preserves values within declared tolerances.
- Accumulated microbatches match a single mixed batch's valid-token loss/gradient,
  including a final partial group. A bounded mocked-asset CPU CLI run executes
  AdamW/scheduler, accumulation, logging and checkpoint export.
- ZeRO-2/BF16 resolved configuration and effective batch arithmetic. The actual
  DeepSpeed engine is **not** exercised: the package and CUDA are unavailable.

Real annotations/cache and tiny-width 24/36-layer BF16 Qwen models with the full
151936-row vocabulary passed DPT, SFT and DFT forward/backward. All 729 image tokens
were retained, with no position sampling. CPU measurements (four threads):

| Stage | Batch shape | Response/write positions | Seconds | Peak process RSS MiB |
| --- | --- | --- | --- | --- |
| DPT | 1 x 743 | 14 / 13 | 1.886 | 9824.27 |
| SFT | 2 x 853 | 112 / 39 (write disabled) | 2.241 | 9824.27 |
| DFT | 2 x 853 | 112 / 39 | 5.014 | 9824.27 |

RSS is the process lifetime high-water mark, including annotation/index parsing,
not isolated activation memory. Tiny-width BF16 block reconstruction maximum
absolute difference was `0.0009765625`, with relative L2 error `0.00496555`.
The FP32 maximum absolute difference was `1.49012e-8`, with relative L2 error
`7.97577e-8`; ordinary forward difference was zero in both dtypes.
The reconstruction diagnostic measures the maximum per-position relative L2
error, with tolerances 0.02 for BF16 and 1e-5 for FP32, configured in
`configs/stage_smoke.json`. The ordinary path itself is never reassociated and
must agree bit-for-bit with hooks enabled.

On the supplied student, reconstructing
`h + (o_total-o_vision) + o_vision` in BF16 gives maximum absolute error 1.25
and maximum per-position relative L2 error 0.01829563. An earlier pointwise
`atol=0.04, rtol=0.02` assertion failed on 29 of 34944 state coordinates because
of cancellation/rounding. This is a measured BF16 limitation, not exact local
arithmetic equality. Repeating with the same weights upcast to FP32 gives
maximum absolute error 3.43323e-5 and relative L2 error 5.81936e-7. Ordinary
logits agree exactly in both dtypes. The negative branch uses BF16 model
arithmetic; FP32 probes cannot undo this branch-state rounding.

The supplied 0.5B student / frozen 3B teacher also passed all three BF16
forward/backward checks, using the same real examples, full vocabulary, all image
tokens, and all valid write positions. The complete `--real-models` command
exited successfully, including both reconstruction diagnostics; its JSON report
is `/tmp/vred-real-stage-smoke.json`. These checks perform backward without an
optimizer step; they do not constitute the DPT-to-SFT-to-DFT training experiment.

| Stage | CE | Write (vocabulary sum) | Final KL | Seconds | Peak process RSS MiB |
| --- | --- | --- | --- | --- | --- |
| DPT | 8.433605 | 0.00176808 | 0 | 63.039 | 12656.85 |
| SFT | 1.478834 | 0 | 0 | 49.197 | 16353.02 |
| DFT | 1.478834 | 0.00341849 | 0.754515 | 166.482 | 18199.18 |

Connector gradient norms were 170.551 / 8.557 / 16.859 respectively. DPT had no
language parameter gradients; SFT/DFT first-block MLP gradient norms were
4.480 / 8.161. Teacher parameters had no gradients throughout. Timings are bounded
CPU diagnostics, not a GPU throughput projection. Peak RSS includes parsing,
loaded model weights, and gradients; it excludes full optimizer-state allocation.

A separate synthetic full-vocabulary DFT memory check (six tiny blocks, 729 image
tokens, 128 supervised positions, BF16 backbone, FP32 probe, query chunks of 32)
isolates probe retention from annotation parsing:

| Probe-loss checkpointing | Peak CPU RSS MiB | Loss | Connector gradient norm |
| --- | --- | --- | --- |
| Off, ordinary eager autograd reference | 8044.26 | 11.4845151901 | 0.1268637180 |
| On, nonreentrant recomputation | 5649.85 | 11.4845151901 | 0.1268637180 |

This measured reduction of about 2394 MiB motivates the default probe
checkpointing. It does not reduce the eager backbone's full attention graphs or
ordinary CE logits. Reproduction commands (run in separate processes):

```bash
/home/inspur/miniconda3/envs/dino/bin/python -m vred.tests.memory_probe
/home/inspur/miniconda3/envs/dino/bin/python -m vred.tests.memory_probe --checkpoint
```

## Remaining limits

- This correctness path requires Transformers Qwen2 eager attention (verified
  4.45.0 and 4.51.0) and full causal attention. Fused attention and backbone checkpointing
  during effect capture are not implemented. Enabling backbone checkpointing
  raises clearly because backward recomputation would outlive temporary hooks.
  Probe-loss checkpointing is supported and tested.
- CPU fallback AdamW operates on BF16 parameters; the intended DeepSpeed path
  provides FP32 optimizer master weights. Six-4090/ZeRO-2, peak GPU memory,
  max-length throughput, and GPU-hours remain unmeasured.
- Effect-KL's sign split and small smoothing mass make near-zero effects
  sensitive to cancellation. In a tiny FP32 test, changing query size 2 to 1
  changed a decoder gradient by up to `8.0648e-5` while scalar KL changed by about
  `2.24e-7`. Checkpointing at the same query size produced identical gradients.
  This is recorded numerical sensitivity, not a top-k or other approximation.
  Keep chunk size fixed in controlled experiments and validate the loss family.
- No long training, optimizer-tuned research result, semantic-probe usefulness
  claim, complete cache/online equivalence audit, or benchmark result is asserted.

## Next Commands

```bash
cd /data/zsx/LLaVA-KD
/home/inspur/miniconda3/envs/dino/bin/python -m unittest discover -s vred/tests -v
/home/inspur/miniconda3/envs/dino/bin/python -m vred.stage_smoke --report /tmp/vred-stage-smoke.json
/home/inspur/miniconda3/envs/dino/bin/python -m vred.stage_smoke --real-models --report /tmp/vred-real-stage-smoke.json
```

Full stage and ZeRO-2 launch commands are in `vred/README.md`. Prompt 03 is
implemented and verified below.

## Prompt 03 Delivery And Verification

Current environment: Python 3.11, PyTorch 2.10.0+cu128, Transformers 4.51.0;
CUDA is unavailable and DeepSpeed is not installed. The current filesystem has
no `pretrained_hg/`, `checkpoints/`, `mllm_datasets/`, or `eval_dataset/` project
asset directories. No supplied-model training or real benchmark scoring was
rerun for prompt 03. Earlier supplied-model numbers are retained as historical
prompt-02 evidence, not current asset-availability claims.

Implemented entry points and contracts:

- `launch.py` and `configs/launch/{dpt,sft,dft}.json`: current-interpreter
  one-node/six-rank ZeRO-2 launch, explicit predecessor/resume/output paths and
  seed. Shared training JSON records BF16 precision, losses, coefficients,
  microbatch/accumulation and checkpoint interval. Dry-run resolves these values.
- `train.py`: records corpus/cache coverage, tokenizer special IDs and vocabulary,
  layer pairs, reductions, trainable parameter/module counts, and actual exposure.
  Checks the full selected-stage annotation/cache-key/file coverage before loading
  models. `preflight.py` can separately audit both files and validate every array.
  Cache mappings now use the encoder script's key normalization, reject collisions
  and prohibit escaping the cache root. No bad entry is silently skipped.
- Same-stage resume restores optimizer, scheduler, epoch/microbatch cursor,
  model RNG and write-position RNG. Settings/assets/seed/world size must match.
  Checkpoints export every 1000 optimizer steps and at exit. A completion marker
  rejects interrupted exports; the latest root export is not transactional, so
  preserve an older completed checkpoint when durable fallback is needed.
  `--max-steps` caps absolute steps; scheduler horizon remains the configured
  full epochs. CPU resume is tested; actual ZeRO engine resume is unverified.
- `infer.py`: deterministic cached-image or text-only generation using the
  ordinary language-model generate implementation. Every visual token remains;
  no image processor or vision encoder is instantiated.
- `evaluate.py`/`configs/evaluation.json`: ten named legacy layouts (GQA,
  ScienceQA, TextVQA, POPE, MME, MMMU, VQAv2, VizWiz, MMBench EN/CN), legacy
  answer JSONL, repository/local scorers, and submission converters. Embedded
  TSV images require an explicit question-ID/cache-key map. Multiple images fail;
  MMMU text-only questions use generation, without random fallback answers.
  VQAv2, VizWiz-val, and MMBench EN/CN dev are scored locally from local
  annotations/TSV with strict answer coverage (VizWiz uses the previous
  `TextVQAAccuracyEvaluator` soft accuracy; MMBench uses the dev TSV answer
  column and the prediction's first standalone option letter) rather than the
  old converter's silent missing-answer fill/write bug.
  Local scorer logs
  are preserved; collector JSON/CSV distinguishes local scores, external
  submissions, subsets, failures and not-run benchmarks. Numeric values are
  copied from structured scorer output or printed metrics; no aggregate is invented.
  `--action summary` writes a compact `<output>/metrics.json` containing only
  the ten benchmark metrics plus `avg10`/`avg7`: MMMU `overall.acc * 100`,
  MME `total / 20`, POPE mean per-category `Accuracy` (0-1 to percent), and
  `avg7` excludes VQAv2/POPE/MMMU. It fails with the missing benchmark names
  rather than averaging an incomplete set. `--action run-all --benchmarks all`
  invokes the same writer automatically after its final collect (unless
  `--limit` is set).
- `diagnose.py`: nine variants on exactly the same first N training examples,
  with identical initial weights/seed/optimizer, K=3/6 and all three write losses.
  Default N=2, cap 8; JSON/CSV include raw/weighted losses, positions/exposure,
  gradients and memory. It does not use test benchmarks or initiate a full sweep.

Verification completed on CPU:

- All 19 unit/integration tests pass. In addition to prompt-02 numerical tests,
  these verify mid-epoch and epoch-boundary resume against uninterrupted training
  with exact parameter, optimizer-state, RNG and exposure agreement; periodic
  checkpointing; cache failures/collisions/path escape; six-rank launch resolution;
  cached generation; full inference -> legacy TextVQA scoring -> result collection
  on a labeled synthetic fixture; and rejection of scoring a subset. Fixture
  scores are test assertions, not project benchmark results.
- End-to-end subprocess smoke passed DPT -> same-stage DPT resume -> saved DPT
  reload into SFT -> saved SFT reload into DFT -> export/reload -> cached-image
  generation. It uses explicitly synthetic six-block models, a 128-entry
  tokenizer, a real-format tiny LoRA checkpoint, one text record and one cached
  image record. All 729 x 1152 finite FP16 features are retained. Zero warmup
  ensures nonzero learning rates for the bounded optimizer steps.

| Synthetic stage | Optimizer steps | Example exposures | Response tokens | Peak process RSS MiB |
| --- | --- | --- | --- | --- |
| DPT including resume | 2 | 4 | 14 | 1001.20 |
| SFT | 1 | 2 | 9 | 999.39 |
| DFT | 1 | 2 | 9 | 1032.68 |

Each mixed SFT/DFT batch has five image-valid write positions. DFT CE was
`4.8504801`, write loss `5.18163e-6`, and final KL `0.00988876`. These are synthetic implementation
metrics only. RSS is a lifetime process high-water mark, not isolated activations
or projected real-model memory. The saved/reloaded logits agreed exactly (error
zero). Cached generation produced four tokens from a 766-token expanded prompt;
module/factory assertions verified no training vision encoder.

- Bounded diagnostics completed all 9 variants x 2 examples = 18 records, with
  identical example order and 9 response-token exposures per variant; K=3/6 and
  SmoothL1/Effect-KL/Effect-JS all executed finite forward/backward/optimizer steps.
  Latest JSON/CSV artifacts, including explicit layer pairs, are under
  `/tmp/vred-prompt03-complete-diagnostics/`.
- Shell entry points for full-array preflight on synthetic fixtures, saved-model
  cached inference, launch dry-run, benchmark listing and result collection pass.
  Benchmark listing reports all ten question files unavailable. Collection with
  no runs emits `not_run` and no scores.

Reproduction (use fresh output paths):

```bash
cd /data/zsx/LLaVA-KD
PYTHON=/home/inspur/miniconda3/envs/dino/bin/python
"$PYTHON" -m unittest discover -s vred/tests -v
"$PYTHON" -m vred.pipeline_smoke --synthetic --output /tmp/vred-prompt03-final-smoke
"$PYTHON" -m vred.diagnose --synthetic --examples 2 --output /tmp/vred-prompt03-final-diagnostics
"$PYTHON" -m vred.preflight --foundation-config /tmp/vred-prompt03-final-smoke/synthetic_assets/foundation.json --validate-arrays --output /tmp/vred-prompt03-coverage.json
"$PYTHON" -m vred.infer --model /tmp/vred-prompt03-final-smoke/dft --foundation-config /tmp/vred-prompt03-final-smoke/synthetic_assets/foundation.json --image-key synthetic/square.jpg --question 'What is shown?' --max-new-tokens 4 --output /tmp/vred-prompt03-inference.json
"$PYTHON" -m vred.launch dft --dry-run
"$PYTHON" -m vred.evaluate --action list
"$PYTHON" -m vred.evaluate --action collect --output /tmp/vred-prompt03-results
```

Final external prerequisites:

- Restore/set the student and teacher checkpoints/tokenizers, both training JSONs
  and training cache paths. Audit complete cache coverage/integrity and construct
  a training-only validation split before long training or tuning.
- Provide the six CUDA GPUs and compatible DeepSpeed installation. Six-GPU
  ZeRO-2 launch/resume, max-length VRAM/throughput and real optimizer training
  remain unverified here. Backbone checkpointing is still unsupported during
  effect capture; exact probe checkpointing is supported.
- Restore all benchmark question/image/annotation directories plus matching
  evaluation feature caches. External GQA/MME/MMMU scorer tools and dependencies
  must match the configured legacy layout; their full integrations could not be
  exercised without those directories. MME's legacy converter writes within
  its benchmark directory and requires a unique experiment name.
- VQAv2, VizWiz val, and MMBench EN/CN dev are scored locally; VizWiz test and
  external MMBench test/submission specs need official external
  submission/scoring. No submissions were sent and no real benchmark scores or
  research improvement are claimed. Ten benchmark commands,
  setup and troubleshooting are in README.

## Training Flag Overrides

Both `vred.launch` and `vred.train` now accept explicit flags for learning rate,
epochs, batch/accumulation, seed, stage loss coefficients/pairs, write-loss type
and reduction, temperatures, smoothing, sequence length, query chunks, optional
write sampling, probe checkpointing, optimizer settings, checkpoint frequency,
workers/threads, and asset/data paths. The launcher also exposes GPU count,
config paths, predecessor, output and resume. Flags override JSON and resolved
settings are logged; BF16 precision and stage-specific objective constraints
remain enforced. Resume requires matching effective flags/configuration.

All 21 tests pass, including launch resolution/forwarding and a bounded actual
training run with flag overrides for loss, seed, learning rate, accumulation,
probe checkpointing and sampled positions. A six-rank DFT dry-run with Effect-JS
and coefficient/chunk/step overrides passes; no real GPU training was launched.
Expanded stage commands and flag reference are in `vred/README.md`.

Launcher compatibility follow-up: positional stages and `--stage` are both
accepted, with conflicts rejected. All 22 tests pass. The user's exact DPT
settings (six ranks, microbatch 32, accumulation 1, global batch 192) pass a
`--stage dpt --dry-run` under `/home/inspur/miniconda3/envs/nnunet/bin/python`.
That environment reports PyTorch 2.4.0+cu124 and Transformers 4.45.0. Initial
verification covered parsing only; effect extraction is now verified below.
No GPU job was started.

## Attention API Compatibility Debugging (2026-10-16)

Reproduced the reported unpacking failure in `nnunet`: Transformers 4.45.0
returns `(output, attention_weights, past_key_value)` and clears attention
weights unless explicitly requested. The hook now accepts both supported
return formats and temporarily requests weights only on selected attention
modules. Positional and keyword attention flags are supported. Decoder/model
output flags, cache objects, ordinary forward and attention dropout are preserved.

Verification:

- `nnunet` (Python 3.10, PyTorch 2.4.0+cu124, Transformers 4.45.0): all 23 tests
  pass in 4.476 seconds.
- `dino` (Python 3.11, PyTorch 2.10.0+cu128, Transformers 4.51.0): all 23 tests
  pass in 4.276 seconds.
- Regression coverage includes selected-block flags, positional flags, hook
  cleanup, unchanged ordinary logits/outputs, causal/padding normalization,
  RoPE/GQA/bias decomposition and identical dropout draws. Existing BF16 tests
  verify DPT connector/DFT decoder gradients and frozen teacher behavior.
- The `nnunet` CPU synthetic pipeline completes DPT, DPT resume, SFT and DFT
  optimizer steps, export/reload with exactly zero logit difference, and cached
  image inference with all `(729,1152)` features and no vision encoder.
  DFT peak RSS is 731.80 MiB. Results are in
  `/tmp/vred-attention-445-smoke-20261016/smoke.json`.

Commands (replace `python` with the interpreter being checked):

```bash
python -m unittest discover -s vred/tests -v
python -m vred.pipeline_smoke --synthetic --output /tmp/vred-attention-445-smoke-20261016
```

Use a fresh output directory when repeating the pipeline. CUDA is unavailable
in this session, so six-GPU DeepSpeed training and actual-model memory remain
unverified. The installed legacy DeepSpeed emits an ignored shutdown exception
when its Triton autotune cache targets a read-only home directory; all tests and
pipeline processes exit successfully. A writable `TRITON_CACHE_DIR` can be set
for this environment. No benchmark or research improvement is claimed.

## Output Overwrite And Training Progress (2026-10-17)

Fresh training runs now replace an existing output directory marked by
`run_config.json`, removing stale checkpoints, rank states and metrics before
writing the new run. Cleanup occurs on rank zero after model loading, with
distributed barriers around output preparation. Explicit in-place `--resume`
preserves existing checkpoints/metrics and restores training state as before.

DPT, SFT and DFT share a rank-zero tqdm bar showing optimizer-step progress,
speed, elapsed time, ETA, epoch, loss, learning rate and valid-token throughput.
Gradient accumulation advances progress only at optimizer boundaries; resume
starts the bar at the saved step. Per-microbatch JSON metrics remain in
`metrics.jsonl` without printing over the bar. `tqdm` is an explicit dependency.

All 25 CPU tests pass in both installed environments: Transformers 4.45.0
(`nnunet`, 6.270 seconds) and 4.51.0 (`dino`, 5.870 seconds). New integration
checks rerun training into an occupied directory and verify stale rank/DeepSpeed
files are removed, metrics are reset, and fresh optimizer counts are saved.
They also run all three stages with two accumulated microbatches per step,
check rendered bars and resume progress. Existing checkpoint/resume equivalence
and numerical tests continue to pass. Six-GPU runtime remains unverified here.

## FP32 Probe Reuse And DPT Teacher Backbone (2026-10-17)

`write_loss` now creates one detached FP32 vocabulary/norm parameter snapshot
for each model per loss call. All branches, layer pairs, chunks and nonreentrant
checkpoint recomputations share those tensors. Empty write batches allocate no
snapshots. They are scoped to the loss graph and refreshed for the next call,
so DFT optimizer updates cannot leave a persistent stale cache. Both supplied
FP32 head snapshots together occupy approximately 1.67 GiB per rank while live;
this is tensor-size arithmetic, not a measured GPU peak or memory saving.

`VREDModel.forward_backbone` shares the ordinary connector/input preparation
and calls only the Qwen decoder backbone. DPT teacher extraction uses this path,
omitting the unused vocabulary projection and CE. Ordinary model forward,
student CE and DFT teacher final-logit KD retain their previous behavior.
Full-vocabulary FP32 probes, visual tokens, write positions, block pairs,
normalizations and checkpoint formats are unchanged. No new flags are required.

Verification:

- All 27 CPU tests pass on Transformers 4.45.0 / PyTorch 2.4.0 (`nnunet`,
  8.621 seconds) and Transformers 4.51.0 / PyTorch 2.10.0 (`dino`, 7.822 seconds).
- BF16 DPT/DFT comparisons against per-probe weight conversion give exactly
  matching losses, metrics and student gradients for all three write-loss modes.
  Instrumentation confirms only two snapshot preparations per loss, including
  checkpointed backward, and collection of snapshots after releasing the graph.
  A subsequent changed student head is observed by the next loss.
- FP32/BF16 full-teacher versus decoder-only captures match exactly at every
  captured field. DPT losses and connector gradients match exactly. DPT succeeds
  with teacher CausalLM/head forward patched to raise; DFT still calls the full
  teacher forward. Teacher parameters remain gradient-free.
- `nnunet` synthetic DPT/resume/SFT/DFT pipeline passes, including checkpoint
  save/reload with zero logit difference, `(729,1152)` cached-image inference and
  no vision encoder. DFT peak CPU RSS is 722.77 MiB. Report:
  `/tmp/vred-probe-cache-backbone-smoke/smoke.json`.

Commands:

```bash
python -m unittest discover -s vred/tests -v
python -m vred.pipeline_smoke --synthetic --output /tmp/vred-probe-cache-backbone-smoke
```

Use a fresh smoke output directory for an independent verification. CUDA is
unavailable here; real six-GPU throughput and GPU peak-memory changes are not
measured. No numerical approximation or research improvement is claimed.

## Memory, Local Scoring, And Teacher Provenance (2026-10-17)

This section supersedes the earlier "assets absent / no GPU run" statements. The
project assets are present, and a six-GPU DPT run has completed; this entry
records the new audit and the code changes made in response to the SFT OOM.

### Actual run state

- Completed DPT (`checkpoints/vred/dpt`): 2907 optimizer steps, 558,132 examples,
  7,246,875 response tokens, 42.9 GPU-hours, micro 2 / accumulation 16 / six
  ranks, CE 7.14 to about 2.6, peak reserved 22.3 GiB. The saved language model is
  bitwise identical to base Qwen2.5-0.5B; only the connector changed.
- The attempted SFT (`micro 8 / accumulation 2`) produced only `run_config.json`
  and an empty TensorBoard scalar set. The traceback ends in eager-attention
  softmax, but the allocation failure follows dense full-vocabulary logits/CE and
  retained attention matrices; the teacher is not loaded in SFT. No stderr was
  retained, so OOM is inferred rather than logged.
- DFT has not run. `checkpoints/vred/sft` has no model, so a DFT launch currently
  fails regardless.

### SFT memory changes

- `VREDModel.forward_sparse` applies the language head only at shifted response
  positions, returning ordered `[positions, vocabulary]` logits and the same
  mean CE. `logits_kd_sparse` uses those selected logits for final KD.
  `VREDTrainer` uses the sparse path by default (`sparse_logits: true`) and passes
  `sparse=True` through the DeepSpeed engine wrapper; `--no-sparse-logits` restores
  dense CE/KD for comparison. Dense and sparse selected logits are identical on the
  real 0.5B BF16 checkpoint; CE differs by 1.19e-7; the tiny BF16 suite checks
  loss, selected logits, connector gradients and KD.
- SFT may enable `--gradient-checkpointing` (student backbone, SFT only).
  DPT/DFT reject the global flag because the selected effect-capture blocks need
  the ordinary forward; instead `--selective-checkpointing` checkpoints only the
  non-selected student blocks. The selected blocks keep the real attention
  probabilities/post-MLP outputs, and tiny DPT/DFT checks give identical loss and
  gradients to the non-checkpointed path. This trades about 30% extra recompute
  for substantial activation-memory savings; it is off by default.
- SDPA was measured and rejected: on the real 0.5B BF16 DPT checkpoint, eager vs
  SDPA selected logits differ by up to 1.372 and caption CE by 0.01635. Training
  and evaluation remain eager for cross-stage numerical consistency.
- Recommended SFT retry after these changes: keep micro 8 / accumulation 2 first;
  if still OOM use micro 4 / accumulation 4 (same global 96), optionally
  `--gradient-checkpointing`, and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
  GPU peak memory of the new path was not measured here (CUDA unavailable).

### DPT write coefficient evidence

- The completed DPT run used `lambda_write=1`, vocabulary sum SmoothL1. Aggregated
  by optimizer step, write loss fell from about 0.00437 to about 0.00358 by
  step 100 (minimum 0.00286 at step 58), then rose to about 0.00420 and stayed
  there through step 2907 while CE kept falling. The write term was about 0.07% to
  0.16% of CE, and the late trajectory shows CE dominating the shared connector.
- New bounded `vred.gradient_probe` (one real DPT caption at the final connector,
  CPU BF16, lambda=1): `||g_CE||=6.6444`, `||g_write||=0.014456`, ratio 0.002176;
  write/CE scalar values 0.002259 / 3.5953. That single example was a low-ratio
  outlier; an eight-example GPU run gave per-example ratios 0.00189-0.00758,
  mean ratio 0.004647, mean CE grad 8.2368, mean write grad 0.035858. The
  per-example lambda for equal gradients ranged 132-529 (mean 268.3); the
  ratio-of-means/median is about 230. Lambda for 10% of CE gradient is about
  23-27, for 1% about 2.3-2.7. Lambda=100 is about 44-46% of the CE gradient,
  not 22%; lambda=50 about 22-23%. A scalar-matched lambda is about 673 on the
  eight-example mix (1592 on the first example), illustrating why scalar matching
  is not a stable rule. Do not choose lambda by scalar equality: run the probe
  on more examples and validate a small lambda sweep on held-out DPT data and
  post-SFT behavior. The background's provisional sweep {0.1, 1, 10} gives at
  most about 4.6% of the CE gradient at this state (lambda=10); a wider sweep is
  required. A safe first pilot is lambda ~10-25 (about 4-11% of CE gradient);
  lambda ~230 is gradient-equal and should be treated as an aggressive endpoint.

### Teacher provenance

- VRED ignores the teacher checkpoint's `vision_tower` and feeds the shared
  `(729,1152)` cached base-SigLIP2 features. The main teacher checkpoint's saved
  tower is default-initialized (all 448 tensors differ from the base; LN weight
  exactly 1, biases zero). `checkpoints/eval_fixed/ta-qwen25-3b-stage0_2-lora-finetune`
  is a byte-identical copy apart from its tower, which equals the base SigLIP2
  bf16 weights exactly. A one-example teacher probe gave CE 5.0848 with cached
  base features, 7.5211 with zero features, and 9.6587 with the main checkpoint's
  saved tower features. The teacher computation used by VRED is therefore the
  intended one; use `eval_fixed` only for legacy image-based evaluation that
  loads the tower.
- The teacher `trainer_state` indicates about 665k SFT-style examples (6926 steps,
  global batch about 96, train loss 0.853), while DPT teacher targets are computed
  on caption/pretrain prompts. This is a design/data-distribution caveat, not a
  loader bug.
- Legacy `VisionTower.forward` dropped the first visual token (`[:, 1:]`, 728
  tokens); VRED keeps all 729. New `vred.visual_token_check` compares teacher CE
  for 729 vs 728 (drop first and drop last). The comparison has not been run in
  this CUDA-free session.

### Local scoring and benchmark fixes

- VQAv2 is now scored locally on `llava_vqav2_mscoco_val2014.jsonl` and
  `v2_mscoco_val2014_annotations.json` with the official EvalAI normalization
  and `min(matches/3, 1)` soft accuracy. This is val2014, not test-dev.
- GQA is scored locally with balanced exact match from
  `questions1.2/testdev_balanced_questions.json`, including structural/semantic
  breakdowns; the old official `eval/eval.py` scene-graph/choices assets are no
  longer required.
- MME `calculation.py` stdout is parsed into per-category/per-task scores; MMMU
  `main_eval_only.py` stdout is parsed into category and overall metrics. Both
  write `scores.json`.
- `--action list` reports `scorer_ready` and missing local prerequisites; run/score
  fail before generation or scoring when they are absent. MMBench EN/CN dev are
  scored locally from the TSV answer column; `openpyxl` is needed only for a
  `scoring: submission` MMBench spec. VizWiz val is scored locally from
  `llava_val.jsonl` + `Annotations/val.json`; VizWiz test remains submission-only
  because its answers are not local. The evaluation feature cache remains a
  separate external prerequisite.
- Real-data checker: gold answers as predictions give GQA 100.0% (12,578
  balanced) and VQAv2 87.478% (first of ten gold answers, matching the expected
  formula over 214,354 val2014 questions).

### Verification

All 35 CPU tests pass in `nnunet` (PyTorch 2.4.0, Transformers 4.45.0,
tensorboard 2.20.0, openpyxl present). `dino` currently lacks `tensorboard`
(and `openpyxl`), so its TensorBoard/CLI tests fail for environment reasons.
New tests cover sparse dense-equivalence and KD, the DeepSpeed-wrapper sparse
dispatch, SFT-only gradient checkpointing, selective DPT/DFT checkpointing
equivalence, chunked FP32 probe heads and chunked write_loss, VQAv2/GQA local
scorers, MME/MMMU parsing, benchmark prerequisites and the updated submission
count.

Reproduction:

```bash
export TRITON_CACHE_DIR=/tmp/triton-cache-vred
/home/inspur/miniconda3/envs/nnunet/bin/python -m unittest discover -s vred/tests -v
/home/inspur/miniconda3/envs/nnunet/bin/python -m vred.evaluate --action list
/home/inspur/miniconda3/envs/nnunet/bin/python -m vred.gradient_probe --examples 1 --device cpu --input-model checkpoints/vred/dpt --output /tmp/vred-gradient-probe.json
```

## Chunked FP32 Probe Head (2026-10-18)

The previous design copied the full detached vocabulary head to FP32 once per
model per write-loss call: about 1.19 GiB for the 3B teacher and 0.54 GiB for the
0.5B student (1.67 GiB combined while the graph was live). That section is now
superseded.

- `semantic_probe` keeps the final RMSNorm weight in FP32 (small) and reads the
  detached head/bias in vocabulary chunks, converting each chunk to FP32 inside
  the probe. It accumulates the full `[positions, vocabulary]` FP32 logits and
  softmax exactly as before. The default `probe_vocab_chunk_size=16384` gives a
  teacher chunk of about 128 MiB and a student chunk of about 56 MiB; the full
  head snapshot is no longer materialized. Lowering the chunk size lowers the
  head-chunk peak; raising it reduces conversion iterations.
- `write_loss` accepts `vocab_chunk_size` and passes it to every student/teacher
  probe, including checkpointed recomputation. The trainer reads
  `probe_vocab_chunk_size` from `training.json`; the CLI/launcher expose
  `--probe-vocab-chunk-size`. `--no-sparse-logits` and all previous loss options
  are unchanged.
- The FP32 probe arithmetic and full-vocabulary reductions are preserved. New
  tests compare the chunked probe with a full FP32 head reference at chunk sizes
  1, 7, 97 and 16384 (values within 1e-7; state gradients within 1e-7) and compare
  `write_loss` at vocabulary chunk 7 with the default. The probe-parameter reuse
  test still checks one parameter preparation per model per loss and refresh on
  the next call; it no longer expects a full FP32 head snapshot.
- On the supplied vocabulary this removes about 1.5-1.6 GiB of transient FP32
  head memory per rank at the cost of repeating the (small) head conversion per
  probe. The chunk conversion is memory-bound; the teacher forward remains
  dominant. GPU peak memory was not measured in this CUDA-free session.

All 35 CPU tests pass in `nnunet`; the tests cover the chunked probe, chunked
`write_loss`, sparse/dense equivalence, selective DPT/DFT checkpointing,
SFT-only gradient checkpointing, local VQAv2/GQA scorers, MME/MMMU parsing and
benchmark prerequisites.

## Response-Position CE Chunking (2026-10-18)

The sparse CE path still materialized the full `[all_response_positions,
151936]` head output and FP32 CE/softmax for every microbatch. For long SFT/DFT
responses this can be a peak even though dense `[batch, sequence, 151936]`
logits are gone (for N response positions the FP32 CE plus softmax is roughly
`2 * N * 151936 * 4` bytes; N=1000 is about 1.2 GiB, N=4000 about 4.9 GiB).

- `VREDModel.forward_sparse` now accepts `ce_chunk_size`. With a positive value
  it computes the head and `cross_entropy(reduction="sum")` per chunk and returns
  the exact mean over all response positions. For SFT/DPT the trainer sets
  `return_logits=False`, so no full `[N, vocabulary]` logits are materialized.
  For DFT `return_logits=True` is still required for final KD; the student and
  teacher BF16 logits at response positions are materialized, while the CE FP32
  tensors are chunked and final KD is already chunked over positions.
- Default `ce_chunk_size=32` in `training.json`; CLI/launcher
  `--ce-chunk-size`; `0` restores the all-position CE. The trainer passes it to
  the student forward; the teacher's sparse logits are unaffected.
- Real 0.5B SFT mixed batch (112 response positions): full CE 1.5532658100;
  chunk 32 1.5532656908 (difference 1.2e-7) with the same connector gradient
  norm 2.7456455. Very small chunks (1 or 7) changed the BF16 head GEMM shapes
  enough to move the connector gradient norm by about 0.15% (2.7414551 vs
  2.7456455); use chunk >= 32 for close fidelity to the full path. The tiny
  FP32/BF16 tests compare chunk 1/7/32 against dense within the declared
  tolerances.
- This caps the CE/softmax peak at `ce_chunk_size * vocabulary * 4` bytes
  (32 x 151936 x 4 = about 19 MiB) instead of the full response-position tensor.
  It does not reduce teacher weights, backbone activations, backward
  recomputation, DeepSpeed optimizer/gradient buckets, or the DFT BF16 logits
  needed for KD; those must be measured per phase. The earlier OOM in this
  session was a foreign process holding 20.72 GiB on one GPU, while the SFT
  process itself had only 1.81 GiB allocated by PyTorch; that was not a CE peak.
- All 35 CPU tests pass; the sparse CE test also compares chunked CE values and
  connector gradients with the dense path.

## Online Evaluation Vision Encoder (2026-10-19)

Evaluation no longer requires a precomputed evaluation feature cache. The
training path remains cache-only and still never loads or imports a vision
encoder.

- `vred/vision.py` is an evaluation-only module: it loads the base
  `SiglipVisionModel`/`AutoImageProcessor` from
  `foundation.json:vision_tower` (default
  `pretrained_hg/siglip2-so400m-patch14-384`), applies the same resize/crop and
  `select_patch_features` logic as `cache_vision_features.py`, and returns a
  `(729, 1152)` feature. `cache_vision_features.py` now imports the shared
  `infer_expected_tokens`/`select_patch_features` so training-cache preparation
  and online evaluation cannot drift.
- `vred.evaluate` supports both modes: `--cache-index` keeps the old offline
  path, while omitting it and passing `--vision-tower` (or relying on
  `assets.vision_tower`) encodes each image at generation time. `--image-root`
  overrides the per-benchmark image root; `--vision-dtype fp16|bf16` selects the
  loader dtype. The report records `online_vision`, `vision_tower`, and
  `image_root`.
- `configs/evaluation.json` now declares the image root for each benchmark
  (GQA images, ScienceQA images/test, TextVQA train_images, POPE val2014,
  MME release root, MMMU all_images, VQAv2 val2014, VizWiz val, and the
  optional extracted MMBench folders). MMBench/MMBench-CN embedded TSV base64
  images are decoded directly, so online mode needs no `--image-map` and no
  extracted image folder. `--image-map` is still required for cached embedded
  TSV images.
- `vred.infer` also accepts `--image` + `--vision-tower` for a single online
  image; `--image-key`/`--cache-index` remain for cached inference.
- Training is untouched: `vred/train.py`, `vred/trainer.py`, `vred/model.py`,
  and `VREDModel` do not import `vred.vision` or load SigLIP. `foundation.json`'s
  `vision_tower` key is read only by evaluation/inference. A CPU test confirms
  the training `VREDModel` still has no `vision_tower` attribute.
- `vred.evaluate` online tests use a fake encoder to exercise the no-cache
  dispatch, image-root resolution, report fields, and `answers.jsonl` metadata;
  a separate test checks MMBench TSV base64 retention/decoding. All 45 CPU tests
  pass.
- The completed DFT run (6927 steps, 1 epoch, loss 1.305 -> 1.185, CE 1.036 ->
  0.986, final KD 0.250 -> 0.184, no NaN, peak reserved 19.92 GiB) loads from
  `checkpoints/vred/dft`; the ten benchmark commands can now omit the evaluation
  feature cache entirely.

## Six-GPU Sequential-Benchmark Scheduler (2026-10-19)

`vred.evaluate --action run-all` runs all ten benchmarks with one command and
automatically parallelizes each benchmark across all GPU workers:

- `--gpus 6` (default) / `--gpu-ids 0,1,2,3,4,5` launches six shard processes of
  the selected benchmark. `--shard-index`/`--shard-count` are passed to the
  existing `--action run`; each shard writes
  `answers.shard-<i>-of-<n>.jsonl`/`result.shard-<i>-of-<n>.json` and never
  scores or collects on its own.
- When all shards of benchmark N complete, `run-all` merges their answers in the
  original question order, verifies coverage (no duplicate/missing/extra IDs),
  writes `answers.jsonl`/`result.json`, and runs the scorer/exporter once. This
  finalization runs in a background CPU thread while the same six GPU workers
  immediately start benchmark N+1; a final `collect` runs after all threads.
- The shard/merge test covers two cached shards, original-order merge, coverage,
  and local TextVQA scoring. A manual two-worker `run-all textvqa` smoke produced
  `answers.jsonl`, `result.json`, `scorer.log`, both shard answers/reports/logs,
  and `results.json`. All 38 CPU tests pass.
- The command remains cache-only for training; evaluation shards either use the
  optional precomputed index or the online SigLIP2 encoder described above.

## Evaluation Batch Size (2026-10-19)

Evaluation previously ran exactly one question/image at a time (batch size 1);
the only batched image path was the training feature cache builder's
`--batch-size`. `vred.evaluate` now exposes `--eval-batch-size` (default 1):

- Online SigLIP2 encoding uses `OnlineVisionEncoder.encode_batch`, so a batch of
  images is preprocessed and encoded in one vision forward.
- Qwen generation uses `generate_cached_batch`: prompts are left-padded, the
  attention mask marks real tokens, image features are inserted at each sample's
  padded image span, and positions come from the attention mask. With batch 1
  this is the same path as before; the single-question `generate_cached` wraps
  it.
- The flag is per shard: `--action run-all --gpus 6 --eval-batch-size 8` has six
  shards each using batch 8, i.e. 48 concurrent generations/vision images. The
  default 1 preserves the exact previous behavior.
- `vred.evaluate --action run` records `eval_batch_size` in `result.json`; the
  run-all scheduler forwards the flag to every shard. `--action score`/`collect`
  do not need it.
- A CPU test compares batched generation for an image prompt and a shorter
  text-only prompt against the single-question results (token ids, text, prompt
  lengths, cached-image flag); the online-eval test exercises `encode_batch`; all
  39 tests pass. The DFT checkpoint remains `checkpoints/vred/dft`; training is
  still cache-only and never loads the vision encoder.

## Evaluation Progress Bars (2026-10-19)

Each `vred.evaluate --action run` now wraps its batch loop in tqdm with
`total=len(items)`, a benchmark/shard description, questions/s, elapsed and
remaining/ETA (`elapsed<remaining`). Answers are flushed after every batch so
external progress observers see movement.

`--action run-all` starts the six shard subprocesses, then displays one
aggregate tqdm bar per benchmark (`<benchmark> [6 GPUs]`) whose position is the
sum of completed answer lines across the six shard files. The bar shows ETA and
questions/s; it is refreshed about once per second and closes exactly when the
shard processes exit. Each shard subprocess also has its own tqdm bar in
`shard-<gpu>.log`, so per-GPU progress remains inspectable. Merge/scoring then
runs in the background while the next benchmark's aggregate bar starts.
All 39 CPU tests pass; the shard/merge, batch-generation, online-encoder and
run-all textvqa smoke are unchanged in behavior apart from the progress output.

## Per-Benchmark Terminal Score Summary (2026-10-19)

After a benchmark's merge/scoring completes, `vred.evaluate` prints **exactly
one line**:

```text
gqa: accuracy=62.4
pope: Accuracy=... Precision=... Recall=... F1=...
mme: total=...
mmmu: overall_acc=...
vizwiz: accuracy=59.64
```

- The line is emitted after the single-benchmark `--action run`/`--action score`
  and by each `--action run-all` background finalizer.
- Primary metrics are selected from `scores.json` or, when absent, parsed labels
  from `scorer.log` (TextVQA/POPE). Non-score labels such as `Samples` are
  filtered out.
- The full nested `scores.json`/`results.json` is written to disk but no longer
  dumped to the terminal. `--limit` prints `subset_only (no official score)`;
  submission-only benchmarks print `submission_required`; failures print
  `scoring_failed <error>`.
- The textvqa adapter test captures the `textvqa:` line; all 39 CPU tests pass.

## Duplicate Question IDs In Sharded TextVQA (2026-10-19)

The six-GPU TextVQA merge failed with `duplicate question id
181f00d3ee2b2076 across shards`. Root cause: TextVQA has 5,000 questions but only
3,166 unique `question_id` values; `question_id` is the image id and one image
can carry multiple questions. The old shard merge used `question_id` as its dict
key, so valid repeated ids looked like duplicates.

- Every answer now records `benchmark_index`, the original benchmark row
  position. `merge_shards` keys and orders by `benchmark_index`, validates
  row coverage (missing/extra positions), and only then checks the ordered
  `question_id` list, which may legitimately contain duplicates.
- The TextVQA scorer already keys annotations by `(image_id, question)`, so it
  scores repeated image ids correctly once the rows are restored in order.
- Old shard files have no `benchmark_index`; the merge now raises an explicit
  "rerun the shards" error instead of a misleading duplicate-id error. Remove
  the affected benchmark directory and rerun it.
- Added a duplicate-question-id shard/merge test; all 40 CPU tests pass.

## MME Converter Prompt Fix (2026-10-19)

The sharded MME run generated 2,374 answers and merged them, but scoring failed
in `convert_answer_to_mme.py` with:

```text
KeyError: ('code_reasoning', '0020.txt',
           '<image>\nIs a python code shown in the picture?  Please answer yes or no.')
```

The MME converter builds ground truth as
`GT[(category, file, question)]` from the release `.txt` files and looks up
`answer["prompt"]`. Our generic image prompt prefixed `"<image>\n"`, while the
GT question is the raw text, so the lookup missed.

- For `mme`, the answer `prompt` is now the raw question text after removing the
  model's `<image>` placeholder (same string the converter expects). TextVQA
  still records its original OCR prompt; other benchmarks keep the generic
  image-prefixed prompt.
- The converter's instruction stripping and `Please answer yes or no.` handling
  then match the GT key.
- MME has no unit-level score test that runs the external converter; the fix is
  covered by the real runbook. All 40 CPU tests still pass.
- To rerun MME after this fix, remove the stale generated answers and converter
  outputs first: `results/vred-dft/mme`,
  `eval_dataset/eval/MME/answers/vred_vred-dft.jsonl`, and
  `eval_dataset/eval/MME/eval_tool/answers/vred_vred-dft/`.

## Overwrite Instead Of File-Exists Errors (2026-10-19)

Rerunning a benchmark no longer fails when generated/scored artifacts already
exist. Removed the two guards:

- `vred.evaluate --action run`: `answers.jsonl`/`answers.shard-*.jsonl` are
  opened with `"w"`, so existing answers are truncated and regenerated.
- MME `score_or_export`: the fixed converter input
  `eval_dataset/eval/MME/answers/vred_<run>.jsonl` is copied over the old file
  instead of raising `FileExistsError`; the converter also clears/rewrites its
  `eval_tool/answers/<experiment>/` text files.

`scorer.log` is now truncated at the start of `score_or_export`, so stale
scorer output cannot be appended or parsed as the current score. `result.json`,
`scores.json`, `submission.json`, `answers.xlsx`, shard files, and
`results.json`/`results.csv` are likewise overwritten. The textvqa adapter test
now runs the same command twice against the same output and checks both
`textvqa:` lines; all 40 CPU tests pass.


## Optional DFT Write Controls (latest)

Added opt-in write controls that leave the previous DPT/SFT/DFT path unchanged
when no related flag is supplied:

- `write_position_scope`: `response` (default), `prompt`, or `prompt_response`.
  Prompt positions are image-conditioned, have no CE label, and causally precede
  the response; prompt_response is the union.
- `write_effect_sampling`: `uniform` (default) or `stratified`. Stratified
  requires `write_position_limit` and draws equal counts from low/mid/high
  per-example strata of the mean teacher hidden effect norm, instead of
  uniformly sampling all eligible positions.
- `write_lambda_schedule`: `constant` (default) or `warmup_cosine`. The latter
  uses `warmup_ratio` for a linear ramp and then a cosine decay to zero; it only
  scales `lambda_write`.
- `write_loss_type=reverse_kl` / `--reverse-kl`: reverse KL with the same eta
  smoothing, in addition to the existing SmoothL1, forward effect-KL, and
  effect-JS losses.

`vred/trainer.py` captures teacher effects at all candidates for stratified
sampling, selects positions, then slices the teacher effects before the student
capture and write loss. `vred/train.py` sets the schedule total and current
optimizer step. `vred/gradient_probe.py` uses the configured scope. New tests
cover prompt/prompt_response scopes, stratified selection, reverse KL, schedule
values/weighted write, and launcher flag resolution. All 49 CPU tests pass. With
no new flag, the existing loss/gradient tests still pass bit-identically.

### Optional online vision-tower training

Added an opt-in training path that leaves the cached path unchanged when
`--online-vision` is absent:

- `CachedDataset(..., online=True, image_root=...)` opens the referenced image
  instead of reading a `.npy` feature; `Collator` carries PIL images in
  `Batch.images`. `attach_online_features` encodes only non-text rows and
  replaces them with tensors before the connector.
- `OnlineVisionEncoder.encode_batch_train` uses `torch.no_grad` (not
  `inference_mode`) and returns detached ordinary tensors; the tower is frozen.
  `vred.train.make_online_encoder` resolves `foundation.json:vision_tower`,
  freezes it, and keeps it outside DeepSpeed.
- `--online-vision` / `--no-online-vision` and `--online-vision-root` are
  forwarded by the launcher. The default SFT/DFT data path is
  `sft_json` (`mllm_datasets/text_files/llava_v1_5_mix665k_clean.json`); the
  image root defaults to the parent of `text_files` (`mllm_datasets`).
- Online preflight checks image existence instead of cache keys; no cache index
  is read. `run_config.json` records `online_vision` and `online_image_root`.
- Tests cover online dataset/collator/preflight, feature attachment, launcher
  flag resolution and a one-step training run with a fake frozen encoder.
  All 53 CPU tests pass; without `--online-vision`, the existing tests remain
  bit-identical.


## Optional Effect-Weighted Final-Logit KD (latest)

Added an opt-in DFT term that leaves the original path unchanged when
`--effect-weighted-kd` is absent:

- `--effect-weighted-kd` / `--no-effect-weighted-kd` and
  `--lambda-effect-kd` are forwarded through the launcher. The term is DFT-only
  and requires sparse logits; DPT/SFT or dense logits are rejected.
- The teacher plus forward is the ordinary image-conditioned pass. A second
  no-grad teacher forward uses the same text and visual positions with the
  pre-connector visual features zeroed; effect-capture hooks are removed before
  this second pass so write targets are not overwritten.
- `effect_weighted_kd_sparse` computes
  `mean_t(w_t * KL(p_t+ || p_s+))`, with
  `w_t = ||p_t+ - p_t-||_1` and `p = softmax(z / tau)`. Weights are detached and
  not normalized; the term is scaled by the same response-count normalization as
  ordinary KD. The ordinary KD should stay a small anchor, not be kept at the
  same scale as the effect term: the recommended first run is
  `lambda_logits=0.2`, `lambda_effect_kd=1.0` (1:5). `lambda_effect_kd=2` is not
  the default recommendation; it is appropriate only when the raw
  `effect_kd / logits_kd` ratio is below about 0.1. Compare weighted terms
  (`0.2 * logits_kd` vs `1.0 * effect_kd`), not just raw losses.
- Tests cover the formula, zero-effect fallback, opt-in trainer path, default-off
  metrics, launcher forwarding, flag conflict, stage restriction and
  sparse-logit restriction. All 56 CPU tests pass.

## Hidden-Effect Floored KD (latest)

Added an opt-in alternative to the logits-difference L_ewkd weight source. The
default path is unchanged when `--effect-weighted-kd` is absent; when it is
present without a source flag, the existing logits-difference behavior is used.

- `--effect-kd-weight-source logits|hidden` and `--effect-kd-floor FLOAT` are
  forwarded by the launcher. Hidden mode is DFT + sparse only, requires
  `write_position_scope=response`, no `write_position_limit`, and uniform write
  sampling; the launcher and trainer reject other combinations.
- Hidden weights reuse the teacher plus `EffectCapture` and require no second
  teacher forward. `response_effect_scores` aligns teacher per-position hidden
  effect magnitudes `||u_plus-u_minus||_2` (averaged over layer pairs) to every
  shifted response position; text-only positions have score zero.
- `floored_effect_weights` computes
  `w_t = floor + (1-floor) * s_t / mean(s)`; for nonzero effects the mean weight
  is one, and all-zero effects receive the floor. `weighted_kd_sparse` computes
  `mean_t(w_t * KL(p_t+ || p_s+))`. Hidden mode skips the teacher-minus logits
  forward and, when `lambda_logits=0`, also skips ordinary KD.
- Recommended hidden-mode launch: `lambda_logits=0`, `lambda_effect_kd=0.5`,
  `--effect-kd-floor 0.15`, response-scope full write, K9, `micro=2/accum=8`,
  query chunk64. The normalized weights mean the old logits-mode coefficient
  should not be reused blindly.
- Tests cover floored weights (including all-zero), weighted KD, the hidden
  trainer path without a teacher-minus forward, launcher forwarding/defaults,
  and rejection of misaligned hidden-mode combinations. All 58 CPU tests pass.

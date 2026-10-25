# VRED: Setup And Execution

VRED defaults to cached `(729,1152)` features, each model's own GELU connector,
and Qwen language models. Training is cache-only unless the opt-in
`--online-vision` path is selected; evaluation can encode images online with the
base SigLIP2 tower.
Effect extraction preserves the ordinary `VREDModel.forward(batch)`.
Smoke checks establish implementation behavior, not research improvement.

## Setup

Run from the repository root with Python 3.11 and a suitable PyTorch build:

```bash
cd /data/zsx/LLaVA-KD
python3.11 -m venv .venv-vred
. .venv-vred/bin/activate
python -m pip install -r vred/requirements.txt
# On the target CUDA node, install a DeepSpeed version compatible with PyTorch:
python -m pip install deepspeed
```

The complete verified environment here is
`/home/inspur/miniconda3/envs/nnunet/bin/python` (Python 3.10, PyTorch 2.4.0,
Transformers 4.45.0), which also has `tensorboard`, `deepspeed` and `openpyxl`;
all checks below were run with it. The `dino` environment (PyTorch 2.10.0,
Transformers 4.51.0) passes the numerical code but currently lacks
`tensorboard` and `openpyxl`, so the TensorBoard/CLI tests and the optional
MMBench submission export fail there for environment reasons; local MMBench
scoring needs only `pandas`. The legacy root requirements
pin a different Transformers release; use VRED's requirements.
Effect capture supports both two-item and three-item Qwen2 attention returns and
requests attention weights only at selected blocks. An upgrade is not required
for the `too many values to unpack` attention-hook error; use the updated code.
No dependency installation is performed automatically. Model/tokenizer loads
are local and require tokenizer files as well as weights.

Set paths in `vred/configs/foundation.json`: student base, teacher checkpoint/base,
two annotation JSON files, and cache `index.json`. Relative asset paths resolve
against the repository root; absolute paths are supported. The supplied asset
directories are present in this workspace; audit them with `vred.preflight` and
`vred.evaluate --action list` before a long run.

The pinned teacher format is unmerged LLaVA-KD base weights plus root LoRA.
The loader validates and merges adapters once on CPU, then freezes the teacher.
It validates full token-ID semantics, tokenizer backends/templates, special IDs
and vocabulary widths; equal vocabulary sizes alone are insufficient.

Before a long run, audit both training annotation files:

```bash
python -m vred.preflight --output /tmp/vred-coverage.json
# Expensive full encode/array validation, rather than path coverage only:
python -m vred.preflight --validate-arrays --output /tmp/vred-integrity.json
```

Training automatically checks its stage's record structure, cache-key coverage
and referenced-file existence. Array shape/finiteness is checked on each load.
Errors identify records and paths; invalid records are never skipped. Normalized
cache-key collisions and root-escaping mappings are rejected. Repeated image IDs
are logged (expected across instruction questions). Path coverage does not prove
array integrity or online/cache equivalence.

## Three Stages

Launch profiles are `vred/configs/launch/{dpt,sft,dft}.json`. They explicitly set
six GPUs, seed, predecessor, optional same-stage `resume`, shared config paths,
and output directory. Edit them for a new experiment. Shared
`vred/configs/training.json` contains precision, batch/accumulation, learning
rates, losses, temperatures, checkpoint interval and optimizer settings.

| Stage | Trainable | Objective | K | Microbatch/rank | Accumulation | Global batch | LR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DPT | Connector | CE + write | 3 | 1 | 32 | 192 | 1e-3 |
| SFT | Connector + LM | CE | 0 | 1 | 16 | 96 | 2e-5 |
| DFT | Connector + LM | CE + write + final KL | 6 | 1 | 16 | 96 | 2e-5 |

Defaults: seed 1234, BF16 model/FP32 probes and losses, one epoch, AdamW,
weight decay 0, cosine schedule with 0.03 warmup. Write and final-KL coefficients
are 1 where active, otherwise 0. SmoothL1 uses vocabulary **sum** (raw mean also
logged); KL/JS sum their signed-distribution axis. Losses average valid prediction
positions across accumulation/ranks and write loss averages layer pairs.

```bash
python -m vred.launch dpt --dry-run
python -m vred.launch dpt
python -m vred.launch sft
python -m vred.launch dft
```

Important settings can be overridden without editing JSON. The same training
flags work with `python -m vred.train`; launch additionally accepts `--gpus`,
`--profile`, `--config`, `--foundation-config`, `--input-model`, `--output`,
and `--resume`. Flags override JSON, and `--dry-run` prints the resolved settings
and exact subprocess command before any model loading.
Both stage forms are accepted: `python -m vred.launch dpt` and
`python -m vred.launch --stage dpt`. If a shell reports a different launcher
interface, check the loaded file with that same interpreter:
`python -c 'import vred.launch; print(vred.launch.__file__)'`, then run
`python -m vred.launch --help`.

```bash
python -m vred.launch dpt --gpus 6 --seed 1234 --epochs 1 --lr 1e-3 \
  --micro-batch-size 1 --accumulation 32 --pairs 3 --lambda-write 1 \
  --write-loss-type smoothl1 --write-vocab-reduction sum --output checkpoints/vred/dpt
python -m vred.launch sft --gpus 6 --seed 1234 --epochs 1 --lr 2e-5 \
  --micro-batch-size 1 --accumulation 16 \
  --input-model checkpoints/vred/dpt --output checkpoints/vred/sft
python -m vred.launch dft --gpus 6 --seed 1234 --epochs 1 --lr 2e-5 \
  --micro-batch-size 1 --accumulation 16 --pairs 6 --lambda-write 1 --lambda-logits 1 \
  --write-loss-type smoothl1 --write-vocab-reduction sum \
  --input-model checkpoints/vred/sft --output checkpoints/vred/dft
```

| Flags | Meaning |
| --- | --- |
| `--lr`, `--epochs`, `--seed` | Learning rate, training epochs, initialization/data seed |
| `--micro-batch-size`, `--accumulation` | Effective batch = GPUs x microbatch x accumulation |
| `--lambda-write`, `--lambda-logits`, `--pairs` | Stage coefficients and relative layer-pair count |
| `--write-loss-type` | `smoothl1`, `effect_kl`, `effect_js`, or `reverse_kl` (or use `--reverse-kl`) |
| `--write-vocab-reduction` | SmoothL1 `sum` or `mean` |
| `--write-position-scope` | `response` (default), `prompt`, or `prompt_response`; prompt includes image-conditioned prompt tokens that have no CE labels |
| `--write-effect-sampling` | `uniform` (default) or `stratified`; stratified needs `--write-position-limit` and balances low/mid/high teacher hidden-effect positions |
| `--write-lambda-schedule` | `constant` (default) or `warmup_cosine`; the latter uses `--warmup-ratio` for the ramp and decays to zero |
| `--tau-write`, `--tau-logits`, `--smoothl1-beta`, `--effect-kl-eta` | Probe/KD temperature and loss settings |
| `--max-length`, `--query-chunk-size` | Expanded training sequence limit and exact probe query chunks |
| `--ce-chunk-size` | Response positions per student CE chunk (default 32; `0` uses all; caps the `[positions, vocab]` CE tensors) |
| `--probe-vocab-chunk-size` | Vocabulary rows converted to FP32 per probe chunk (default 16384; no full head snapshot) |
| `--write-position-limit` | Write sampling budget per example; `0` restores all positions; required by stratified sampling |
| `--checkpoint-probes`, `--no-checkpoint-probes` | Enable/disable probe-loss recomputation |
| `--sparse-logits`, `--no-sparse-logits` | CE/final KD only at shifted response positions (default enabled; numerically equivalent to dense shifted CE) |
| `--gradient-checkpointing`, `--no-gradient-checkpointing` | Student backbone checkpointing; SFT only |
| `--selective-checkpointing`, `--no-selective-checkpointing` | DPT/DFT: checkpoint only non-selected student blocks; selected effect blocks stay ordinary |
| `--weight-decay`, `--warmup-ratio`, `--max-grad-norm` | Optimizer/scheduler/clipping settings |
| `--max-steps`, `--save-every-steps` | Absolute step cap and checkpoint interval |
| `--num-workers`, `--cpu-threads` | Data loading and CPU computation settings |
| `--student`, `--teacher`, `--teacher-base` | Local base/teacher asset paths |
| `--dpt-json`, `--sft-json`, `--cache-index` | Training data and feature-cache paths |
| `--online-vision`, `--no-online-vision` | Opt in/out of loading the frozen SigLIP2 tower during training (default: cached features) |
| `--online-vision-root` | Image root for online training; default is the parent of the JSON's `text_files` directory |

Precision remains BF16 as specified by the method. SFT rejects write/final-KL
coefficients and layer pairs; DPT rejects final-KL supervision. Resume requires
the same resolved configuration, including any overrides used for the original
run; repeat those flags when resuming. Use fresh outputs for comparisons.

### Optional write controls (off by default)

The write controls below are opt-in. With no related flag, the DPT/SFT/DFT
loss, positions, gradients, and configuration are the previous path. They affect
DPT/DFT write supervision; SFT has zero write loss.

- `--write-position-scope response|prompt|prompt_response`: `response` keeps the
  original response-token positions. `prompt` supervises image-conditioned prompt
  tokens that have no CE label but causally precede the answer. `prompt_response`
  is the union.
- `--write-effect-sampling uniform|stratified`: after choosing a scope, this
  decides which positions receive the write loss when the budget is smaller than
  the candidate set. `uniform` samples uniformly; `stratified` uses the mean
  teacher hidden effect norm `||u_plus-u_minus||_2` across layer pairs, splits
  each example's candidates into low/mid/high strata, and draws equal counts from
  each. It requires `--write-position-limit`; it does not change the loss form or
  `lambda_write`. With a budget covering all candidates both modes are identical;
  stratified only avoids over-representing the strongest-effect tokens.
- `--write-lambda-schedule constant|warmup_cosine`: `constant` is the previous
  behavior. `warmup_cosine` multiplies only `lambda_write` by a linear warmup
  followed by a cosine decay to zero. The warmup fraction is `--warmup-ratio`
  (default 0.03), so changing it also changes the LR schedule.
- `--write-loss-type ... reverse_kl` or `--reverse-kl`: reverse KL
  `KL(student_effect || teacher_effect)` with the same eta smoothing. The default
  remains forward `effect_kl`; reverse KL is a separate option and should not be
  stacked with forward KL.

Choose `lambda_write` by **gradient share**, not by the loss value: aim for the
write-gradient norm to be about 5-10% of the CE-gradient norm. On the DPT probe,
`lambda_write=1` measured `||g_write||/||g_CE||=0.002176`; the current DPT
`lambda_write=25` is therefore about 5.4%. For DFT, start from these ranges with
SmoothL1 `sum` (the current reduction), then verify with the probe or a one-step
gradient measurement:

| Write loss | Starting `lambda_write` | Note |
| --- | --- | --- |
| `smoothl1` + `sum` | 20-50 (current 25) | Raw loss is tiny; loss share is not gradient share |
| `effect_kl` | 0.5-2 | Forward KL; start at 1 |
| `effect_js` | 1-5 | Bounded; start at 2-3 |
| `reverse_kl` | 0.25-1 | Mode-seeking; start at 0.5, use eta=1e-6, monitor collapse |

With `prompt` scope or a small position budget, the write loss is averaged over
fewer positions, so the total write gradient falls roughly with the
selected/candidate ratio; scale `lambda_write` by that ratio as a first
approximation and re-measure rather than exceeding a 10% gradient share. A
`warmup_cosine` schedule should reach the chosen peak after the warmup and decay
to zero over the final part of training; do not compensate a too-small peak by
extending the schedule.

### Optional effect-weighted final-logit KD (off by default)

`--effect-weighted-kd` adds a DFT-only token-level counterfactual KD term. With
no flag, the original CE + write + final-KD path is unchanged. The term requires
DFT and sparse logits (the default); DPT/SFT and dense logits are rejected.

Two weight sources are supported:

- **Logits effect (default, `--effect-kd-weight-source logits`).** The plus
  teacher is the normal image-conditioned forward; the minus teacher uses the
  same text, labels, attention mask, and visual positions, but zeros the
  pre-connector visual features. For each shifted response token,
  `w_t = ||softmax(z_t+ / T) - softmax(z_t- / T)||_1` and
  `L_ewkd = mean_t(w_t * KL(softmax(z_t+ / T) || softmax(z_s+ / T)))`.
  Weights are unnormalized, so the raw term is naturally smaller than ordinary
  KD. This mode costs one extra frozen-teacher forward per step.
- **Hidden effect with floor (`--effect-kd-weight-source hidden`).** Reuses the
  teacher's existing plus `EffectCapture`; for response token `t`,
  `s_t = mean_pair ||u_(l,t)+ - u_(l,t)-||_2`, then
  `w_t = floor + (1 - floor) * s_t / mean(s)` and
  `L_ewkd = mean_t(w_t * KL(softmax(z_t+ / T) || softmax(z_s+ / T)))`.
  Positions without a visual effect (text-only rows, or the unused slots of a
  mixed batch) receive `w_t = floor`. For nonzero effects the mean weight is one,
  so the loss keeps a controlled scale; the floor prevents low-effect tokens from
  disappearing. This mode does **not** run the extra teacher-minus forward and
  is cheaper and lower-memory than the logits mode.

Hidden mode requires the write path to cover every response position, because
the KD weights are aligned with `batch.response_positions`: use
`--write-position-scope response`, do not pass `--write-position-limit`, and use
`write_effect_sampling=uniform`. The launcher and trainer reject other
combinations instead of silently building misaligned weights.

Coefficient guidance:

- Logits mode: keep ordinary KD as a small anchor and do not set both terms near
  one. The recommended first run is `lambda_logits=0.2` and
  `lambda_effect_kd=1.0` (1:5). The unnormalized effect term is usually smaller
  than ordinary KD because `w_t` is below one, so a coefficient of 1 already
  makes it primary. `lambda_effect_kd=2` is too aggressive as a default.
- Hidden mode: the weights are normalized to mean one, so do **not** reuse the
  logits-mode coefficient blindly. Start with `lambda_logits=0` (the term replaces
  ordinary KD), `lambda_effect_kd=0.5`, and `--effect-kd-floor 0.15`; a practical
  range is 0.3-0.7. Compare weighted contributions, not raw losses. If the
  effect term is far below ordinary-KD scale, raise `lambda_effect_kd` toward
  0.7-1.0; if it dominates, lower it toward 0.25-0.3. The floor is in `[0,1)`;
  lowering it concentrates more weight on high-effect tokens, while raising it
  makes the term closer to uniform KD.
- Both modes: log `effect_kd` and `effect_kd_weighted`, and do not choose the
  coefficient by equalizing total scalar losses alone. The hidden mode is a
  follow-up experiment, not a validated replacement for ordinary KD.

The flag surface is:

```text
--effect-weighted-kd / --no-effect-weighted-kd
--lambda-effect-kd FLOAT                 (default 1.0)
--effect-kd-weight-source logits|hidden  (default logits)
--effect-kd-floor FLOAT                  (hidden mode; default 0.1)
```

The default-off metrics do not include `effect_kd`, so the original training
path and its logged metrics are unchanged.

### Optional online vision-tower training (off by default)

Without `--online-vision`, training follows the exact cached path described
above. With `--online-vision`, `vred.train` loads the frozen base SigLIP2 tower
from `foundation.json:vision_tower`, encodes each batch's images on its local GPU
under `torch.no_grad`, and feeds the resulting `(729,1152)` features to the
ordinary connector/backbone path. The tower is never optimized, saved, or passed
to DeepSpeed; only the connector and language model receive gradients.

- The SFT/DFT dataset is still `sft_json`. The default is already
  `mllm_datasets/text_files/llava_v1_5_mix665k_clean.json`; DPT uses `dpt_json`.
  Image paths in those JSON files are relative to the image root.
- By default the image root is derived as the parent of the JSON's `text_files`
  directory, i.e. `mllm_datasets` for the default files. Override it with
  `--online-vision-root mllm_datasets` when using another layout.
- The feature cache and `cache-index` are not read in this mode. Preflight checks
  that every referenced image exists (and decodes it with `--validate-arrays`);
  `run_config.json` records `online_vision=true` and `online_image_root`.
- The image processor/vision tower run on every rank, so this path uses
  additional GPU memory and is slower than reading cached features. Reduce
  `--micro-batch-size` or use 48G/96G cards; keep the global batch and LR
  unchanged. `--num-workers` controls image decoding in the data loader.

```bash
python -m vred.launch sft --gpus 6 --seed 1234 --epochs 1 --lr 2e-5 \
  --micro-batch-size 2 --accumulation 8 --online-vision \
  --input-model checkpoints/vred/dpt --output checkpoints/vred/sft-online
```

Sparse CE/final-KD is enabled by default (`sparse_logits`). The model applies
CE only at shifted response positions instead of materializing
`[batch, sequence, 151936]` logits; the real 0.5B check gives identical selected logits and
a loss difference of 1.2e-7 from the dense path. `ce_chunk_size` (default 32)
further processes response positions in chunks: SFT/DPT do not materialize the
full `[response_positions, vocabulary]` logits at all, and DFT only materializes
the BF16 logits needed for final KD. `--ce-chunk-size 0` restores the
all-response-positions CE. SFT may additionally enable
`--gradient-checkpointing`. DPT/DFT reject the global flag because the selected
effect-capture blocks need the ordinary forward; use `--selective-checkpointing`
there to recompute only non-selected student blocks. This is bit-equivalent in
the tiny DPT/DFT checks and trades roughly 30% extra compute for activation-memory
savings. Probe-loss checkpointing remains independent.

Eager Qwen2 attention remains the default and the requirement for effect capture.
Do not switch training/evaluation to SDPA for memory: on the real 0.5B BF16
model, eager vs SDPA selected logits differ by up to 1.37 and caption CE by
0.0163. `load_model(attn="sdpa")` exists only for explicit experiments.

Each launcher uses the current Python interpreter and `torch.distributed.run`,
one node/six ranks. ZeRO-2 explicitly enables BF16 and disables FP16. The frozen
teacher is replicated outside the student engine; SFT loads no teacher. Outputs
default to `checkpoints/vred/{dpt,sft,dft}` and reload the preceding stage's LM,
connector and tokenizer. Legacy scripts are not modified.

`run_config.json` records dataset/cache counts, tokenizer IDs/vocabulary size,
seed, layer pairs, reductions, resolved configs and trainable parameters/modules.
`metrics.jsonl` records losses, actual token/example exposure, optimizer steps,
timing and memory; `summary.json` records completion. Distributed sampling may
pad a dataset, so exposure and unique corpus size can differ.

Exports and optimizer/RNG/cursor checkpoints are saved every 1000 optimizer steps
and at normal exit (`save_every_steps` config). Resume the latest completed
same-stage checkpoint with:

```bash
python -m vred.launch dpt --resume checkpoints/vred/dpt
```

Resume restores optimizer, scheduler, sampler cursor, model RNG and shared-position
sampler RNG. Stage/config/seed/assets/world size must match. The direct runner's
`--max-steps` is an absolute optimizer-step cap, not additional steps; the
scheduler retains the full configured epoch horizon. Fresh runs overwrite the
contents of an output directory containing `run_config.json`, including previous
checkpoints and logs. Use `--resume` to continue a run instead; resuming in place
preserves its checkpoints and appends metrics. A completion marker rejects interrupted exports. Keep a copy
of a completed checkpoint before resuming in place when an older recovery point
is needed: the latest root export is overwritten.

All three stages show a rank-zero tqdm bar with completed/total optimizer steps,
step speed, elapsed time, ETA, epoch, loss, learning rate and valid tokens/second.
Accumulation advances the bar once per optimizer step; resumed bars start at the
saved step. Full per-microbatch metrics remain in `metrics.jsonl`.

TensorBoard is enabled by default for DPT, SFT and DFT. Install the dependency
in the training environment if needed, then start the viewer:

```bash
python -m pip install tensorboard
tensorboard --logdir checkpoints/vred --port 6006
```

Open `http://localhost:6006`. Each stage writes events under its output directory
in `tensorboard/`; only rank zero writes. Charts include total/CE/raw and weighted
write/final-KL losses, learning rate, epoch, optimizer-step time, global token
throughput, processed examples/tokens and peak CPU/GPU memory. Loss charts
average the microbatch contributions across each accumulation group, using the
same valid-token/rank normalization as training. The x-axis is optimizer steps.
Events flush every 30 seconds, at checkpoints and on writer close.

Use `--tensorboard-log-dir runs/vred/dpt` to override the event directory or
`--no-tensorboard` to disable monitoring; both flags work with `vred.launch` and
`vred.train`. Give each independent run/stage a distinct custom directory.
Resume retains completed-step history and hides events beyond the restored
checkpoint. A fresh run resets the default event directory with the other
output files; reused custom directories hide old scalar history starting at
step zero. Logging flags can change when resuming without altering training
configuration. `run_config.json` records the resolved event path.

Write-loss probes keep the final RMSNorm weight in FP32 (small) and convert the
detached vocabulary head to FP32 **per vocabulary chunk** while computing the
probe logits. No full `151936 x hidden` FP32 head copy is materialized: the
teacher chunk is about 128 MiB and the student chunk about 56 MiB at the default
`probe_vocab_chunk_size=16384`, instead of the previous ≈1.67 GiB combined
snapshot. FP32 probe arithmetic, the full vocabulary, all selected positions and
the loss reductions are preserved; the chunked probe matches a full-head FP32
reference within 1e-7 (and gradients within 1e-7) in the tests. The chunk
conversion is repeated per probe; it is memory-bound but small relative to the
teacher forward. Lower `--probe-vocab-chunk-size` if the head chunks dominate
memory; raise it to reduce conversion iterations. The full head is re-read from
the model on every loss call, so a DFT optimizer update cannot leave a stale
probe cache.

DPT teacher targets use only the connector and decoder backbone, bypassing its
unused final vocabulary projection and CE. The student uses sparse shifted CE by
default; DFT uses sparse teacher/student logits at the same response positions for
final-logit KD. `--no-sparse-logits` restores the dense path for numerical
comparisons. Both optimizations apply with the existing launch commands and
checkpoint format.

## Checks And Bounded Smoke

```bash
python -m unittest discover -s vred/tests -v
# Requires the real project assets:
python -m vred.pipeline_smoke --output /tmp/vred-real-pipeline-smoke
# Explicit synthetic alternative when project assets are unavailable:
python -m vred.pipeline_smoke --synthetic --output /tmp/vred-synthetic-pipeline-smoke
```

Two bounded real-asset diagnostics:

```bash
# Teacher CE with 729 cached visual tokens vs legacy-style 728 (drop first/last).
python -m vred.visual_token_check --examples 8 --device cuda --output /tmp/vred-token-check.json
# CE vs write connector gradient norms at the current DPT state; lambda extrapolations.
python -m vred.gradient_probe --examples 4 --input-model checkpoints/vred/dpt \
  --device cuda --output /tmp/vred-gradient-probe.json
```

The first is the pending check for the legacy `[:, 1:]` 728-token teacher input;
the second is for choosing `lambda_write` from gradient evidence rather than from
equal scalar losses (the project background forbids the latter).

Use a fresh output directory. CPU smoke runs DPT optimizer steps and resume,
reloads DPT into SFT and SFT into DFT, exports/reloads, checks exact forward
agreement, and generates from a cached image. Synthetic mode creates labeled tiny
six-block checkpoints/tokenizers/annotations and an FP16 `(729,1152)` cache; it
still runs the real CLI and LoRA loader. Its config uses two mixed-example epochs,
microbatch two and zero warmup so bounded stages update weights. Synthetic text
has no scientific interpretation. CPU AdamW updates BF16 parameters; intended
DeepSpeed training has FP32 optimizer master weights.

Prompt-02 checks remain available when real assets are restored:

```bash
python -m vred.smoke --tiny-only
python -m vred.stage_smoke --real-models --report /tmp/vred-real-stages.json
```

## Bounded Ablations

```bash
python -m vred.diagnose --input-model checkpoints/vred/sft --examples 2 --output /tmp/vred-diagnostic
# Without project assets:
python -m vred.diagnose --synthetic --examples 2 --output /tmp/vred-synthetic-diagnostic
```

One bounded command compares nine DFT variants: CE-only, CE+final KL, CE+write
for K=3/6 and each of SmoothL1/Effect-KL/Effect-JS, and CE+final KL+write. Every
variant resets to identical initial weights/seed/optimizer and sees the same
first N clean **training** examples once in the same order (N=2, hard cap 8).
The teacher is fixed. `metrics.json`/`metrics.csv` include positions/exposure,
coefficients, raw/weighted losses, gradient norms, time and memory. Timing is
diagnostic only: this reference computes write targets even for zero coefficients.
No benchmark examples are used to choose losses or coefficients.

## Cached Inference

```bash
python -m vred.infer --model checkpoints/vred/dft \
  --image-key coco/train2017/000000033471.jpg --question 'What is shown?' \
  --max-new-tokens 32 --output /tmp/vred-inference.json
```

Use an actual key from your cache index; this example key is illustrative.
Pass `--cache-index` for evaluation features, or omit `--image-key` for text only.
Online single-image inference is also supported without a cache:

```bash
python -m vred.infer --model checkpoints/vred/dft \
  --image /path/to/image.jpg --vision-tower pretrained_hg/siglip2-so400m-patch14-384 \
  --question 'What is shown?' --max-new-tokens 32
```

Generation uses the same Qwen base instruction template and all 729 features.
The online path loads SigLIP2 in the inference process only; `VREDModel` never
loads a vision tower. Training remains cache-only unless `--online-vision` is
explicitly selected. Useful answers require a trained model.

## Ten Benchmarks

`vred/configs/evaluation.json` names the ten legacy layouts and their local image
roots. Set its root (default `eval_dataset/eval`), or pass `--root`. Evaluation
loads the base SigLIP2 tower and encodes each image at generation time
(`--vision-tower`, default `foundation.json:vision_tower`); **no evaluation
feature cache is required**. Default training remains cache-only: `vred.train`,
`vred.trainer`, `vred.model`, and `VREDModel` never load or import a vision
encoder. The opt-in `--online-vision` training flag described above is the only
exception.

One command runs all ten benchmarks: each benchmark is sharded across all
`--gpus` workers on GPUs `0..gpus-1`; when its shards finish, the next benchmark
starts on all GPUs immediately while the previous merge/scoring runs on CPU in
the background. A final collect writes the combined results; when
`--benchmarks all` (also the default) is used without `--limit`, it then
automatically writes `<output>/metrics.json`.

```bash
MODEL=checkpoints/vred/dft
VISION_TOWER=pretrained_hg/siglip2-so400m-patch14-384
OUT=results/vred-dft
python -m vred.evaluate --action run-all --gpus 6 --benchmarks all \
  --eval-batch-size 8 \
  --model "$MODEL" --vision-tower "$VISION_TOWER" --output "$OUT"
```

`--eval-batch-size N` (default 1) batches both online SigLIP2 image encoding and
Qwen generation with left padding; each of the six shards uses N, so the
effective concurrent batch is `6 * N`. The default 1 reproduces the previous
one-question-at-a-time behavior exactly; raise it until GPU memory or latency
stops improving.

Every `run` shows a tqdm bar with elapsed/remaining (ETA) and questions/s.
`--action run-all` shows one aggregate bar per benchmark across its six shards
(`<benchmark> [6 GPUs]: ... [elapsed<remaining]`); each shard's own bar is also
written to `shard-<gpu>.log`. The aggregate bar is driven by completed answer
lines, so it advances as soon as a shard flushes its current batch.

When a benchmark finishes scoring, the terminal prints exactly one line with
the primary score(s), for example:

```text
gqa: accuracy=62.4
pope: Accuracy=... Precision=... Recall=... F1=...
mme: total=...
mmmu: overall_acc=...
vizwiz: accuracy=59.64
mmbench: accuracy=68.45
mmbench_cn: accuracy=66.27
```

After all ten benchmarks have local results, build the compact metrics JSON:

```bash
python -m vred.evaluate --action summary --output "$OUT"
```

`--action run-all --benchmarks all` already runs the same step automatically
after its final collect (except a `--limit` diagnostic run), so this standalone
command is for reruns or for points where the ten results were produced
separately. It writes `$OUT/metrics.json` and prints the same JSON. The file
contains only
the ten benchmark metrics plus `avg10`/`avg7` (no statuses, counts, model names,
or timestamps). MMMU is scaled from its 0-1 `overall.acc` to percent
(`0.302 -> 30.2`); MME is `total / 20` (`1509.488 -> 75.4744`); POPE is the mean
of its per-category `Accuracy` values (fractions converted to percent);
TextVQA uses its printed `Accuracy`; ScienceQA uses `acc`. `avg10` averages all
ten. `avg7` excludes VQAv2, POPE, and MMMU, so it averages GQA, VizWiz,
ScienceQA, TextVQA, MME, MMBench, and MMBench-CN. The action is strict: if any
benchmark has no local metric (for example an old `submission_required` or
`not_run` result), it raises a `ValueError` listing the missing benchmarks
instead of averaging a subset.

Scores are read from `scores.json` or the legacy `scorer.log`; non-score labels
such as `Samples` are omitted. The full nested `scores.json`/`results.json` is
still written under `results/vred-dft/` but is not dumped to the terminal.
A subset (`--limit`) prints `subset_only`; no score is invented.
The summary is printed both by a single
`--action run` and by each background finalizer in `--action run-all`. Use `--benchmarks vqav2,gqa` for a subset;
`--gpu-ids 0,1,2,3,4,5` to choose physical devices; `--limit 32` for a
diagnostic smoke (not official). Each shard
writes `answers.shard-<i>-of-<n>.jsonl` and `result.shard-<i>-of-<n>.json`, plus
`shard-<gpu>.log`; the scheduler merges them into the canonical `answers.jsonl`
and scores/exports exactly once. The final `results.json`/`results.csv` are
written after all benchmark finalizers finish.

For an individual benchmark (or when debugging one scorer), the single-benchmark
commands remain available:

```bash
python -m vred.evaluate --action list
python -m vred.evaluate --benchmark gqa --model "$MODEL" --vision-tower "$VISION_TOWER" --output "$OUT"
python -m vred.evaluate --benchmark scienceqa --model "$MODEL" --vision-tower "$VISION_TOWER" --output "$OUT"
python -m vred.evaluate --benchmark textvqa --model "$MODEL" --vision-tower "$VISION_TOWER" --output "$OUT"
python -m vred.evaluate --benchmark pope --model "$MODEL" --vision-tower "$VISION_TOWER" --output "$OUT"
python -m vred.evaluate --benchmark mme --model "$MODEL" --vision-tower "$VISION_TOWER" --output "$OUT"
python -m vred.evaluate --benchmark mmmu --model "$MODEL" --vision-tower "$VISION_TOWER" --output "$OUT"
python -m vred.evaluate --benchmark vqav2 --model "$MODEL" --vision-tower "$VISION_TOWER" --output "$OUT"
python -m vred.evaluate --benchmark vizwiz --model "$MODEL" --vision-tower "$VISION_TOWER" --output "$OUT"
python -m vred.evaluate --benchmark mmbench --model "$MODEL" --vision-tower "$VISION_TOWER" --output "$OUT"
python -m vred.evaluate --benchmark mmbench_cn --model "$MODEL" --vision-tower "$VISION_TOWER" --output "$OUT"
python -m vred.evaluate --action collect --output "$OUT"
```

Existing benchmark outputs are overwritten on rerun: `answers.jsonl`,
`answers.shard-*.jsonl`, `result.json`/`result.shard-*.json`, `scores.json`,
`scorer.log` (truncated per scoring), `submission.json`, `answers.xlsx`, and the
MME converter input `eval_dataset/eval/MME/answers/vred_<run>.jsonl`/
`eval_tool/answers/<experiment>/` are refreshed in place. A rerun does not need
a unique output directory or a prior `rm`.

TextVQA reuses `question_id` (the image id) for multiple questions
(5,000 rows / 3,166 unique ids locally). Sharded merge therefore keys on the
answer's original benchmark row `benchmark_index`, not `question_id`; duplicate
question ids are valid and the TextVQA scorer matches by
`(image_id, question)`. Shard answers without `benchmark_index` are old-format
and must be regenerated.

MMBench/MMBench-CN TSV images are decoded directly from the embedded base64
column, so they need neither an extracted image folder nor `--image-map` in
online mode. `--image-root` overrides the per-benchmark image root; `--image-map`
is only needed when using a precomputed cache for embedded TSV images. The
precomputed `--cache-index` path remains supported for reproducibility and for
offline/accelerated evaluation; `--vision-dtype` selects the online encoder's
`fp16` (default) or `bf16` weights.

The adapter emits legacy `answers.jsonl` and invokes LLaVA-KD/local scorers or
converters. TextVQA/ScienceQA/POPE use repository scorers. VQAv2 and GQA are
scored fully locally: VQAv2 uses the official soft accuracy on the local
`llava_vqav2_mscoco_val2014.jsonl` + `v2_mscoco_val2014_annotations.json`
(val2014, not the official test-dev/leaderboard split); GQA uses balanced exact
match from `questions1.2/testdev_balanced_questions.json`. Neither needs the
old GQA `eval/eval.py` scene-graph assets or an external submission. MME/MMMU
still invoke the supplied tools, but their stdout is parsed into structured
`scores.json` (per-category and per-task for MME, per-category and overall for
MMMU). MME's fixed-interface converter keys ground truth by
`(category, file, question)` and writes `MME/answers/vred_<run>.jsonl` and tool
output. The adapter therefore writes the raw MME question as `prompt` (no
`<image>` prefix); an image tag makes `GT[category, file, prompt]` raise
`KeyError`. Use a unique run name.
VizWiz is scored locally on the val split: `llava_val.jsonl` +
`Annotations/val.json` (`images: val`), using the previous local pipeline's
`TextVQAAccuracyEvaluator` soft accuracy (EvalAI answer normalization, each of
the ten annotator answers scored against the other nine with
`min(matches/3, 1)`). The official test split has no public answers and remains
submission-only; a spec with `scoring: submission` still exports
`submission.json`. MMBench EN/CN are also scored locally on the dev TSV:
`MMBench_DEV_EN_legacy.tsv` and `MMBench_DEV_CN_legacy.tsv` carry the answer
column, and the prediction's first standalone option letter is compared exactly
with that answer over 4,329 questions. This matches the previous local metric
(`68.4454` EN and `66.2740` CN on the earlier `hcid_asa_student_cdft1_final`
predictions). A `scoring: submission` MMBench spec still uses the existing Excel
converter (`pandas`, `openpyxl`; install `openpyxl` in the active interpreter)
for an external split; no score is invented.

`--action list` reports both `available` (question file) and `scorer_ready`
plus the missing local files/tools, including VQAv2/GQA/VizWiz annotation files,
MME/MMMU tools, ScienceQA/TextVQA/POPE annotations, and `openpyxl` only for a
`scoring: submission` MMBench spec; local MMBench EN/CN need no annotation file
and no `openpyxl`.
`--action run`/`score` fail before generating if a local prerequisite is
missing. Online evaluation checks the benchmark image roots reported by `list`;
a precomputed evaluation feature cache is optional and is not checked by `list`.

The collector writes `results.json`/`results.csv` with distinct `locally_scored`,
`submission_required`, `subset_only`, `scoring_failed`, and `not_run` statuses.
Original scorer stdout/stderr is retained in `scorer.log`; local scorers write
`scores.json`. `--limit N` produces inference diagnostics only and cannot be
promoted to an official score. Retry failed scoring with `--action score` after
restoring prerequisites. Keep benchmark test results out of tuning.

## Troubleshooting And Limits

- Missing path/cache: restore assets or edit configs; the error identifies the
  record/key/file. Run preflight before GPU allocation.
- Tokenizer mismatch: restore matching checkpoint tokenizers; never suppress
  semantic validation or remap logits using vocabulary width alone.
- CUDA/DeepSpeed unavailable: use explicit synthetic CPU checks. Six-GPU runtime,
  max-length VRAM and throughput remain unverified in this session.
- OOM: sparse shifted CE (default) removes the dense `[batch, sequence,
  151936]` logits and CE temporaries; the real 0.5B check gives identical selected logits and
  a 1.2e-7 loss difference from the dense path. For SFT, `--gradient-checkpointing` is supported;
  for DPT/DFT use `--selective-checkpointing` (non-selected blocks only). If still short, lower `--micro-batch-size` and raise
  `--accumulation` to keep the same global batch (for example 4/4 instead of
  8/2 at six ranks), set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
  and check for other GPU processes. Lower `--query-chunk-size` for probe
  retention, `--ce-chunk-size` for the response-position CE tensors and
  `--probe-vocab-chunk-size` for the FP32 vocabulary-head chunk. Optional `write_position_limit` sampling must be recorded/shared
  across models. Never top-k approximate vocabulary probabilities or drop image
  tokens.
- Do not use SDPA for training/evaluation to save memory: measured real-0.5B
  BF16 selected logits differ from eager by up to 1.37 and caption CE by 0.0163.
  Eager attention is required for exact cross-stage consistency and effect
  capture. The global backbone-checkpointing flag is rejected during effect
  capture; `--selective-checkpointing` leaves the selected blocks ordinary and
  checkpoints the rest. Probe-loss checkpointing is supported.
- Supplied-depth pairs are (7,11)/(15,23)/(23,35) in DPT and
  (3,5)/(7,11)/(11,17)/(15,23)/(19,29)/(23,35) in DFT. Negative branches are local
  auxiliaries; ordinary decoder outputs are reused.
- BF16 reconstruction rounding and the FP32 reference are documented in
  `IMPLEMENTATION_STATUS.md`. Keep query size fixed for Effect-KL comparisons
  near zero/sign boundaries.

`Batch` exposes expanded IDs, labels, padding/causal masks, CPU raw features,
half-open visual spans and explicit shifted response/write coordinates.
Prediction position p targets label p+1. Text-only records contribute CE/final
KL and zero write loss. `load_model` and `VREDModel.forward` remain the model API.
No magnitude, affinity, handoff, CMAD or CMAED objective is added.

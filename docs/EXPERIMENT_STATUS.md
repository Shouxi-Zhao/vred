# VRED Distillation: Stage-Wise Experiment Report

## 1. Scope and Common Setup

This report records the completed DFT experiments, their evaluation results, the
effect-KD code changes, and the currently running `dft-ewkd-hidden` experiment.
It is an internal engineering/research status document, not a paper draft.

Common setup:

- Student: Qwen2.5-0.5B.
- Teacher: Qwen2.5-3B with merged LoRA; the teacher vision tower is not used.
- Vision features: cached SigLIP2 SO400M 384 features, `729 x 1152`, FP16.
- Training data: LLaVA-1.5 mix665k cache; DPT uses the BLIP/LAION cache.
- Hardware: 6 GPUs for the reported DFT runs; 1 epoch.
- DPT: 3 relative-depth pairs, `lambda_write=25`, `lr=1e-3`,
  `micro_batch_size=16`, `accumulation=2`, 2907 steps.
- SFT: no write/KL objective, `lr=2e-5`, `micro_batch_size=8`,
  `accumulation=2`, 6927 steps.
- DFT: initialized from `checkpoints/vred/sft`; `lr=2e-5`, one epoch,
  6927 optimizer steps, 73,336,518 response tokens in the full K6/K9 runs.
- Selective checkpointing and probe checkpointing are enabled for DFT.
- The DFT write loss is SmoothL1 with `write_vocab_reduction=sum`,
  `tau_write=1.0`; final logit KD uses `tau_logits=1.0`.
- Evaluation is the local ten-benchmark suite. `avg10` averages all ten
  benchmarks. `avg7` averages GQA, VizWiz, ScienceQA, TextVQA, MME, MMBench,
  and MMBench-CN; it excludes VQAv2, POPE, and MMMU.
- All results below are single-seed results. No confidence intervals or
  multi-seed tests have been run.

## 2. Completed Experiments

### 2.1 DFT configuration matrix

All four runs use the same SFT input and the same DFT data. The K6 full
checkpoint is no longer retained; its evaluation is preserved under
`results/vred-dft-pairs6`. The retained `dft-pilot-lambda25` directory is a
150-step K6 pilot with `micro=1/accum=16` and is not the source of the K6
numbers below.

| Run | Checkpoint / result root | Pairs | lambda_write | lambda_logits | L_ewkd | Write scope | micro / accum | Steps | Evaluation |
|---|---|---|---:|---:|---|---|---|---:|---|
| K6 | `results/vred-dft-pairs6` (full checkpoint removed) | 6 | 25 | 1 | none | response | 2 / 8 | 6927 | complete |
| K9 | `checkpoints/vred/dft-pairs9`, `results/vred-dft-pairs9` | 9 | 25 | 1 | none | response | 2 / 8 | 6927 | complete |
| K9, write disabled | `checkpoints/vred/dft-pairs9-write0`, `results/dft-pairs9-write0` | 9 | 0 | 1 | none | response | 2 / 8 | 6927 | complete |
| K9, lambda_write=100 | `checkpoints/vred/dft-write-l1-100`, `results/dft-write-l1-100` | 9 | 100 | 1 | none | response | 2 / 8 | 6927 | complete |
| K9, hidden effect-KD | `checkpoints/vred/dft-ewkd-hidden` | 9 | 100 | 0 | hidden, lambda=0.5, floor=0.15 | response, uniform, no limit | 2 / 8 | 2673/6927 at report time | running; no evaluation yet |

Notes:

- The K6 full run used the six relative-depth pairs
  `[[3,5],[7,11],[11,17],[15,23],[19,29],[23,35]]`.
- K9 uses the nine pairs
  `[[2,3],[5,7],[7,11],[10,15],[13,19],[15,23],[18,27],[21,31],[23,35]]`.
- K9 / write0 / lambda_write=100 use `query_chunk_size=256`,
  `ce_chunk_size=256`, `probe_vocab_chunk_size=16384`, sparse logits,
  `checkpoint_probes=true`, and `selective_checkpointing=true`.
- The K6 full run reached a peak reserved memory of about 19.9 GiB and used
  about 110.4 GPU-hours. K9 and the ablations use the same 6927-step,
  one-epoch schedule; their exact wall-clock totals depend on the run.
- The write path in K9 is the original response-scope, uniform, unlimited
  response write. The lambda_write=100 run explicitly records
  `write_position_scope=response`.

### 2.2 Evaluation results

All values are percentages on the local evaluation suite. Higher is better.

| Benchmark | K6 | K9 | K9 write0 | K9 lambda_write=100 |
|---|---:|---:|---:|---:|
| VQAv2 | 77.3156 | 77.3227 | 77.3311 | 77.3446 |
| GQA | 60.0652 | 59.9857 | 60.1924 | 60.1526 |
| VizWiz | 41.5304 | 43.4777 | 42.3570 | 44.6654 |
| ScienceQA | 58.1467 | 57.9344 | 57.7222 | 58.1702 |
| TextVQA | 52.5700 | 52.6600 | 52.9700 | 51.7300 |
| MME | 75.0511 | 76.0232 | 76.0498 | 76.7670 |
| MMBench | 71.9566 | 71.9104 | 71.6563 | 72.1876 |
| MMBench-CN | 67.6369 | 67.4059 | 67.8679 | 67.3828 |
| POPE | 87.3061 | 87.2953 | 87.3956 | 87.5297 |
| MMMU | 30.7000 | 32.1000 | 32.3000 | 30.7000 |
| **avg10** | **62.2279** | **62.6115** | **62.5842** | **62.6630** |
| **avg7** | **60.9938** | **61.3425** | **61.2594** | **61.5794** |

### 2.3 Deltas

| Comparison | avg10 | avg7 | Main gains | Main losses |
|---|---:|---:|---|---|
| K9 - K6 | +0.3836 | +0.3487 | VizWiz +1.9473, MMMU +1.4000, MME +0.9721 | MMBench-CN -0.2310, ScienceQA -0.2123 |
| write0 - K9 | -0.0273 | -0.0831 | GQA +0.2067, MMBench-CN +0.4620, TextVQA +0.3100 | VizWiz -1.1207, MMBench -0.2541, ScienceQA -0.2122 |
| lambda_write=100 - K9 | +0.0515 | +0.2369 | VizWiz +1.1877, MME +0.7438, MMBench +0.2772 | MMMU -1.4000, TextVQA -0.9300 |
| lambda_write=100 - write0 | +0.0788 | +0.3200 | VizWiz +2.3084, MME +0.7172, MMBench +0.5313 | MMMU -1.6000, TextVQA -1.2400, MMBench-CN -0.4851 |

### 2.4 Interpretation of Completed Runs

1. K9 is better than K6 on the average metrics, but the gain is concentrated in
   VizWiz, MMMU, and MME. MMBench-CN and ScienceQA move slightly down. The
   K9-vs-K6 gap is small relative to cross-benchmark variance and is only one
   seed.

2. Disabling the DFT write loss (K9 write0) does not produce a stable average
   loss: K9 is higher by only +0.0273 avg10 and +0.0831 avg7. Write0 wins GQA,
   MMBench-CN, TextVQA, and MMMU; K9 wins VizWiz, ScienceQA, and MMBench. The
   current evidence does not support the claim that the DFT write objective is
   reliably useful at this scale. This ablation only removes DFT write; the DPT
   stage was still trained with `lambda_write=25`, so it does not test the full
   write recipe.

3. Increasing lambda_write from 25 to 100 gives a slightly higher average
   (+0.0515 avg10, +0.2369 avg7) but has clear regressions on MMMU and TextVQA.
   This is not strong enough to treat 100 as a validated improvement. It is also
   confounded with the write-loss gradient scale, not just the loss value.

4. All deltas are small. The benchmark suite has no repeated seeds, no
   confidence intervals, and no held-out selection protocol. These results
   should be treated as directional, not as established rankings.

## 3. Effect-KD Code Changes

The effect-KD work adds two opt-in modes. With no new flag, the original
CE + write + ordinary final-KD path is unchanged.

### 3.1 Original logits-effect mode

The first version implements effect-weighted final-logit KD:

- Teacher plus logits: `z_t+` from the normal image-conditioned teacher forward.
- Teacher minus logits: `z_t-` from a teacher forward with zeroed pre-connector
  visual features.
- `p_t+ = softmax(z_t+ / tau)`, `p_t- = softmax(z_t- / tau)`.
- `w_t = ||p_t+ - p_t-||_1`, not normalized.
- `L_ewkd = mean_t(w_t * KL(p_t+ || p_s+))`.

Code: `effect_weighted_kd_sparse` in `vred/losses.py`. Flags:
`--effect-weighted-kd`, `--lambda-effect-kd`.

Cost: this mode requires a second full teacher forward per microbatch. That is
the dominant reason the earlier L_ewkd run was slower than K9.

### 3.2 Hidden-effect mode with floor

The second version avoids the second teacher forward by reusing the existing
teacher plus `EffectCapture`:

- For each response token `t`,
  `s_t = mean_pair ||u_(l,t)+ - u_(l,t)-||_2`.
- `w_t = floor + (1 - floor) * s_t / mean(s)`.
- Text-only positions receive the floor.
- `L_ewkd = mean_t(w_t * KL(p_t+ || p_s+))`.

For nonzero effects, the weights average one; this makes the effective loss
scale larger than the unnormalized logits-mode weights. The starting coefficient
is therefore `lambda_effect_kd=0.5`, not the old logits-mode value of 1.0.

Code:

- `floored_effect_weights` and `weighted_kd_sparse` in `vred/losses.py`.
- `response_effect_scores` in `vred/effects.py`, which aligns teacher hidden
  effect scores to every `batch.response_positions`.
- `vred/trainer.py`: hidden mode runs no teacher-minus forward; it uses the
  existing plus effect capture. When `lambda_logits=0`, the ordinary final-KD
  term is skipped entirely.
- `vred/options.py`: new flags
  `--effect-kd-weight-source logits|hidden` and `--effect-kd-floor`.
  Hidden mode is rejected unless the stage is DFT, logits are sparse, write
  scope is `response`, there is no write-position limit, and sampling is
  `uniform`. This guarantees that the hidden weights align with every response
  token.

Default behavior is opt-in: the effect-KD flags are absent by default, and the
default data/loss path remains the original one. The current suite has 58 CPU
tests, including the hidden formula, floor behavior, one-teacher-forward
behavior, and CLI validation. Relevant docs were updated in `vred/README.md`,
`训练启动命令.md`, `vred/IMPLEMENTATION_STATUS.md`, `project_background.md`, and
`docs/MLLM_KD_RESEARCH_NOTES.md`.

## 4. Running Experiment: dft-ewkd-hidden

The reported run is not the clean A configuration recommended in the final
launch command. Its actual `run_config.json` is:

- K9 DFT, initialized from `checkpoints/vred/sft`.
- `lambda_write=100` (not 25).
- `lambda_logits=0`; ordinary final KD is skipped.
- `lambda_effect_kd=0.5`, `effect_kd_weight_source=hidden`,
  `effect_kd_floor=0.15`.
- `write_position_scope=response`, uniform sampling, no write limit.
- `micro_batch_size=2`, `accumulation=8`, global batch 96.
- `query_chunk_size=256`, `ce_chunk_size=256`,
  `probe_vocab_chunk_size=16384`.
- `checkpoint_probes=true`, `selective_checkpointing=true`.

Status at the time of this report:

- About 2673 of 6927 optimizer steps, approximately 38.6%.
- No evaluation has been run; only training logs are available.
- No NaN or Inf; peak reserved memory about 23.4 GiB on 24 GiB GPUs.
- Median microstep time about 1.40 s; the run is materially faster than the
  earlier logits-effect L_ewkd runs because it does not run the second teacher
  forward.
- In the 100-step window around 38%, CE is about 0.952, raw effect KD about
  0.232, weighted effect KD about 0.116, and weighted write loss about 0.128.
  The effect and write terms are therefore comparable and each is roughly
  12–13% of CE.
- At the same optimizer step, CE is about 0.988 for this run, versus about
  0.984 for K9 and about 0.984 for write0. There is no same-step CE advantage
  for the hidden-effect run.
- Training is stable, but the current run is a combination of hidden effect-KD
  and an aggressive lambda_write=100. It cannot by itself isolate the effect of
  the hidden-effect KD.

## 5. Current Assessment

- The code changes are implemented and tested; the default path is unchanged.
- The hidden mode removes the expensive teacher-minus forward and is a
  practical speed improvement.
- The completed K9/K6/write0/lambda_write=100 experiments do not establish a
  stable average benefit from DFT write, nor a clear benefit from increasing
  lambda_write to 100.
- The running hidden-effect experiment is stable and fast enough, but it has no
  evaluation result yet and its same-step CE is not better than K9/write0.
- The current evidence supports an ablation study, not a paper claim that
  effect-weighted KD is already superior.

## 6. Limitations and Risks

1. Single seed per run; small average differences may be noise.
2. No confidence intervals or paired tests across benchmarks.
3. The hidden-effect weight is a hidden-state magnitude proxy, not the actual
   token-probability effect used by the logits mode.
4. The running experiment uses lambda_write=100 and lambda_logits=0, so it
   combines three changes: hidden weighting, no ordinary KD, and aggressive
   write. It is not a clean A-only run.
5. The K6 full checkpoint is no longer retained; its metrics are preserved but
   the model cannot be re-evaluated without retraining.
6. The local scorers and benchmark versions are fixed in this repository; the
   numbers are not directly comparable to external leaderboard numbers obtained
   with different prompts, splits, or parsers.

## 7. Recommended Next Steps

1. Let the current `dft-ewkd-hidden` run finish or reach the next exported
   checkpoint, then evaluate it with the same local protocol. Do not tune on the
   test benchmarks; use a held-out split for model selection where possible.
2. Run a clean A experiment with `lambda_write=25`, `lambda_logits=0`,
   `lambda_effect_kd=0.5`, `effect_kd_floor=0.15`, response scope, uniform
   sampling, and no write limit. Compare it directly with K9 and write0.
3. Run a small coefficient sweep only after the clean A run:
   `lambda_effect_kd` in {0.3, 0.5, 0.7}; floor in {0.1, 0.15, 0.25}.
   Change one axis at a time and use at least two seeds for the final choice.
4. Compare hidden-effect KD against the original logits-effect KD at matched
   compute. If hidden mode has no gain, the hidden magnitude may be too weak a
   proxy; do not assume that increasing lambda is the fix.
5. If the goal is a paper, frame the contribution as one counterfactual
   visual-effect principle with multiple instantiations (write regularizer and
   effect-weighted KD), and report the negative or mixed ablations honestly.
   The current results do not justify claiming that both components are
   independently necessary.

## 8. Prompt for GPT

Copy the following block together with this report (or paste the report as an
attachment). It is written to make the model challenge the evidence rather than
simply repeat the conclusions.

```text
You are a senior ML research lead reviewing a multimodal distillation project.
I will give you an internal experiment report. Analyze it critically and help me
decide the next research steps.

Project summary:
- Student: Qwen2.5-0.5B. Teacher: Qwen2.5-3B with merged LoRA.
- Vision: cached SigLIP2 SO400M 384 features, 729 visual tokens, 1152 width.
- Three stages: DPT (connector only, CE + write), SFT (connector + LM, CE only),
  DFT (connector + LM, CE + write + final logit KD).
- Write loss: teacher local visual-message effect g = u_plus - u_minus at
  selected blocks, probed through the final RMSNorm and vocabulary head;
  SmoothL1 by default.
- Original final KD: response-token KL between teacher and student.
- Effect-KD: L_ewkd = mean_t(w_t * KL(p_t+ || p_s+)), where w_t is derived from
  teacher plus/minus behavior. Two variants exist: an expensive logits mode
  requiring a second teacher forward, and a cheaper hidden mode using teacher
  hidden-effect magnitude plus a floor.

Completed local results (percentages; avg7 excludes VQAv2, POPE, MMMU):

Run              VQAv2   GQA    VizWiz  SQA    TextVQA MME     MMBench MMBCN   POPE    MMMU    avg10    avg7
K6               77.3156 60.0652 41.5304 58.1467 52.5700 75.0511 71.9566 67.6369 87.3061 30.7000 62.2279 60.9938
K9               77.3227 59.9857 43.4777 57.9344 52.6600 76.0232 71.9104 67.4059 87.2953 32.1000 62.6115 61.3425
K9 write0        77.3311 60.1924 42.3570 57.7222 52.9700 76.0498 71.6563 67.8679 87.3956 32.3000 62.5842 61.2594
K9 lambda_w=100  77.3446 60.1526 44.6654 58.1702 51.7300 76.7670 72.1876 67.3828 87.5297 30.7000 62.6630 61.5794

Key deltas:
- K9 - K6: avg10 +0.3836, avg7 +0.3487; gains mainly VizWiz/MMMU/MME.
- write0 - K9: avg10 -0.0273, avg7 -0.0831; no stable write benefit.
- lambda_w=100 - K9: avg10 +0.0515, avg7 +0.2369, but MMMU -1.4 and
  TextVQA -0.93.

Current run: dft-ewkd-hidden, K9, lambda_write=100, lambda_logits=0,
lambda_effect_kd=0.5, floor=0.15, hidden mode, response scope, uniform sampling.
At about 38.6% of training (2673/6927 steps), stable, no NaN, peak memory
23.4 GiB, median microstep 1.40s. Same-step CE is about 0.988 versus about
0.984 for K9 and write0. No evaluation yet. The run is not a clean A-only run
because lambda_write is 100, not the recommended 25.

Please answer:
1. Which conclusions are supported by the current evidence, and which are
   overclaims? Be explicit about single-seed noise, benchmark variance, and
   confounds.
2. Is the DFT write loss useful? What experiment would distinguish "write is
   redundant" from "write is useful but lambda/positions are wrong"?
3. Is hidden-effect KD a sound approximation of logits-effect KD? What is the
   most likely failure mode of using hidden-effect magnitude as w_t?
4. Design the next three experiments in priority order. Give exact configs,
   controls, number of seeds, evaluation protocol, and decision criteria.
5. What should be changed in the code or training loop before the next run, and
   what should explicitly NOT be changed?
6. Propose a paper narrative and an ablation matrix. Include honest negative
   results and avoid claiming two independent innovations if they share one
   mechanism.
7. Point out missing baselines, evaluation risks, and any result that looks
   suspicious or internally inconsistent.

Constraints:
- 6 x 24 GiB GPUs for the current setup; larger GPUs may be available later.
- One epoch per stage; full DFT is about 6927 optimizer steps.
- Avoid test-set tuning; prefer held-out validation for decisions.
- Do not weaken the method merely to make the numbers look better.
- Return a concise executive summary, a prioritized experiment table, and a
  short list of concrete next actions.
```

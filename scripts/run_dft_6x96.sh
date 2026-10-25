#!/usr/bin/env bash
# DFT write experiments on 6 x 96G GPUs.
# Strictly preserve the original global batch: 6 x 12 x 1 = 72 (original 6 x 6 x 2).
# If OOM, add --checkpoint-probes --selective-checkpointing to each command.
set -euo pipefail
cd "$(dirname "$0")/.."

python -m vred.launch dft --gpus 6 --seed 1234 --epochs 1 --lr 2e-5 \
  --micro-batch-size 12 --accumulation 1 --pairs 9 \
  --lambda-write 5 --lambda-logits 1 \
  --write-position-scope response \
  --ce-chunk-size 256 --query-chunk-size 256 --probe-vocab-chunk-size 16384 \
  --write-lambda-schedule warmup_cosine --online-vision \
  --write-loss-type effect_js --write-vocab-reduction sum \
  --input-model checkpoints/vred/sft --num-workers 4 \
  --output checkpoints/vred/dft-write-js-5

python -m vred.launch dft --gpus 6 --seed 1234 --epochs 1 --lr 2e-5 \
  --micro-batch-size 12 --accumulation 1 --pairs 9 \
  --lambda-write 2 --lambda-logits 1 \
  --write-position-scope response \
  --ce-chunk-size 256 --query-chunk-size 256 --probe-vocab-chunk-size 16384 \
  --write-lambda-schedule warmup_cosine --online-vision \
  --write-loss-type effect_kl --write-vocab-reduction sum \
  --input-model checkpoints/vred/sft --num-workers 4 \
  --output checkpoints/vred/dft-write-kl-2

python -m vred.launch dft --gpus 6 --seed 1234 --epochs 1 --lr 2e-5 \
  --micro-batch-size 12 --accumulation 1 --pairs 9 \
  --lambda-write 1 --lambda-logits 1 \
  --write-position-scope response \
  --ce-chunk-size 256 --query-chunk-size 256 --probe-vocab-chunk-size 16384 \
  --write-lambda-schedule warmup_cosine --online-vision \
  --write-loss-type reverse_kl --write-vocab-reduction sum \
  --input-model checkpoints/vred/sft --num-workers 4 \
  --output checkpoints/vred/dft-write-reverse-1

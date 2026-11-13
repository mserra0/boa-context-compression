#!/usr/bin/env bash
# LLM-as-judge evaluation of one checkpoint (needs a running judge server: scripts/eval/vllm_judge.sh).
# Usage: bash scripts/eval/run_eval.sh <experiment-dir>
set -e
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} uv run python src/evaluate.py \
  --eval-config configs/eval/config_eval.yaml \
  --experiment-dir "${1:?usage: bash scripts/eval/run_eval.sh <experiment-dir>}" \
  --device cuda

#!/usr/bin/env bash
# Evaluate one or more Boa checkpoints on MRQA.
#
# Scoring is official MRQA F1/EM (span overlap) — offline, no server to start.
#
# Usage:
#   scripts/eval/eval_mrqa.sh <run_dir> [<run_dir> ...]
#   The experiment folder NAMES are resolved against CKPT_ROOT; alternatively list them
#   in EXPERIMENTS below.
#
# Each experiment dir must contain best_model.pt and config.yaml.
#
# Override defaults via env:
#   CKPT_ROOT=/path/to/checkpoints             base dir holding the experiments
#   CONFIG=configs/eval/config_eval_MRQA.yaml   eval config
#   GPU=0                                        CUDA device index
#   DEVICE=cuda                                  cuda | cpu
#   WANDB=0                                      disable W&B logging (on by default)
set -euo pipefail

CKPT_ROOT="${CKPT_ROOT:-/path/to/checkpoints}"

# Experiment folder names (resolved against CKPT_ROOT): pass them as arguments or list them here.
EXPERIMENTS=("$@")
if [[ ${#EXPERIMENTS[@]} -eq 0 ]]; then
  EXPERIMENTS=(
    # <run_dir_1>
    # <run_dir_2>
  )
fi
[[ ${#EXPERIMENTS[@]} -gt 0 ]] || { echo "usage: $0 <run_dir> [<run_dir> ...]  (or list them in EXPERIMENTS)" >&2; exit 1; }

CONFIG="${CONFIG:-configs/eval/config_eval_MRQA.yaml}"
GPU="${GPU:-0}"
DEVICE="${DEVICE:-cuda}"
WANDB_FLAG=""; [[ "${WANDB:-1}" == "0" ]] && WANDB_FLAG="--no-wandb"

[[ -f "$CONFIG" ]] || { echo "ERROR: eval config not found: $CONFIG" >&2; exit 1; }

for name in "${EXPERIMENTS[@]}"; do
  dir="$CKPT_ROOT/$name"
  [[ -f "$dir/best_model.pt" ]] || { echo "ERROR: best_model.pt not found in $dir" >&2; exit 1; }
  [[ -f "$dir/config.yaml"   ]] || { echo "ERROR: config.yaml not found in $dir"   >&2; exit 1; }

  echo "=== Evaluating: $name ==="
  CUDA_VISIBLE_DEVICES="$GPU" uv run python -m src.evaluation \
    --config "$CONFIG" \
    --experiment-dir "$dir" \
    --device "$DEVICE" \
    $WANDB_FLAG
done

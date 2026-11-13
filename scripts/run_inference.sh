#!/bin/bash
# Interactive inference: compare the frozen LLM (teacher) with the compressed student.
#
# Usage:
#   bash scripts/run_inference.sh <config.yaml> <checkpoint.pt>
#   CUDA_VISIBLE_DEVICES=1 bash scripts/run_inference.sh <config.yaml> <checkpoint.pt>
#
# <config.yaml> is the training config of the checkpoint (a copy is saved next to every
# checkpoint as config.yaml); <checkpoint.pt> is e.g. <checkpoint_dir>/<run>/best_model.pt.
set -e

CONFIG_PATH=${1:?usage: bash scripts/run_inference.sh <config.yaml> <checkpoint.pt>}
CHECKPOINT_PATH=${2:?usage: bash scripts/run_inference.sh <config.yaml> <checkpoint.pt>}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

uv run python src/inference.py --config "$CONFIG_PATH" \
    --checkpoint "$CHECKPOINT_PATH" \
    --device cuda

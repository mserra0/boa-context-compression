#!/bin/bash
# Launch DDP training with torchrun on the GPUs listed in CUDA_VISIBLE_DEVICES (default: GPU 0).
#
# Usage:
#   bash scripts/run_training.sh <config.yaml>
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_training.sh <config.yaml>
#
# Optional environment:
#   ENV_FILE    a file to `source` that defines CACHE_ROOT and any other variables you need
#   CACHE_ROOT  if set, TRITON_CACHE_DIR defaults to $CACHE_ROOT/triton
set -e

CONFIG_PATH=${1:?usage: bash scripts/run_training.sh <config.yaml>}

if [ -n "${ENV_FILE:-}" ] && [ -f "$ENV_FILE" ]; then
    source "$ENV_FILE"
fi

if [ -n "${CACHE_ROOT:-}" ]; then
    export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${CACHE_ROOT}/triton}"
fi
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
NUM_TRAINERS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' ' ' | wc -w)

uv run torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NUM_TRAINERS \
    main.py --config $CONFIG_PATH

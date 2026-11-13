#!/bin/bash

set -e

# Launch the Qwen3 FullFT baseline (DDP via torchrun).
# Usage:
#   bash scripts/run_baseline.sh <config_path> [mode]
#     <config_path>  required — e.g. configs/baselines/qwen3_1_7b_fullft.yaml
#     [mode]         optional — "train" (default) | "dump"
#   Override GPUs:   CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_baseline.sh <config> [mode]

CONFIG_PATH=$1
MODE=${2:-train}

if [ -z "$CONFIG_PATH" ]; then
    echo "ERROR: config path required."
    echo "Usage: bash scripts/run_baseline.sh <config_path> [train|dump]"
    exit 1
fi

if [ ! -f "$CONFIG_PATH" ]; then
    echo "ERROR: config not found: $CONFIG_PATH"
    exit 1
fi

case "$MODE" in
    train) MODULE="src.baselines.train" ;;
    dump)  MODULE="src.baselines.dump_logits" ;;
    *) echo "ERROR: mode must be 'train' or 'dump' (got '$MODE')"; exit 1 ;;
esac

# Optional: ENV_FILE points to a file that defines CACHE_ROOT and other variables;
# if CACHE_ROOT is set, TRITON_CACHE_DIR defaults to $CACHE_ROOT/triton.
if [ -n "${ENV_FILE:-}" ] && [ -f "$ENV_FILE" ]; then
    source "$ENV_FILE"
fi
if [ -n "${CACHE_ROOT:-}" ]; then
    export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${CACHE_ROOT}/triton}"
fi
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
NUM_PROC=$(echo $CUDA_VISIBLE_DEVICES | tr ',' ' ' | wc -w)

echo "Launching baseline [$MODE] on GPUs '$CUDA_VISIBLE_DEVICES' (nproc=$NUM_PROC) with $CONFIG_PATH"

uv run torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NUM_PROC \
    -m $MODULE \
    --config $CONFIG_PATH

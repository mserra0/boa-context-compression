#!/bin/bash
# Usage:
#   scripts/run_training_icae.sh [--gpu N[,N,...]] [config_path]
#
# --gpu   GPU index(es) to use (e.g. "3" or "0,1"). If omitted, CUDA_VISIBLE_DEVICES
#         is used, or GPU 0 when it is unset.
# config  Path to the training YAML. Defaults to configs/train/config_ICAE_ae.yaml

set -e

GPU=""
CONFIG_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)      GPU="$2";          shift 2 ;;
        --gpu=*)    GPU="${1#*=}";     shift   ;;
        *)          CONFIG_ARGS+=("$1"); shift  ;;
    esac
done

CONFIG_PATH="${CONFIG_ARGS[0]:-configs/train/config_ICAE_ae.yaml}"

GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"

export CUDA_VISIBLE_DEVICES="$GPU"
echo "CONFIG_PATH=$CONFIG_PATH"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

NUM_TRAINERS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' ' ' | wc -w)

uv run torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NUM_TRAINERS \
    main.py --config "$CONFIG_PATH"

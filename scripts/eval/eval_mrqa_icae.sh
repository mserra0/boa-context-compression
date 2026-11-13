#!/bin/bash
# ICAE evaluation script — deterministic MRQA F1/EM scoring (src/evaluation package),
# using the official MRQA/SQuAD span metrics — no vLLM judge server needed.
#
# Usage:
#   scripts/eval/eval_mrqa_icae.sh [--gpu N] <experiment-dir> [eval-config]
#
# --gpu             GPU index to use (e.g. "3"). If omitted, CUDA_VISIBLE_DEVICES
#                   is used, or GPU 0 when it is unset.
# <experiment-dir>  must contain best_model.pt and config.yaml (the checkpoint dir).
# <eval-config>     defaults to configs/eval/config_eval_MRQA.yaml (scorer: f1_em)
#
# Example:
#   scripts/eval/eval_mrqa_icae.sh /path/to/checkpoints/<icae_run_dir>

set -e

GPU=""
ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)      GPU="$2";      shift 2 ;;
        --gpu=*)    GPU="${1#*=}"; shift   ;;
        *)          ARGS+=("$1");  shift   ;;
    esac
done

EXPERIMENT_DIR="${ARGS[0]:?Usage: $0 [--gpu N] <experiment-dir> [eval-config]}"
EVAL_CONFIG="${ARGS[1]:-configs/eval/config_eval_MRQA.yaml}"

GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"

export CUDA_VISIBLE_DEVICES="$GPU"

echo "EXPERIMENT_DIR=$EXPERIMENT_DIR"
echo "EVAL_CONFIG=$EVAL_CONFIG"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

uv run python -m src.evaluation \
    --config "$EVAL_CONFIG" \
    --experiment-dir "$EXPERIMENT_DIR" \
    --device cuda

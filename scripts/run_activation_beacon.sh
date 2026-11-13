#!/bin/bash
set -euo pipefail

CONFIG_PATH="${1:-configs/eval/config_activation_beacon.yaml}"
DEVICE="${2:-cuda}"

export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/${USER}/activation-beacon-triton}"
mkdir -p "$TRITON_CACHE_DIR"

uv run --no-sync \
  python src/evaluate_activation_beacon.py \
  --eval-config "$CONFIG_PATH" \
  --device "$DEVICE"

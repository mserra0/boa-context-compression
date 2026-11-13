#!/bin/bash
set -euo pipefail

CONFIG_PATH="${1:-configs/train/config_activation_beacon_qwen3_17b_fullft_fineweb_edu_10b_official.yaml}"
shift || true

export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/${USER}/activation-beacon-triton}"
DEFAULT_TRANSFORMERS_VERSION="4.40.2"
if grep -Eq 'beacon_backend:.*qwen3_' "$CONFIG_PATH"; then
  DEFAULT_TRANSFORMERS_VERSION="4.57.1"
fi
export ACTIVATION_BEACON_TRANSFORMERS_VERSION="${ACTIVATION_BEACON_TRANSFORMERS_VERSION:-${DEFAULT_TRANSFORMERS_VERSION}}"
mkdir -p "$TRITON_CACHE_DIR"

uv run \
  --with "transformers==${ACTIVATION_BEACON_TRANSFORMERS_VERSION}" \
  --with 'deepspeed==0.19.2' \
  python src/pretrain_activation_beacon.py --config "$CONFIG_PATH" "$@"

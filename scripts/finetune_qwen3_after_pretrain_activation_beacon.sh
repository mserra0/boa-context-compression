#!/bin/bash
set -euo pipefail

CONFIG_PATH="${1:-configs/train/config_activation_beacon_qwen3_17b_fullft_after_fineweb_edu_10b_mrqa_official.yaml}"
shift || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

export ACTIVATION_BEACON_TRANSFORMERS_VERSION="${ACTIVATION_BEACON_TRANSFORMERS_VERSION:-4.57.1}"
bash scripts/train_activation_beacon.sh "$CONFIG_PATH" "$@"

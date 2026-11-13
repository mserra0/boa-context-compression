#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/eval/config_sac_r8_mrqa.yaml}"
DEVICE="${2:-cuda}"

uv run python src/evaluate_sac.py --eval-config "$CONFIG_PATH" --device "$DEVICE"

#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:?Usage: bash scripts/run_sac_pretrain.sh <config.json> [work_dir] [port]}"
WORK_DIR="${2:-/path/to/checkpoints/sac/$(basename "${CONFIG_PATH%.*}")}"
PORT="${3:-${SAC_PORT:-14527}}"

mkdir -p "$WORK_DIR"
cp "$CONFIG_PATH" "$WORK_DIR/config.json"

cd "$(dirname "$0")/../SAC/pretrain"
uv run python pre_trainer.py --work_dir "$WORK_DIR" --port "$PORT"


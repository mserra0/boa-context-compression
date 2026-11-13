#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:?Usage: bash scripts/run_sac_finetune.sh <config.json> [work_dir] [port]}"
WORK_DIR="${2:-/path/to/checkpoints/sac/$(basename "${CONFIG_PATH%.*}")}"
PORT="${3:-${SAC_PORT:-14527}}"

mkdir -p "$WORK_DIR"
cp "$CONFIG_PATH" "$WORK_DIR/config.json"
if [[ ! -f "$WORK_DIR/output/adapter.pt" ]]; then
  echo "[ERROR] Missing pretraining adapter: $WORK_DIR/output/adapter.pt" >&2
  echo "Run scripts/run_sac_pretrain.sh first, or pass the work_dir that already contains output/adapter.pt." >&2
  exit 1
fi

cd "$(dirname "$0")/../SAC/sft"
uv run python instruction_trainer.py --work_dir "$WORK_DIR" --port "$PORT"

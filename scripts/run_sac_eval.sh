#!/usr/bin/env bash
set -euo pipefail

WORK_DIR="${1:?Usage: bash scripts/run_sac_eval.sh <work_dir> [batch_size]}"
BATCH_SIZE="${2:-1}"

if [[ ! -f "$WORK_DIR/output/instruction_adapter.pt" ]]; then
  echo "[ERROR] Missing SAC SFT adapter: $WORK_DIR/output/instruction_adapter.pt" >&2
  echo "Run scripts/run_sac_finetune.sh first." >&2
  exit 1
fi

# matplotlib, rouge and nltk are needed by the original SAC evaluator but are not project dependencies,
# so they are added to the environment for this run only.
cd "$(dirname "$0")/../SAC/sft"
uv run --with matplotlib --with rouge --with nltk python instruction_evaluator.py --work_dir "$WORK_DIR" --batch_size "$BATCH_SIZE"
uv run --with matplotlib --with rouge --with nltk python ../util/evaluate_iid.py --work_dir "$WORK_DIR"
uv run --with matplotlib --with rouge --with nltk python ../util/evaluate_ood.py --work_dir "$WORK_DIR"

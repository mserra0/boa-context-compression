#!/usr/bin/env bash
#
# Run the boa student on the RULER long-context benchmark.
#
# Pipeline per (seq_length, task):
#   1. RULER data/prepare.py   -> generate synthetic data jsonl
#   2. src/eval/ruler_predict.py -> compressor predictions (RULER-format jsonl)
#   3. RULER eval/evaluate.py  -> summary.csv per length
#
# RULER has no uv/pyproject of its own, and its prepare.py/evaluate.py resolve imports
# relative to their own directory, so we cd into the right RULER dir and pin the
# repository's uv environment with `uv run --project "$REPO_DIR"`.
#
# We use `--no-sync` throughout: RULER's data-gen deps (nltk, wonderwords, html2text)
# are pip-installed into the venv but are NOT in the compressor's pyproject, so a normal
# `uv run` sync would strip them. Install them once with:
#   uv pip install --python "$REPO_DIR/.venv/bin/python" nltk wonderwords html2text
#
# Usage:
#   bash scripts/eval/run_ruler.sh <EXPERIMENT_DIR>
set -euo pipefail

# --- config -----------------------------------------------------------------
EXPERIMENT_DIR="${1:?usage: bash scripts/eval/run_ruler.sh <EXPERIMENT_DIR>}"
BENCHMARK="synthetic"
SUBSET="validation"
DEVICE="cuda"
GPU="${GPU:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_SAMPLES="${NUM_SAMPLES:-50}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-best_model.pt}"

# Default: a short sweep. Override via env, e.g. SEQ_LENGTHS="4096 8192 16384".
read -r -a SEQ_LENGTHS <<< "${SEQ_LENGTHS:-4096 8192}"
read -r -a TASKS <<< "${TASKS:-niah_single_1 niah_single_2 vt qa_1}"

# --- paths ------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"          # boa-context-compression repo root
if [[ ! -d "$REPO_DIR/../RULER" ]]; then
  echo "ERROR: RULER checkout not found at $REPO_DIR/../RULER (clone https://github.com/NVIDIA/RULER next to this repository)." >&2
  exit 1
fi
RULER_DIR="$(cd "$REPO_DIR/../RULER" && pwd)"        # sibling RULER checkout
RUN_NAME="$(basename "$EXPERIMENT_DIR")"
ROOT_DIR="$REPO_DIR/benchmark_root/$RUN_NAME/$BENCHMARK"

uv_run() { uv run --no-sync --project "$REPO_DIR" python "$@"; }

# Tokenizer used for RULER data prep = the checkpoint's frozen LLM tokenizer.
TOKENIZER_PATH="$(uv_run -c "import yaml; print(yaml.safe_load(open('$EXPERIMENT_DIR/config.yaml'))['llm_model_name'])")"
echo "Experiment : $EXPERIMENT_DIR"
echo "Tokenizer  : $TOKENIZER_PATH (hf)"
echo "Lengths    : ${SEQ_LENGTHS[*]}"
echo "Tasks      : ${TASKS[*]}"
echo "Samples    : $NUM_SAMPLES"

# --- run --------------------------------------------------------------------
for SEQ in "${SEQ_LENGTHS[@]}"; do
    DATA_DIR="$ROOT_DIR/$SEQ/data"
    PRED_DIR="$ROOT_DIR/$SEQ/pred"
    mkdir -p "$DATA_DIR" "$PRED_DIR"

    for TASK in "${TASKS[@]}"; do
        echo "=== [len $SEQ] prepare $TASK ==="
        ( cd "$RULER_DIR/scripts/data" && \
          CUDA_VISIBLE_DEVICES="$GPU" uv run --no-sync --project "$REPO_DIR" python prepare.py \
            --save_dir "$DATA_DIR" \
            --benchmark "$BENCHMARK" \
            --task "$TASK" \
            --tokenizer_path "$TOKENIZER_PATH" \
            --tokenizer_type hf \
            --max_seq_length "$SEQ" \
            --model_template_type base \
            --num_samples "$NUM_SAMPLES" )

        echo "=== [len $SEQ] predict $TASK ==="
        CUDA_VISIBLE_DEVICES="$GPU" uv run --no-sync --project "$REPO_DIR" python "$REPO_DIR/src/eval/ruler_predict.py" \
            --experiment-dir "$EXPERIMENT_DIR" \
            --data-dir "$DATA_DIR" \
            --save-dir "$PRED_DIR" \
            --task "$TASK" \
            --subset "$SUBSET" \
            --max-seq-len "$SEQ" \
            --batch-size "$BATCH_SIZE" \
            --device "$DEVICE" \
            --checkpoint-name "$CHECKPOINT_NAME"
    done

    echo "=== [len $SEQ] evaluate ==="
    ( cd "$RULER_DIR/scripts/eval" && \
      uv run --no-sync --project "$REPO_DIR" python evaluate.py --data_dir "$PRED_DIR" --benchmark "$BENCHMARK" )
    echo "Summary for length $SEQ: $PRED_DIR/summary.csv"
done

echo "Done. Results under $ROOT_DIR/<seqlen>/pred/summary*.csv"

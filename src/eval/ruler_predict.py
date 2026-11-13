"""Run the boa student on RULER synthetic tasks.

RULER (https://github.com/NVIDIA/RULER) generates one prompt string per sample. The
compressor, however, works on a (context, question) pair: it compresses `context` into
soft prompts and — for `compress_question=False` (Design B-ctx) checkpoints — keeps
`question` verbatim. So for each RULER sample we split `input` into:

    [preamble + haystack]  ->  context  (compressed)
    [trailing question line]  ->  question  (kept verbatim)

RULER's synthetic generators always place the task question on the FINAL line of `input`
(after the last newline); everything before it is the preamble + haystack. So we split on
the last newline — a generator-agnostic rule that survives the generators' own singular/
plural rephrasing (e.g. niah's "What are all the special magic numbers" -> "What is the
special magic number"). The per-sample answer_prefix lives in a separate jsonl field and
is NOT part of `input`; we do not append it, matching RULER's own call_api which feeds
only `input`.

NOTE: whether the question is actually kept verbatim depends on the *loaded checkpoint*.
`ModelRunner.run_student` reads `compress_question` from the model, not from us. For a
`compress_question=True` checkpoint the split is harmless: the question is simply
re-appended to the context and compressed with it.

Pipeline role: this replaces RULER's `pred/call_api.py`. It reads the jsonl produced by
RULER's `data/prepare.py` and writes RULER-format prediction jsonl (original fields plus
`pred`) that RULER's `eval/evaluate.py` scores.

Usage (run from repo root, like src/evaluate.py):
    uv run python src/eval/ruler_predict.py \
        --experiment-dir /path/to/checkpoint_dir \
        --data-dir  benchmark_root/<run>/synthetic/4096/data \
        --save-dir  benchmark_root/<run>/synthetic/4096/pred \
        --task      niah_single_1 \
        --max-seq-len 4096
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.model_runner import ModelRunner


# Base RULER task -> default tokens_to_generate (from RULER/scripts/data/synthetic/constants.py).
TASK_TOKENS = {
    "niah": 128,
    "variable_tracking": 30,
    "common_words_extraction": 120,
    "freq_words_extraction": 50,
    "qa": 32,
}

# RULER config task name (from synthetic.yaml) -> base task. Matched by prefix.
_PREFIX_TO_BASE = [
    ("niah", "niah"),
    ("vt", "variable_tracking"),
    ("cwe", "common_words_extraction"),
    ("fwe", "freq_words_extraction"),
    ("qa", "qa"),
]


def resolve_base_task(task_name: str) -> str:
    for prefix, base in _PREFIX_TO_BASE:
        if task_name.startswith(prefix):
            return base
    raise ValueError(
        f"Unknown RULER task '{task_name}'. Known prefixes: {[p for p, _ in _PREFIX_TO_BASE]}"
    )


def split_context_question(text: str) -> tuple[str, str]:
    """Split a RULER prompt into (context, question) on the last newline.

    The question is the final line; the preamble + haystack (to compress) is everything
    before it. If there is no newline, treat the whole prompt as the question.
    """
    idx = text.rfind("\n")
    if idx == -1:
        return "", text
    return text[:idx], text[idx + 1:]


def read_jsonl(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Run the compressor student on a RULER task")
    parser.add_argument("--experiment-dir", required=True,
                        help="Checkpoint dir containing best_model.pt and config.yaml")
    parser.add_argument("--data-dir", required=True, type=Path,
                        help="RULER data dir containing <task>/<subset>.jsonl")
    parser.add_argument("--save-dir", required=True, type=Path,
                        help="Output dir for <task>.jsonl prediction files")
    parser.add_argument("--task", required=True, help="RULER task name, e.g. niah_single_1")
    parser.add_argument("--subset", default="validation")
    parser.add_argument("--max-seq-len", type=int, required=True,
                        help="Max context length fed to the compressor. Set >= the RULER "
                             "sequence length so long context is NOT truncated.")
    parser.add_argument("--tokens-to-generate", type=int, default=None,
                        help="Override generation length (default: RULER per-task value)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint-name", default="best_model.pt",
                        help="Checkpoint filename inside experiment-dir, e.g. "
                             "epoch_1_model.pt for a periodic (non-best) save")
    args = parser.parse_args()

    base_task = resolve_base_task(args.task)
    tokens_to_generate = args.tokens_to_generate or TASK_TOKENS[base_task]

    task_file = args.data_dir / args.task / f"{args.subset}.jsonl"
    if not task_file.exists():
        raise FileNotFoundError(f"RULER data file not found: {task_file}")

    data = read_jsonl(task_file)
    print(f"[ruler_predict] task={args.task} base={base_task} "
          f"samples={len(data)} max_seq_len={args.max_seq_len} "
          f"tokens_to_generate={tokens_to_generate}")

    # Split each prompt into compressed context + verbatim trailing-question line.
    samples = []
    empty_q = 0
    for row in data:
        context, question = split_context_question(row["input"])
        if not question:
            empty_q += 1
        samples.append({"context": context, "question": question})
    if empty_q:
        print(f"[ruler_predict] WARNING: {empty_q}/{len(data)} samples had no trailing "
              f"question line (whole prompt compressed).")

    runner = ModelRunner(args.experiment_dir, device=args.device, checkpoint_name=args.checkpoint_name)
    # Override so RULER's long context is not truncated and generation length matches the task.
    runner.config["max_seq_len"] = args.max_seq_len
    runner.config["max_new_tokens"] = tokens_to_generate
    print(f"[ruler_predict] model compress_question="
          f"{getattr(runner.model, 'compress_question', True)} "
          f"student_template={getattr(runner.model, 'student_template', 'bare')} "
          f"prompt_mode={getattr(runner.model, 'prompt_mode', 'embed')}")

    preds = runner.run_student(samples, batch_size=args.batch_size)
    assert len(preds) == len(data), f"pred/data count mismatch: {len(preds)} vs {len(data)}"

    args.save_dir.mkdir(parents=True, exist_ok=True)
    pred_file = args.save_dir / f"{args.task}.jsonl"
    with open(pred_file, "w", encoding="utf-8") as fout:
        for row, pred in zip(data, preds):
            out = dict(row)  # preserve index/input/outputs/length for evaluate.py
            # RULER's evaluate.py reads line['others'].get('id', ...); call_api injects this
            # field, so we must too (RULER data generators don't always emit it).
            out.setdefault("others", {})
            out["pred"] = pred
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"[ruler_predict] wrote {len(data)} predictions to {pred_file}")


if __name__ == "__main__":
    main()

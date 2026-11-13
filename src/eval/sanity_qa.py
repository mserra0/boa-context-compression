"""Sanity check: run the compressor on its training distribution (MRQA extractive QA).

This script runs the model on the data it was trained on — short, single-context
extractive QA (MRQA dev) — through the SAME ModelRunner path the RULER predictor uses,
and reports three numbers with simple SQuAD-style matching (no vLLM judge):

    teacher_ctx    frozen LLM WITH the full context   (upper bound)
    student        compressor soft prompts            (compressed context)
    teacher_noctx  frozen LLM with NO context         (closed-book lower bound)

A healthy compressor should land student well above teacher_noctx and approaching
teacher_ctx. Uses the checkpoint's native config (e.g. max_seq_len 1024) — NOT overridden.

Usage:
    uv run --no-sync python src/eval/sanity_qa.py \
        --experiment-dir /path/to/checkpoint_dir \
        --dataset mrqa_squad --num-samples 30
"""

import argparse
import os
import re
import string
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.model_runner import ModelRunner
from src.utils.evaluate_utils import DatasetLoader


def normalize(s: str) -> str:
    """SQuAD answer normalization: lowercase, strip punctuation/articles/extra whitespace."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def contains_match(pred: str, gold: str) -> bool:
    """Lenient core-answer match: normalized gold appears in normalized prediction."""
    g = normalize(gold)
    return bool(g) and g in normalize(pred)


def accuracy(preds: list, golds: list) -> float:
    hits = sum(contains_match(p, g) for p, g in zip(preds, golds))
    return 100.0 * hits / max(len(preds), 1)


def main():
    parser = argparse.ArgumentParser(description="Trained-distribution QA sanity check")
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--dataset", default="mrqa_squad",
                        help="One of DatasetLoader keys, e.g. mrqa_squad/mrqa_newsqa/"
                             "mrqa_trivia/mrqa_hotpot/squad")
    parser.add_argument("--mrqa-data-dir", default="/path/to/data/MRQA/dev")
    parser.add_argument("--num-samples", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--show", type=int, default=8, help="how many examples to print")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    datasets_cfg = {
        "squad": args.dataset == "squad",
        "mrqa_squad": args.dataset == "mrqa_squad",
        "mrqa_newsqa": args.dataset == "mrqa_newsqa",
        "mrqa_trivia": args.dataset == "mrqa_trivia",
        "mrqa_hotpot": args.dataset == "mrqa_hotpot",
        "mrqa_data_dir": args.mrqa_data_dir,
        "max_samples": args.num_samples,
    }
    samples = DatasetLoader(datasets_cfg).load()[args.dataset]
    golds = [s["answer"] for s in samples]
    print(f"[sanity_qa] dataset={args.dataset} samples={len(samples)}")

    runner = ModelRunner(args.experiment_dir, device=args.device)
    print(f"[sanity_qa] compress_question={getattr(runner.model, 'compress_question', True)} "
          f"student_template={getattr(runner.model, 'student_template', 'bare')} "
          f"prompt_mode={getattr(runner.model, 'prompt_mode', 'embed')} "
          f"max_seq_len={runner.config.get('max_seq_len')} "
          f"max_new_tokens={runner.config.get('max_new_tokens')}")

    student = runner.run_student(samples, batch_size=args.batch_size)
    teacher_ctx = runner.run_teacher(samples, use_context=True, batch_size=args.batch_size)
    teacher_noctx = runner.run_teacher(samples, use_context=False, batch_size=args.batch_size)

    print("\n" + "=" * 80)
    for i in range(min(args.show, len(samples))):
        print(f"Q       : {samples[i]['question']}")
        print(f"gold    : {golds[i]}")
        print(f"student : {student[i][:160]!r}  {'OK' if contains_match(student[i], golds[i]) else 'x'}")
        print(f"teacher : {teacher_ctx[i][:160]!r}  {'OK' if contains_match(teacher_ctx[i], golds[i]) else 'x'}")
        print("-" * 80)

    print("\n=== accuracy (normalized contains-match) ===")
    print(f"  teacher_ctx   (upper bound)        : {accuracy(teacher_ctx, golds):5.1f}%")
    print(f"  student       (compressed)         : {accuracy(student, golds):5.1f}%")
    print(f"  teacher_noctx (closed-book floor)  : {accuracy(teacher_noctx, golds):5.1f}%")


if __name__ == "__main__":
    main()

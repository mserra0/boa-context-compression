"""Graded-difficulty needle-in-a-haystack in RULER format.

RULER's niah combines several difficulty axes (4K+ noise, exact 7-digit needle). This builds
the SAME niah structure (preamble + haystack-with-needle + trailing question line — exactly
what src/eval/ruler_predict.py splits on the last newline) but varies two axes independently:

  * needle type : a memorable WORD  vs an exact 7-digit NUMBER (RULER's value type)
  * context len : short (~128 tok, well within training) -> ~1024 (training max)

Samples are generated in RULER's `input`/`outputs` schema and run through the SAME
ModelRunner.run_student path, then scored with RULER's own rule (gold substring in pred,
case-insensitive). Reports student vs teacher-with-context per level, so the effect of needle
type and of context length can be separated.

Usage:
    uv run --no-sync python src/eval/mini_niah.py \
        --experiment-dir /path/to/checkpoint_dir --num-samples 20
"""

import argparse
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.model_runner import ModelRunner

PREAMBLE = ("Some special magic values are hidden within the following text. "
            "Make sure to memorize them. I will quiz you about them afterwards.")

FILLER = [
    "The grass is green and the sky is blue.",
    "A gentle breeze moved through the quiet valley.",
    "She poured a cup of coffee and read the morning news.",
    "The train arrived at the station a few minutes late.",
    "Children played in the park until the sun went down.",
    "The old library smelled of dust and paper.",
    "He tightened the last bolt and wiped his hands.",
    "Rain tapped softly against the window all afternoon.",
    "The market was crowded with people buying fresh fruit.",
    "A cat stretched lazily on the warm stone step.",
    "The mountain trail wound upward through the pines.",
    "They watched the boats drift across the calm harbor.",
    "The bakery sold out of bread before noon.",
    "A single lamp lit the corner of the small room.",
    "The river carried leaves slowly toward the bridge.",
]

KEY_ADJ = ["jittery", "roomy", "smoggy", "typical", "scintillating", "quiet", "brave",
           "hollow", "amber", "restless", "velvet", "narrow", "polished", "distant"]
KEY_NOUN = ["hospital", "devil", "caliber", "toilet", "slipper", "garden", "engine",
            "harbor", "lantern", "meadow", "compass", "anchor", "pebble", "orchard"]
WORD_VALUES = ["pomegranate", "tambourine", "aardvark", "kaleidoscope", "marmalade",
               "zeppelin", "wheelbarrow", "harmonica", "cactus", "typhoon", "walnut",
               "obsidian", "mahogany", "flamingo", "saxophone", "avalanche"]


def make_key(rng):
    return f"{rng.choice(KEY_ADJ)}-{rng.choice(KEY_NOUN)}"


def make_value(needle_type, rng):
    if needle_type == "word":
        return rng.choice(WORD_VALUES)
    if needle_type == "num3":
        return str(rng.randint(100, 999))
    return str(rng.randint(1_000_000, 9_999_999))  # num7 (RULER's value type)


_FILLER_TOK_LEN = {}  # cache: filler sentence -> token length (built once per tokenizer)


def build_sample(needle_type, target_tokens, tokenizer, rng):
    key = make_key(rng)
    value = make_value(needle_type, rng)
    needle = f"One of the special magic values: the value for {key} is {value}."

    if not _FILLER_TOK_LEN:
        for s in FILLER:
            _FILLER_TOK_LEN[s] = len(tokenizer(s)["input_ids"]) + 1  # +1 for the "\n" join

    # Grow the haystack (filler sentences) to the target token budget — O(n), no re-tokenizing
    # the whole growing string (matters at 16K+ tokens).
    sentences, tot = [], 0
    while tot < target_tokens:
        s = rng.choice(FILLER)
        sentences.append(s)
        tot += _FILLER_TOK_LEN[s]
    # Insert the needle at a random position among the filler.
    sentences.insert(rng.randint(0, len(sentences)), needle)

    haystack = "\n".join(sentences)
    question = f"What is the special magic value for {key} mentioned in the provided text?"
    input_text = f"{PREAMBLE}\n{haystack}\n{question}"
    return {"input": input_text, "outputs": [value], "question": question,
            "context": f"{PREAMBLE}\n{haystack}"}


def score(preds, samples):
    hits = sum(any(o.lower() in p.lower() for o in s["outputs"])
               for p, s in zip(preds, samples))
    return 100.0 * hits / max(len(preds), 1)


def main():
    parser = argparse.ArgumentParser(description="Graded RULER-format needle sweep")
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show", type=int, default=3, help="examples to print per level")
    parser.add_argument("--needle-type", default="num7", choices=["word", "num3", "num7"],
                        help="needle value type when --lengths is given")
    parser.add_argument("--lengths", default=None,
                        help="comma-separated target context token lengths; if set, sweeps "
                             "these with --needle-type instead of the default mixed levels")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    runner = ModelRunner(args.experiment_dir, device=args.device)
    tok = runner.llm_tokenizer
    print(f"[mini_niah] compress_question={getattr(runner.model, 'compress_question', True)} "
          f"student_template={getattr(runner.model, 'student_template', 'bare')} "
          f"prompt_mode={getattr(runner.model, 'prompt_mode', 'embed')}")

    # (name, needle_type, target context tokens). Either a custom length sweep for one
    # needle type (--lengths), or the default mixed levels isolating needle-type vs length.
    if args.lengths:
        tgts = [int(x) for x in args.lengths.split(",")]
        levels = [(f"{args.needle_type} @~{t} tok", args.needle_type, t) for t in tgts]
    else:
        levels = [
            ("word  @~128 tok", "word", 128),
            ("num7  @~128 tok", "num7", 128),
            ("num7  @~512 tok", "num7", 512),
            ("num7  @~1000 tok", "num7", 1000),
            ("word  @~1000 tok", "word", 1000),
        ]

    results = []
    for name, ntype, tgt in levels:
        rng = random.Random(args.seed)
        samples = [build_sample(ntype, tgt, tok, rng) for _ in range(args.num_samples)]
        # Report the actual median context length so the "target" is grounded.
        ctx_lens = sorted(len(tok(s["context"])["input_ids"]) for s in samples)
        med_len = ctx_lens[len(ctx_lens) // 2]

        # Override generation length: a needle answer is short.
        runner.config["max_seq_len"] = max(tgt + 128, 1024)
        runner.config["max_new_tokens"] = 32

        student = runner.run_student(samples, batch_size=args.batch_size)
        teacher = runner.run_teacher(samples, use_context=True, batch_size=args.batch_size)
        s_acc, t_acc = score(student, samples), score(teacher, samples)
        results.append((name, med_len, s_acc, t_acc))

        print(f"\n=== {name}  (median ctx {med_len} tok) ===")
        for i in range(min(args.show, len(samples))):
            gold = samples[i]["outputs"][0]
            print(f"  gold={gold:>12} | student={student[i][:90]!r} "
                  f"{'OK' if any(o.lower() in student[i].lower() for o in samples[i]['outputs']) else 'x'}")

    print("\n" + "=" * 64)
    print(f"{'level':<20}{'ctx_tok':>9}{'student':>10}{'teacher':>10}")
    print("-" * 64)
    for name, med_len, s_acc, t_acc in results:
        print(f"{name:<20}{med_len:>9}{s_acc:>9.1f}%{t_acc:>9.1f}%")


if __name__ == "__main__":
    main()

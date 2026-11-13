# Baseline QA Evaluation (plain HF models, no compressor)

`src/baselines/evaluate.py` evaluates a bare Hugging Face causal LM on MRQA dev
sets under two conditions — **context** (gold passage + question, RAG-style upper
bound) and **nocontext** (question only, closed-book floor) — with two selectable
scorers:

- `--scorer f1_em` — official MRQA span **F1/EM** (max over all gold answers). No
  server required. Loads the full gold list via `src/evaluation/`.
- `--scorer judge` (default) — LLM-as-judge accuracy (needs a running vLLM server).

## Zero-shot format flag

A fine-tuned model learns bare-span output from its SFT answer targets. A
**non-fine-tuned base model** does not: with the plain prompt it answers in full
sentences (correct content, but EM≈0 and deflated F1). Pass `--zero-shot` to append
a span-only directive to the system message so the base model emits bare spans —
required for a fair base-vs-FullFT F1/EM comparison.

Do **not** pass `--zero-shot` for a fine-tuned model: it was trained on the plain
prompt, and the extra directive would drift from its training distribution.

## Qwen3-4B FullFT

The FullFT baseline (upper-bound teacher; also the offline-KD logits source) was
trained on all six in-domain MRQA sets. Checkpoint locations:

| Item | Path |
|---|---|
| Checkpoint | `/path/to/checkpoints/qwen3_4b_fullft/final` |

Config: `configs/baselines/mn5_qwen3_4b_fullft.yaml`.

### Results — SQuAD dev (F1/EM, 500 samples)

| Model | Condition | EM | F1 |
|---|---|---|---|
| Qwen3-4B base (zero-shot) | context | 79.00 | 86.41 |
| **Qwen3-4B FullFT** | context | **94.00** | **96.93** |
| Qwen3-4B base (zero-shot) | nocontext | 5.40 | 12.78 |
| Qwen3-4B FullFT | nocontext | 9.40 | 14.89 |

FullFT sharpens exact-span extraction: **+15 EM / +10.5 F1** on the context
condition over an already-strong zero-shot base. The `context` number is the
teacher upper bound. `nocontext` is low for both (SQuAD is extractive — the answer
lives in the passage).

Result caches are written to `evals/baselines/<name>/` (`mrqa_squad_{context,nocontext}_f1_em.json` and `summary_f1_em.json`).

## Reproduce

```bash
# base (zero-shot, span-only prompt), no vLLM needed for f1_em
CUDA_VISIBLE_DEVICES=0 uv run python -m src.baselines.evaluate \
  --eval-config configs/eval/config_eval.yaml \
  --model Qwen/Qwen3-4B --name Qwen3-4B-base --no-think --scorer f1_em --zero-shot

# FullFT (plain prompt — as trained)
CUDA_VISIBLE_DEVICES=0 uv run python -m src.baselines.evaluate \
  --eval-config configs/eval/config_eval.yaml \
  --model /path/to/checkpoints/qwen3_4b_fullft/final \
  --name Qwen3-4B-FullFT --no-think --scorer f1_em
```

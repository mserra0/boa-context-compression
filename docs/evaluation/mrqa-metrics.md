# MRQA F1/EM Evaluation

The `src/evaluation/` package scores a checkpoint on the **MRQA Shared Task 2019** dev
sets with the **official F1 / Exact-Match** span metrics used across the context-compression
literature (SAC, ICAE, …). It evaluates both **in-distribution** and **out-of-distribution**
dev sets. An **LLM-as-judge** scorer is also available as an alternative.

```bash
# Span metrics (no server needed)
CUDA_VISIBLE_DEVICES=0 uv run python -m src.evaluation \
  --config configs/eval/config_eval_MRQA.yaml \
  --experiment-dir /path/to/checkpoints/<run_dir> \
  --device cuda

# Or via the helper script (edit EXPERIMENTS in the script first)
bash scripts/eval/eval_mrqa.sh
```

Inference is delegated entirely to [`ModelRunner`](../../src/utils/model_runner.py) — the same
training-aligned generation path the compressor was trained against — so evaluation has **no
train/inference gap**. The evaluation package never re-implements generation.

---

## Package Layout

One self-contained package, one concern per module (keeps the crowded `src/` root clean):

| Module | Responsibility |
|---|---|
| `metrics.py` | Official MRQA/SQuAD-v1.1 primitives: `normalize_answer`, `f1_score`, `exact_match_score`, `metric_max_over_ground_truths`. Pure functions, no deps. |
| `datasets.py` | Multi-gold dev-set loader + in-domain / OOD file maps and grouping. |
| `scorers.py` | `SpanMetricScorer` (F1/EM) and `LLMJudgeScorer` (judge alternative). One shared interface. |
| `runner.py` | `EvalRunner` — orchestrates dataset × condition × scorer, caching, macro averages, W&B, reporting. |
| `__main__.py` | CLI entry point (`python -m src.evaluation`). |

Config: [`configs/eval/config_eval_MRQA.yaml`](../../configs/eval/config_eval_MRQA.yaml) ·
Script: [`scripts/eval/eval_mrqa.sh`](../../scripts/eval/eval_mrqa.sh)

An LLM-judge entry point (`src/evaluate.py` + `src/utils/evaluate_utils.py`) is also available;
`LLMJudgeScorer` reuses its `LLMJudge` rather than duplicating it.

---

## Metrics

Vendored verbatim (behaviour-preserving) from the
[MRQA official eval script](https://github.com/mrqa/MRQA-Shared-Task-2019/blob/master/mrqa_official_eval.py):

- `normalize_answer` — lowercase, strip punctuation, drop articles (`a`/`an`/`the`), collapse whitespace.
- `exact_match_score` — normalized string equality.
- `f1_score` — token-overlap F1; `0` when there is no overlap.
- `metric_max_over_ground_truths` — **max of the metric over all gold answers** for a question.

Reported numbers are the mean over samples, printed as percentages (`×100`).

---

## Datasets — the multi-gold requirement

> **Note.** The training loader `src.dataset.mrqa_generator` keeps only
> `detected_answers[0]['text']` — a *single* gold. Official F1/EM takes the **max over ALL
> golds** in `qa['answers']`. Evaluation therefore has its **own** generator,
> `mrqa_eval_generator`, which yields the full gold list. **Do not reuse the training
> generator for evaluation** — it silently under-reports F1/EM.

`mrqa_eval_generator(path)` yields `{context, question, answers: list[str], qid}` per QA,
skipping header lines and answerless QAs.

`load_eval_datasets(root, in_domain, out_of_domain, max_samples)` returns metrics grouped by
distribution:

```python
{"in_domain": {name: [samples]}, "out_of_domain": {name: [samples]}}
```

Data root: `/path/to/data/MRQA`, with two subdirs:

| Group | Subdir | Datasets |
|---|---|---|
| In-distribution | `dev/` | SQuAD, NewsQA, TriviaQA, HotpotQA, SearchQA, NaturalQuestions |
| Out-of-distribution | `dev_ood/` | BioASQ, DROP, DuoRC, RACE, RelationExtraction, TextbookQA |

Logical config names map to local filenames in `MRQA_INDOMAIN` / `MRQA_OOD` (local names, e.g.
`TriviaQA.jsonl.gz`, not the official `TriviaQA-web` / `NaturalQuestionsShort` / `DuoRC.ParaphraseRC`).

---

## Scorers

Both expose the same interface, so the runner is scorer-agnostic:

```python
metric_keys -> tuple[str, ...]                 # e.g. ("em", "f1")
score(samples, preds) -> (per_sample, aggregate)
```

| Scorer | `scorer:` value | Metrics | Server |
|---|---|---|---|
| `SpanMetricScorer` | `f1_em` (default) | `em`, `f1` | none |
| `LLMJudgeScorer` | `llm_judge` | `accuracy` | vLLM judge required |

`LLMJudgeScorer` wraps the existing `LLMJudge`; it joins the gold aliases into one reference
string for the judge prompt.

---

## Runner

`EvalRunner` runs, per dataset, any subset of three **conditions** (config `conditions:` toggle):

| Condition | Source | Meaning |
|---|---|---|
| `student` | `ModelRunner.run_student` | the compressor under evaluation |
| `teacher_context` | `ModelRunner.run_teacher(use_context=True)` | frozen-LLM baseline, with context |
| `teacher_nocontext` | `ModelRunner.run_teacher(use_context=False)` | frozen-LLM closed-book baseline |

Each condition is reported independently (one table per condition), with three macro rows:
**in-distribution avg**, **OOD avg**, and **overall avg** (macro over *all* datasets, not the
mean of group means). Top-level summary keys mirror the "primary" condition — `student` if run,
otherwise the first requested condition — so teacher-only runs are reported too.

### Caching & outputs

Results are cached as JSON; the scorer tag is in every filename so `f1_em` and `llm_judge`
caches never collide.

```
evals/experiments/<run>/<group>_<dataset>_student_<scorer>.json      # student
evals/baselines/<llm>/<group>_<dataset>_<teacher_cond>_<scorer>.json  # teacher baselines
evals/experiments/<run>/summary_<scorer>.json                        # full summary
```

`force_recompute_experiment` / `force_recompute_baselines` bypass the cache. W&B logging is on
by default (`--no-wandb` or `wandb.enabled: false` to skip); metrics are namespaced
`<condition>/<group>/<dataset>_<metric>`.

---

## Configuration

```yaml
scorer: f1_em                       # f1_em | llm_judge
datasets:
  mrqa_data_dir: "/path/to/data/MRQA"   # contains dev/ and dev_ood/
  max_samples: 500                             # per dataset; null = full dev set
  in_domain:      {squad: true, newsqa: true, ...}
  out_of_domain:  {bioasq: true, drop: true, ...}
conditions:
  student: true
  teacher_context: false
  teacher_nocontext: false
inference: {batch_size: 16}
```

The `judge:` block is only read when `scorer: llm_judge`.

---

## Caveat: span metrics vs. verbose models

F1/EM reward **short extractive** answers. A chatty instruct model that answers correctly but
verbosely scores near-zero EM and low F1; the LLM-as-judge scorer is available for such outputs.

- The fully fine-tuned MRQA teacher (`Qwen3-1.7B FullFT`) emits clean spans → F1/EM is meaningful
  for it.



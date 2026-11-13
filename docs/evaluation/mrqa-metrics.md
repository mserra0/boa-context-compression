# MRQA F1/EM Evaluation Suite

The `src/evaluation/` package scores Boa checkpoints on the **MRQA Shared Task 2019** dev sets with the **official F1 / Exact-Match** span metrics used across the context-compression literature (SAC, ICAE, …). It evaluates both **in-distribution** (6 datasets) and **out-of-distribution** (6 datasets) dev sets.

Scoring is deterministic and offline — no judge model, no server to start.

```bash
# Evaluate a single checkpoint
CUDA_VISIBLE_DEVICES=0 uv run python -m src.evaluation \
  --config configs/eval/config_eval_MRQA.yaml \
  --experiment-dir /path/to/checkpoints/<run_dir> \
  --device cuda

# Batch evaluate multiple checkpoints via helper script
bash scripts/eval/eval_mrqa.sh <run_dir>   # run directory under CKPT_ROOT
```

Inference is delegated entirely to [`ModelRunner`](../../src/utils/model_runner.py) — the training-aligned generation path — ensuring evaluation has **zero train/inference gap**.

---

## Package Layout & Architecture

The evaluation suite is organized into a modular package under `src/evaluation/`:

| Module | Responsibility & Key Symbols |
|---|---|
| [`metrics.py`](../../src/evaluation/metrics.py) | Official MRQA / SQuAD v1.1 span primitives: `normalize_answer`, `f1_score`, `exact_match_score`, `metric_max_over_ground_truths`. Pure functions with zero framework dependencies. |
| [`datasets.py`](../../src/evaluation/datasets.py) | Multi-gold dev-set loader (`mrqa_eval_generator`), `load_eval_datasets`, in-domain (`dev/`) and OOD (`dev_ood/`) dataset file mappings (`MRQA_INDOMAIN`, `MRQA_OOD`). |
| [`scorers.py`](../../src/evaluation/scorers.py) | `SpanMetricScorer` — implements the scorer interface (`metric_keys = ("em", "f1")`, `score(samples, preds)`), evaluating max F1/EM over multi-gold lists. |
| [`runner.py`](../../src/evaluation/runner.py) | `EvalRunner` — orchestrates dataset × condition × scorer execution, JSON caching (`evals/`), macro-averages (`in_domain_avg`, `out_of_domain_avg`, `overall_avg`), rich console reporting, and W&B logging. |
| [`io.py`](../../src/evaluation/io.py) | `save_results`, `load_results`, `results_exist`, and `_make_progress` rich progress bar styling. |
| [`__main__.py`](../../src/evaluation/__main__.py) | CLI entry point accepting `--config`, `--experiment-dir`, `--device`, and `--no-wandb`. |

---

## Metrics (`metrics.py`)

Vendored verbatim (behaviour-preserving) from the official [MRQA 2019 Shared Task evaluation script](https://github.com/mrqa/MRQA-Shared-Task-2019/blob/master/mrqa_official_eval.py):

- **`normalize_answer(s: str)`**: Lowercases text, strips punctuation, removes articles (`a`, `an`, `the`), and collapses whitespace.
- **`exact_match_score(prediction, ground_truth)`**: Returns `True` if `normalize_answer(prediction) == normalize_answer(ground_truth)`.
- **`f1_score(prediction, ground_truth)`**: Computes token-level precision, recall, and harmonic F1 overlap on normalized token lists. Returns `0.0` when tokens do not overlap.
- **`metric_max_over_ground_truths(metric_fn, prediction, ground_truths)`**: Evaluates `metric_fn` against every gold answer candidate in `ground_truths` and returns the **maximum score**.

All aggregate reported values are sample means expressed as percentages (`×100`).

---

## Multi-Gold Dev Sets (`datasets.py`)

> **Multi-Gold Requirement.** The training generator `src.dataset.mrqa_generator` keeps only `detected_answers[0]['text']` (a single gold string). Official MRQA scoring requires taking the **max over ALL acceptable gold answers** in `qa['answers']`. Evaluation therefore uses `mrqa_eval_generator`, which yields `answers: list[str]`. Reusing the training generator for evaluation will under-report F1 and EM.

`mrqa_eval_generator(file_path)` parses `.jsonl.gz` dataset files, yielding `{context, question, answers, qid}` while skipping header metadata and QAs lacking gold answers.

`load_eval_datasets(root, in_domain, out_of_domain, max_samples)` returns dev samples grouped by distribution:

```python
{
    "in_domain": {"squad": [...], "newsqa": [...], ...},
    "out_of_domain": {"bioasq": [...], "drop": [...], ...}
}
```

### Dataset File Map (Root: `/path/to/data/MRQA`)

| Group | Subdirectory | Logical Names & File Mappings |
|---|---|---|
| In-Distribution (`dev/`) | `dev/` | `squad` (`SQuAD.jsonl.gz`), `newsqa` (`NewsQA.jsonl.gz`), `triviaqa` (`TriviaQA.jsonl.gz`), `hotpotqa` (`HotpotQA.jsonl.gz`), `searchqa` (`SearchQA.jsonl.gz`), `naturalquestions` (`NaturalQuestions.jsonl.gz`) |
| Out-Of-Distribution (`dev_ood/`) | `dev_ood/` | `bioasq` (`BioASQ.jsonl.gz`), `drop` (`DROP.jsonl.gz`), `duorc` (`DuoRC.jsonl.gz`), `race` (`RACE.jsonl.gz`), `relationextraction` (`RelationExtraction.jsonl.gz`), `textbookqa` (`TextbookQA.jsonl.gz`) |

---

## Scorers & Interface Contract (`scorers.py`)

`EvalRunner` interacts with metric calculators strictly through the Scorer Interface:

```python
class ScorerInterface:
    metric_keys: tuple[str, ...]
    def score(self, samples: list[dict], preds: list[str], progress=None, task_id=None) -> tuple[list[dict], dict]:
        ...
```

`SpanMetricScorer` is the primary scorer. It iterates over (sample, prediction) pairs, applies `metric_max_over_ground_truths` for EM and F1 across `sample["answers"]`, and computes dataset-level macro means.

---

## Evaluation Orchestration & Baseline Conditions (`runner.py`)

`EvalRunner` supports evaluating three distinct experimental conditions toggled via configuration:

| Condition Key | Generation Method | Description |
|---|---|---|
| `student` | `ModelRunner.run_student` | Evaluates the compressed context soft prompt generator under test |
| `teacher_context` | `ModelRunner.run_teacher(use_context=True)` | Upper-bound baseline using full uncompressed context with frozen teacher LLM |
| `teacher_nocontext` | `ModelRunner.run_teacher(use_context=False)` | Closed-book baseline (question only, zero context) with frozen teacher LLM |

### Result Caching & Outputs

Evaluation results are saved as structured JSON files under `evals/`:

```
evals/experiments/<run_name>/<group>_<dataset>_student_f1_em.json      # student predictions & metrics
evals/baselines/<llm_safe_name>/<group>_<dataset>_<cond>_f1_em.json   # teacher baselines
evals/experiments/<run_name>/summary_f1_em.json                       # full macro summary
```

Caching prevents redundant inference passes when re-running evaluation with new conditions or dataset subsets. `force_recompute_experiment: true` or `force_recompute_baselines: true` forces cache recomputation.

---

## Batch Evaluation Script (`scripts/eval/eval_mrqa.sh`)

`scripts/eval/eval_mrqa.sh` provides batch evaluation across multiple experiment checkpoints:

```bash
# Evaluate one or more run directories (names resolved against CKPT_ROOT)
scripts/eval/eval_mrqa.sh <run_dir> [<run_dir> ...]

# Environment variable overrides
CKPT_ROOT="/path/to/checkpoints" GPU=0 WANDB=0 bash scripts/eval/eval_mrqa.sh <run_dir>
```

Key environment variables supported by `eval_mrqa.sh`:
- `CKPT_ROOT`: Base directory containing checkpoint folders (default: `/path/to/checkpoints`).
- `CONFIG`: Path to eval YAML configuration (default: `configs/eval/config_eval_MRQA.yaml`).
- `GPU`: CUDA device index (default: `0`).
- `DEVICE`: Compute device (default: `cuda`).
- `WANDB`: Set to `0` to disable Weights & Biases logging (default: `1`).

---


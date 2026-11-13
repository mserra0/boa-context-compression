import asyncio
import json
import logging
import os
import re
import string
import unicodedata
from pathlib import Path

from colorama import Fore
from openai import AsyncOpenAI
from rich.console import Console
from rich.progress import (
    Progress, BarColumn, TextColumn, MofNCompleteColumn,
    TimeElapsedColumn, TimeRemainingColumn, SpinnerColumn,
)
from rich.table import Table

from ..dataset import mrqa_generator, MRQA_FILENAME

_CONDITION_STYLE = {
    "teacher_context":   "[bold blue]Teacher+Ctx   [/bold blue]",
    "teacher_nocontext": "[bold yellow]Teacher-NoCtx[/bold yellow]",
    "student":           "[bold magenta]Student       [/bold magenta]",
}

### Utiliy functions ###

def _parse_score(text: str) -> int:
    """Parse a binary score from judge response. Returns 0 or 1; -1 if invalid."""
    # Strip markdown code fences and thinking tokens before attempting JSON parse
    cleaned = re.sub(r"```(?:json)?\s*", "", text)
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL)
    cleaned = cleaned.strip()

    try:
        obj = json.loads(cleaned)
        # normalize keys to be case-insensitive for "score"; be robust to more variants
        score = obj.get("Score", obj.get("score"))
        if score in (1, "1"):
            return 1
        if score in (0, "0"):
            return 0
        return -1
    except (json.JSONDecodeError, AttributeError):
        pass

    # Try to find a JSON object anywhere in the text
    match = re.search(r"\{[^{}]*\}", cleaned, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group())
            score = obj.get("Score", obj.get("score"))
            if score in (1, "1"):
                return 1
            if score in (0, "0"):
                return 0
        except (json.JSONDecodeError, AttributeError):
            pass

    # Last resort: bare digit
    if cleaned == "1":
        return 1
    if cleaned == "0":
        return 0
    return -1

def compute_accuracy(scores: list) -> float:
    """Mean of valid (>=0) scores. Returns 0.0 if no valid scores."""
    valid = [s for s in scores if s >= 0]
    if not valid:
        return 0.0
    return sum(valid) / len(valid)


# normalize_answer and the QA-F1 helpers below follow the official MRQA 2019 / SQuAD v1.1 evaluation
# scripts (see THIRD_PARTY_NOTICES.md).
def normalize_answer(text: str) -> str:
    """SQuAD-style answer normalization for QA F1."""

    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(
            char
            for char in value
            if char not in exclude and not unicodedata.category(char).startswith("P")
        )

    return white_space_fix(remove_articles(remove_punc(text.lower())))


def qa_f1_score(prediction: str, ground_truth: str) -> float:
    """Token-level QA F1 used by LongBench for English QA tasks."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = {}
    for token in pred_tokens:
        common[token] = min(pred_tokens.count(token), gold_tokens.count(token))
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def qa_f1_score_any(prediction: str, ground_truths) -> float:
    """Max QA F1 over one or more gold answers, matching LongBench style."""
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    return max((qa_f1_score(prediction, gold) for gold in ground_truths), default=0.0)


def exact_match_score(prediction: str, ground_truth: str) -> float:
    """Normalized exact match used by extractive QA benchmarks."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def exact_match_score_any(prediction: str, ground_truths) -> float:
    """Exact match against any accepted gold alias."""
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    return max((exact_match_score(prediction, gold) for gold in ground_truths), default=0.0)


def gold_answers(sample: dict) -> list[str]:
    """Return all accepted gold aliases for a sample, falling back to answer."""
    answers = sample.get("answers")
    if isinstance(answers, str):
        answers = [answers]
    if answers:
        return [answer for answer in answers if answer]
    answer = sample.get("answer", "")
    return [answer] if answer else []




def clean_qa_prediction(prediction: str, ground_truths=None) -> str:
    """Clean verbose QA generations without consulting gold answers.

    The raw prediction is still saved. This helper removes Qwen thinking tags
    and keeps the first non-empty generated line so benchmark scoring does not
    get an oracle view of accepted gold aliases.
    """
    prediction = re.sub(
        r"<think>.*?</think>",
        "",
        str(prediction),
        flags=re.DOTALL | re.IGNORECASE,
    )
    prediction = re.sub(r"</?think>", "", prediction, flags=re.IGNORECASE)
    first_line = next(
        (line.strip() for line in prediction.splitlines() if line.strip()),
        prediction.strip(),
    )
    return first_line

def compute_macro_average(per_dataset: dict) -> dict:
    """Macro-average of metric dicts across datasets."""
    if not per_dataset:
        return {}
    keys = list(next(iter(per_dataset.values())).keys())
    return {
        k: sum(d[k] for d in per_dataset.values()) / len(per_dataset)
        for k in keys
    }

def save_results(data: dict, path: str) -> None:
    """Save results dict as JSON, creating parent dirs if needed."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)

def results_exist(path: str) -> bool:
    return os.path.exists(path)

### Dataset loading ###

MRQA_EVAL_IN_DOMAIN_FILES = {
    "squad": "SQuAD.jsonl.gz",
    "newsqa": "NewsQA.jsonl.gz",
    "triviaqa": "TriviaQA.jsonl.gz",
    "hotpotqa": "HotpotQA.jsonl.gz",
    "searchqa": "SearchQA.jsonl.gz",
    "naturalquestions": "NaturalQuestions.jsonl.gz",
}

MRQA_EVAL_OUT_OF_DOMAIN_FILES = {
    "bioasq": "BioASQ.jsonl.gz",
    "drop": "DROP.jsonl.gz",
    "duorc": "DuoRC.jsonl.gz",
    "race": "RACE.jsonl.gz",
    "relationextraction": "RelationExtraction.jsonl.gz",
    "textbookqa": "TextbookQA.jsonl.gz",
}

MRQA_EVAL_LEGACY_FLAGS = {
    "mrqa_squad": ("mrqa_squad", "SQuAD.jsonl.gz"),
    "mrqa_newsqa": ("mrqa_newsqa", "NewsQA.jsonl.gz"),
    "mrqa_trivia": ("mrqa_trivia", "TriviaQA.jsonl.gz"),
    "mrqa_triviaqa": ("mrqa_triviaqa", "TriviaQA.jsonl.gz"),
    "mrqa_hotpot": ("mrqa_hotpot", "HotpotQA.jsonl.gz"),
    "mrqa_hotpotqa": ("mrqa_hotpotqa", "HotpotQA.jsonl.gz"),
    "mrqa_searchqa": ("mrqa_searchqa", "SearchQA.jsonl.gz"),
    "searchqa": ("mrqa_searchqa", "SearchQA.jsonl.gz"),
    "mrqa_naturalquestions": ("mrqa_naturalquestions", "NaturalQuestions.jsonl.gz"),
    "naturalquestions": ("mrqa_naturalquestions", "NaturalQuestions.jsonl.gz"),
}


class DatasetLoader:
    def __init__(self, datasets_config: dict):
        self.cfg = datasets_config

    def load(self) -> dict:
        """Return {dataset_name: list[{question, context, answer}]}.

        Supports both legacy flat MRQA flags and split-aware MRQA configs:
        datasets.mrqa_data_dir can point either to /path/to/data/MRQA/dev for legacy flags
        or to the MRQA root containing dev/ and dev_ood/ for split-aware eval.
        """
        result = {}
        max_samples = self.cfg.get("max_samples")

        if self.cfg.get("squad"):
            result["squad"] = self._load_squad(max_samples)

        result.update(self._load_split_mrqa(max_samples))
        result.update(self._load_legacy_mrqa(max_samples, existing=set(result)))
        return result

    def _mrqa_root(self) -> Path:
        mrqa_dir = self.cfg.get("mrqa_data_dir")
        if not mrqa_dir:
            raise ValueError("MRQA datasets are enabled but 'mrqa_data_dir' is not set in config.")
        return Path(mrqa_dir)

    @staticmethod
    def _enabled_items(section: dict | None, mapping: dict[str, str]):
        if not section:
            return []
        unknown = sorted(set(section) - set(mapping))
        if unknown:
            raise ValueError(
                "Unknown MRQA dataset key(s): "
                f"{unknown}. Valid keys: {sorted(mapping)}"
            )
        return [(key, mapping[key]) for key, enabled in section.items() if enabled]

    def _resolve_split_dir(self, root: Path, split: str) -> Path:
        split_dir = root / split
        if split_dir.is_dir():
            return split_dir
        if split == "dev" and root.is_dir():
            return root
        raise FileNotFoundError(
            f"Expected MRQA split directory {split_dir}. "
            "Set datasets.mrqa_data_dir to the MRQA root containing dev/ and dev_ood/."
        )

    def _load_split_mrqa(self, max_samples) -> dict:
        result = {}
        in_domain = self.cfg.get("in_domain") or {}
        out_of_domain = self.cfg.get("out_of_domain") or {}
        if not in_domain and not out_of_domain:
            return result

        root = self._mrqa_root()
        if in_domain:
            dev_dir = self._resolve_split_dir(root, "dev")
            for key, filename in self._enabled_items(in_domain, MRQA_EVAL_IN_DOMAIN_FILES):
                result[f"mrqa_{key}"] = self._load_mrqa_file(dev_dir / filename, max_samples)
        if out_of_domain:
            ood_dir = self._resolve_split_dir(root, "dev_ood")
            for key, filename in self._enabled_items(out_of_domain, MRQA_EVAL_OUT_OF_DOMAIN_FILES):
                result[f"mrqa_ood_{key}"] = self._load_mrqa_file(ood_dir / filename, max_samples)
        return result

    def _load_legacy_mrqa(self, max_samples, existing: set[str]) -> dict:
        result = {}
        enabled = [item for item in MRQA_EVAL_LEGACY_FLAGS.items() if self.cfg.get(item[0])]
        if not enabled:
            return result
        root = self._mrqa_root()
        for flag, (dataset_name, filename) in enabled:
            if dataset_name in existing:
                continue
            path = root / filename
            if not path.is_file():
                dev_path = root / "dev" / filename
                path = dev_path if dev_path.is_file() else path
            if not path.is_file():
                raise FileNotFoundError(
                    f"Dataset flag '{flag}' expected file {path}. "
                    "For split-aware evaluation use datasets.in_domain/out_of_domain."
                )
            result[dataset_name] = self._load_mrqa_file(path, max_samples)
        return result

    def _load_squad(self, max_samples) -> list:
        from datasets import load_dataset
        ds = load_dataset("rajpurkar/squad", split="validation")
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))
        return [
            {
                "question": s["question"],
                "context": s["context"],
                "answer": s["answers"]["text"][0] if s["answers"]["text"] else "",
                "answers": s["answers"]["text"],
            }
            for s in ds
        ]

    def _load_mrqa_file(self, path: str | Path, max_samples) -> list:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"MRQA file not found: {path}")
        from datasets import Dataset as HFDataset
        ds = HFDataset.from_generator(mrqa_generator, gen_kwargs={"file_path": str(path)})
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))
        return [
            {
                "question": s["question"],
                "context": s["context"],
                "answer": s["answer"],
                "answers": s.get("answers", [s["answer"]]),
            }
            for s in ds
        ]

### LLM Judge ###

class LLMJudge:
    def __init__(
        self,
        judge_cfg: dict | None = None,
        *,
        model: str | None = None,
        base_url: str | None = None,
        prompt_template: str | None = None,
        max_concurrent: int = 32,
        max_tokens: int = 128,
        temperature: float = 1.0,
    ):
        """Accept either a config dict (production use) or flat keyword args (tests/programmatic)."""
        if judge_cfg is not None:
            # Dict-based interface (used by main() and eval YAML configs)
            base_url = judge_cfg["base_url"]
            model = judge_cfg["model"]
            prompt_template = judge_cfg["prompt_template"]
            max_concurrent = judge_cfg.get("max_concurrent", max_concurrent)
            max_tokens = judge_cfg.get("max_tokens", max_tokens)
            temperature = judge_cfg.get("temperature", temperature)

        if model is None or base_url is None or prompt_template is None:
            raise ValueError("model, base_url, and prompt_template are required")

        self.client = AsyncOpenAI(base_url=base_url, api_key="EMPTY")
        self.model = model
        self.prompt_template = prompt_template
        self.max_concurrent = max_concurrent
        self.max_tokens = max_tokens
        self.temperature = temperature

    async def _score_one(self, question: str, gold: str, prediction: str,
                         semaphore: asyncio.Semaphore) -> int:
        """Returns 0/1 or -1 for invalid/error responses."""
        prompt = self.prompt_template.format(
            question=question, gold=gold, prediction=prediction,
            context="", references=gold, predicted=prediction,
        )
        async with semaphore:
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                text = response.choices[0].message.content or ""
                return _parse_score(text)
            except Exception as e:
                logging.warning("Judge API error: %s", e)
                return -1

    async def score_batch(self, samples: list, progress=None, task_id=None) -> list:
        semaphore = asyncio.Semaphore(self.max_concurrent)
        async def _score_and_update(*args):
            if len(args) == 3:
                q, gold, pred = args
            else:
                q, _ctx, gold, pred = args
            result = await self._score_one(q, gold, pred, semaphore)
            if progress is not None and task_id is not None:
                progress.update(task_id, advance=1)
            return result
        return await asyncio.gather(*[_score_and_update(*s) for s in samples])

    def score(self, samples: list, progress=None, task_id=None) -> list:
        """Synchronous entry point. Returns list of int scores (0/1/-1)."""
        return asyncio.run(self.score_batch(samples, progress, task_id))

    def health_check(self) -> bool:
        """Ping the vLLM /health endpoint. Returns True if reachable."""
        import urllib.request
        import urllib.parse
        try:
            parsed = urllib.parse.urlparse(str(self.client.base_url))
            health_url = urllib.parse.urlunparse(parsed._replace(path="/health", query="", fragment=""))
            urllib.request.urlopen(health_url, timeout=5)
            return True
        except Exception:
            return False

### Evaluation runner ###

def _make_progress(console: Console) -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("eta"),
        TimeRemainingColumn(),
        console=console,
    )

class EvalRunner:
    def __init__(self, raw_cfg: dict, runner, judge: LLMJudge | None = None):
        self.cfg = raw_cfg
        self.runner = runner
        self.judge = judge
        self.console = Console()
        evaluation_cfg = raw_cfg.get("evaluation", {})
        self.metric = evaluation_cfg.get("metric", "judge_accuracy")
        self.clean_qa_predictions = bool(evaluation_cfg.get("clean_predictions", False))
        self.conditions = evaluation_cfg.get(
            "conditions", ["teacher_context", "teacher_nocontext", "student"]
        )
        valid_conditions = {"teacher_context", "teacher_nocontext", "student"}
        unknown_conditions = sorted(set(self.conditions) - valid_conditions)
        if unknown_conditions:
            raise ValueError(
                f"Unknown evaluation condition(s): {unknown_conditions}. "
                f"Valid values: {sorted(valid_conditions)}"
            )

    def _metric_suffix(self) -> str:
        return "qa_f1" if self.metric == "qa_f1" else "judge_accuracy"

    @staticmethod
    def _safe_model_name(model_path: str) -> str:
        path = Path(str(model_path))
        if len(path.parts) >= 2 and path.name in {"final", "latest"}:
            name = "__".join(path.parts[-2:])
        else:
            name = path.name or str(model_path)
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "model"

    def _condition_model_name(self, condition: str) -> str:
        if condition.startswith("teacher"):
            return self.runner.config.get("teacher_model_name", self.runner.config["llm_model_name"])
        return self.runner.config["llm_model_name"]

    def _baseline_path(self, dataset_name: str, condition: str) -> str:
        model_safe = self._safe_model_name(self._condition_model_name(condition))
        return os.path.join(
            "evals", "baselines", model_safe,
            f"{dataset_name}_{condition}_{self._metric_suffix()}.json",
        )

    def _experiment_path(self, dataset_name: str) -> str:
        return os.path.join(
            "evals", "experiments", self.runner.run_name,
            f"{dataset_name}_student_{self._metric_suffix()}.json",
        )

    def _summary_path(self) -> str:
        return os.path.join(
            "evals", "experiments", self.runner.run_name,
            f"summary_{self._metric_suffix()}.json",
        )

    def _build_result(self, model: str, dataset: str, condition: str,
                      samples: list, preds: list, scores: list,
                      scored_preds: list | None = None,
                      em_scores: list | None = None) -> dict:
        scored_preds = scored_preds or preds
        value = compute_accuracy(scores)
        em_value = compute_accuracy(em_scores) if em_scores is not None else None
        metric_label = "qa_f1" if self.metric == "qa_f1" else "accuracy"
        result = {
            "model": model,
            "dataset": dataset,
            "condition": condition,
            "metric": metric_label,
            "score": value,
            # Backward-compatible field used by existing summaries. For qa_f1 this
            # is mean F1 in [0, 1]; multiply by 100 for LongBench-style display.
            "accuracy": value,
            "f1_percent": round(100 * value, 2) if self.metric == "qa_f1" else None,
            "exact_match": em_value,
            "em_percent": round(100 * em_value, 2) if em_value is not None else None,
            "n_samples": len(samples),
            "n_invalid": scores.count(-1),
            "samples": [],
        }
        for index, (sample, pred, scored_pred, score) in enumerate(
            zip(samples, preds, scored_preds, scores)
        ):
            item = {
                "question": sample["question"],
                "gold": sample["answer"],
                "golds": gold_answers(sample),
                "predicted": pred,
                "scored_prediction": scored_pred,
                "score": score,
            }
            if em_scores is not None:
                item["exact_match"] = em_scores[index]
            result["samples"].append(item)
        return result

    def _run_condition(self, dataset_name: str, samples: list, condition: str,
                       path: str, force: bool, batch_size: int,
                       progress: Progress) -> dict:
        label = _CONDITION_STYLE[condition]
        if results_exist(path) and not force:
            progress.console.print(f"  [yellow][{condition}] Loading from cache:[/yellow] {path}")
            return load_results(path)

        infer_task = progress.add_task(f"{label} [dim]inference[/dim]", total=len(samples))
        if condition == "teacher_context":
            preds = self.runner.run_teacher(
                samples, use_context=True, batch_size=batch_size,
                progress=progress, task_id=infer_task,
            )
        elif condition == "teacher_nocontext":
            preds = self.runner.run_teacher(
                samples, use_context=False, batch_size=batch_size,
                progress=progress, task_id=infer_task,
            )
        else:
            preds = self.runner.run_student(
                samples, batch_size=batch_size,
                progress=progress, task_id=infer_task,
            )

        scored_preds = preds
        em_scores = None
        if self.metric == "qa_f1":
            score_task = progress.add_task(f"{label} [dim]qa-f1/em [/dim]", total=len(samples))
            scores = []
            em_scores = []
            if self.clean_qa_predictions:
                scored_preds = [
                    clean_qa_prediction(pred, gold_answers(sample))
                    for sample, pred in zip(samples, preds)
                ]
            for sample, pred in zip(samples, scored_preds):
                golds = gold_answers(sample)
                scores.append(qa_f1_score_any(pred, golds))
                em_scores.append(exact_match_score_any(pred, golds))
                progress.update(score_task, advance=1)
            n_invalid = 0
        else:
            if self.judge is None:
                raise ValueError("judge_accuracy evaluation requires a configured LLMJudge.")
            judge_task = progress.add_task(f"{label} [dim]judging  [/dim]", total=len(samples))
            scores = self.judge.score(
                [(s["question"], s.get("context", ""), gold_answers(s), p) for s, p in zip(samples, preds)],
                progress=progress, task_id=judge_task,
            )

            n_invalid = scores.count(-1)
            if n_invalid:
                progress.console.print(
                    f"  [yellow]WARNING:[/yellow] {n_invalid} invalid judge responses excluded"
                )

        result = self._build_result(
            self._condition_model_name(condition), dataset_name, condition,
            samples, preds, scores, scored_preds=scored_preds, em_scores=em_scores,
        )
        save_results(result, path)
        if self.metric == "qa_f1":
            progress.console.print(
                f"  [green][{condition}] QA-F1: {result['f1_percent']:.2f} "
                f"EM: {result['em_percent']:.2f}[/green] — saved to {path}"
            )
        else:
            progress.console.print(
                f"  [green][{condition}] Accuracy: {result['accuracy']:.2f}[/green] — saved to {path}"
            )
        return result

    def run(self, datasets: dict) -> dict:
        force_baseline = self.cfg.get("force_recompute_baselines", False)
        force_experiment = self.cfg.get("force_recompute_experiment", False)
        batch_size = self.cfg.get("inference", {}).get("batch_size", 8)
        per_dataset = {}

        with _make_progress(self.console) as progress:
            for dataset_name, samples in datasets.items():
                progress.console.print(
                    f"\n[bold cyan]=== {dataset_name} ({len(samples)} samples) ===[/bold cyan]"
                )
                results = {}
                if "teacher_context" in self.conditions:
                    results["teacher_context"] = self._run_condition(
                        dataset_name, samples, "teacher_context",
                        self._baseline_path(dataset_name, "teacher_context"),
                        force_baseline, batch_size, progress,
                    )
                if "teacher_nocontext" in self.conditions:
                    results["teacher_nocontext"] = self._run_condition(
                        dataset_name, samples, "teacher_nocontext",
                        self._baseline_path(dataset_name, "teacher_nocontext"),
                        force_baseline, batch_size, progress,
                    )
                if "student" in self.conditions:
                    results["student"] = self._run_condition(
                        dataset_name, samples, "student",
                        self._experiment_path(dataset_name),
                        force_experiment, batch_size, progress,
                    )

                per_dataset[dataset_name] = {
                    condition: result["accuracy"]
                    for condition, result in results.items()
                }
                if self.metric == "qa_f1":
                    per_dataset[dataset_name].update(
                        {
                            f"{condition}_em": result.get("exact_match", 0.0)
                            for condition, result in results.items()
                        }
                    )

        macro_avg = compute_macro_average(per_dataset)
        summary = {
            "run_name": self.runner.run_name,
            "llm_model": self.runner.config["llm_model_name"],
            "teacher_model": self.runner.config.get("teacher_model_name", self.runner.config["llm_model_name"]),
            "conditions_evaluated": list(self.conditions),
            "metric": "qa_f1" if self.metric == "qa_f1" else "judge_accuracy",
            "scale": "0-100 when printed; 0-1 in per_dataset/macro_average",
            "datasets_evaluated": list(datasets.keys()),
            "per_dataset": per_dataset,
            "macro_average": macro_avg,
        }
        save_results(summary, self._summary_path())
        self.console.print(f"[green]\nSummary saved to {self._summary_path()}[/green]")
        return summary

    def _log_wandb(self, summary: dict) -> None:
        import wandb
        wandb_cfg = self.cfg.get("wandb", {})
        wandb.init(
            project=wandb_cfg.get("project", "boa"),
            entity=wandb_cfg.get("entity") or None,
            name=f"eval_{self.runner.run_name}",
            tags=wandb_cfg.get("tags", ["eval"]),
        )

        metrics = {}
        for ds, vals in summary["per_dataset"].items():
            for key, value in vals.items():
                suffix = "em" if key.endswith("_em") else "f1"
                metric_name = key[:-3] if key.endswith("_em") else key
                metrics[f"{ds}_{metric_name}_{suffix}"] = value

        avg = summary["macro_average"]
        for key, value in avg.items():
            suffix = "em" if key.endswith("_em") else "f1"
            metric_name = key[:-3] if key.endswith("_em") else key
            metrics[f"avg_{metric_name}_{suffix}"] = value
        wandb.log(metrics)

        for dataset_name in summary["datasets_evaluated"]:
            if "student" not in summary.get("conditions_evaluated", []):
                continue
            exp = load_results(self._experiment_path(dataset_name))
            table = wandb.Table(columns=["question", "gold", "student", "student_f1", "student_em"])
            for sample in exp["samples"]:
                table.add_data(
                    sample["question"], sample["gold"],
                    sample["predicted"], sample["score"], sample.get("exact_match"),
                )
            wandb.log({f"{dataset_name}_student_table": table})

        wandb.finish()
        print(Fore.GREEN + f"W&B run logged: eval_{self.runner.run_name}")

    def _print_summary(self, summary: dict) -> None:
        table = Table(
            title=f"Evaluation Results — {self.runner.run_name}",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Dataset", style="bold")
        scale = 100 if summary.get("metric") == "qa_f1" else 1
        conditions = summary.get("conditions_evaluated", ["teacher_context", "teacher_nocontext", "student"])
        labels = {
            "teacher_context": "Teacher+Ctx",
            "teacher_nocontext": "Teacher-NoCtx",
            "student": "Student",
        }
        for condition in conditions:
            suffix = " F1/EM" if summary.get("metric") == "qa_f1" else " Accuracy"
            table.add_column(labels.get(condition, condition) + suffix, justify="right")

        def fmt_value(vals: dict, condition: str) -> str:
            if condition not in vals:
                return "-"
            if summary.get("metric") != "qa_f1":
                return f"{scale * vals[condition]:.2f}"
            return f"{scale * vals[condition]:.2f}/{scale * vals.get(condition + '_em', 0.0):.2f}"

        for ds, vals in summary["per_dataset"].items():
            table.add_row(ds, *[fmt_value(vals, condition) for condition in conditions])

        avg = summary["macro_average"]
        table.add_section()
        table.add_row(
            "[bold]Average[/bold]",
            *[f"[bold]{fmt_value(avg, condition)}[/bold]" for condition in conditions],
        )
        self.console.print(table)

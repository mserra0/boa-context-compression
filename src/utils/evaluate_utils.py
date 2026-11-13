import asyncio
import json
import logging
import os
import re

from colorama import Fore
from openai import AsyncOpenAI
from rich.console import Console
from rich.progress import (
    Progress, BarColumn, TextColumn, MofNCompleteColumn,
    TimeElapsedColumn, TimeRemainingColumn, SpinnerColumn,
)
from rich.table import Table

from ..dataset import mrqa_generator, MRQA_FILENAME
from .model_runner import ModelRunner

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

class DatasetLoader:
    def __init__(self, datasets_config: dict):
        self.cfg = datasets_config

    def load(self) -> dict:
        """Return {dataset_name: list[{question, context, answer}]}."""
        result = {}
        max_samples = self.cfg.get("max_samples")

        if self.cfg.get("squad"):
            result["squad"] = self._load_squad(max_samples)

        for key, filename in MRQA_FILENAME.items():
            if self.cfg.get(key):
                mrqa_dir = self.cfg.get("mrqa_data_dir")
                if not mrqa_dir:
                    raise ValueError(
                        f"Dataset '{key}' is enabled but 'mrqa_data_dir' is not set in config."
                    )
                path = os.path.join(mrqa_dir, filename)
                result[key] = self._load_mrqa_file(path, max_samples)

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
            }
            for s in ds
        ]

    def _load_mrqa_file(self, path: str, max_samples) -> list:
        from datasets import Dataset as HFDataset
        ds = HFDataset.from_generator(mrqa_generator, gen_kwargs={"file_path": path})
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))
        return [
            {"question": s["question"], "context": s["context"], "answer": s["answer"]}
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
    def __init__(self, raw_cfg: dict, runner: ModelRunner, judge: LLMJudge):
        self.cfg = raw_cfg
        self.runner = runner
        self.judge = judge
        self.console = Console()

    def _baseline_path(self, dataset_name: str, condition: str) -> str:
        return os.path.join(
            "evals", "baselines", self.runner.llm_name_safe,
            f"{dataset_name}_{condition}.json",
        )

    def _experiment_path(self, dataset_name: str) -> str:
        return os.path.join(
            "evals", "experiments", self.runner.run_name,
            f"{dataset_name}_student.json",
        )

    def _summary_path(self) -> str:
        return os.path.join("evals", "experiments", self.runner.run_name, "summary.json")

    def _build_result(self, model: str, dataset: str, condition: str,
                      samples: list, preds: list, scores: list) -> dict:
        accuracy = compute_accuracy(scores)
        return {
            "model": model,
            "dataset": dataset,
            "condition": condition,
            "accuracy": accuracy,
            "n_samples": len(samples),
            "n_invalid": scores.count(-1),
            "samples": [
                {"question": s["question"], "gold": s["answer"], "predicted": p, "score": sc}
                for s, p, sc in zip(samples, preds, scores)
            ],
        }

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

        judge_task = progress.add_task(f"{label} [dim]judging  [/dim]", total=len(samples))
        scores = self.judge.score(
            [(s["question"], s.get("context", ""), s["answer"], p) for s, p in zip(samples, preds)],
            progress=progress, task_id=judge_task,
        )

        n_invalid = scores.count(-1)
        if n_invalid:
            progress.console.print(
                f"  [yellow]WARNING:[/yellow] {n_invalid} invalid judge responses excluded"
            )

        result = self._build_result(
            self.runner.config["llm_model_name"], dataset_name, condition,
            samples, preds, scores,
        )
        save_results(result, path)
        progress.console.print(
            f"  [green][{condition}] Accuracy: {result['accuracy']:.3f}[/green] — saved to {path}"
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
                tc = self._run_condition(
                    dataset_name, samples, "teacher_context",
                    self._baseline_path(dataset_name, "teacher_context"),
                    force_baseline, batch_size, progress,
                )
                tnc = self._run_condition(
                    dataset_name, samples, "teacher_nocontext",
                    self._baseline_path(dataset_name, "teacher_nocontext"),
                    force_baseline, batch_size, progress,
                )
                student = self._run_condition(
                    dataset_name, samples, "student",
                    self._experiment_path(dataset_name),
                    force_experiment, batch_size, progress,
                )

                per_dataset[dataset_name] = {
                    "teacher_context": tc["accuracy"],
                    "teacher_nocontext": tnc["accuracy"],
                    "student": student["accuracy"],
                }

        macro_avg = compute_macro_average(per_dataset)
        summary = {
            "run_name": self.runner.run_name,
            "llm_model": self.runner.config["llm_model_name"],
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
            metrics[f"{ds}_teacher_context_acc"] = vals["teacher_context"]
            metrics[f"{ds}_teacher_nocontext_acc"] = vals["teacher_nocontext"]
            metrics[f"{ds}_student_acc"] = vals["student"]

        avg = summary["macro_average"]
        metrics["avg_teacher_context_acc"] = avg["teacher_context"]
        metrics["avg_teacher_nocontext_acc"] = avg["teacher_nocontext"]
        metrics["avg_student_acc"] = avg["student"]
        wandb.log(metrics)

        for dataset_name in summary["datasets_evaluated"]:
            tc = load_results(self._baseline_path(dataset_name, "teacher_context"))
            tnc = load_results(self._baseline_path(dataset_name, "teacher_nocontext"))
            exp = load_results(self._experiment_path(dataset_name))

            table = wandb.Table(columns=[
                "question", "gold",
                "teacher_context", "tc_score",
                "teacher_nocontext", "tnc_score",
                "student", "student_score",
            ])
            n_rows = min(len(tc["samples"]), len(tnc["samples"]), len(exp["samples"]))
            if n_rows < max(len(tc["samples"]), len(tnc["samples"]), len(exp["samples"])):
                logging.warning(
                    "Sample count mismatch for %s: tc=%d tnc=%d exp=%d — logging %d rows",
                    dataset_name, len(tc["samples"]), len(tnc["samples"]), len(exp["samples"]), n_rows,
                )
            for i in range(n_rows):
                tc_s = tc["samples"][i]
                table.add_data(
                    tc_s["question"], tc_s["gold"],
                    tc_s["predicted"], tc_s["score"],
                    tnc["samples"][i]["predicted"], tnc["samples"][i]["score"],
                    exp["samples"][i]["predicted"], exp["samples"][i]["score"],
                )
            wandb.log({f"{dataset_name}_table": table})

        wandb.finish()
        print(Fore.GREEN + f"W&B run logged: eval_{self.runner.run_name}")

    def _print_summary(self, summary: dict) -> None:
        table = Table(
            title=f"Evaluation Results — {self.runner.run_name}",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Dataset", style="bold")
        table.add_column("Teacher+Context", justify="right")
        table.add_column("Teacher+NoCtx", justify="right")
        table.add_column("Student", justify="right")

        for ds, vals in summary["per_dataset"].items():
            table.add_row(
                ds,
                f"{vals['teacher_context']:.3f}",
                f"{vals['teacher_nocontext']:.3f}",
                f"{vals['student']:.3f}",
            )

        avg = summary["macro_average"]
        table.add_section()
        table.add_row(
            "[bold]Average[/bold]",
            f"[bold]{avg['teacher_context']:.3f}[/bold]",
            f"[bold]{avg['teacher_nocontext']:.3f}[/bold]",
            f"[bold]{avg['student']:.3f}[/bold]",
        )
        self.console.print(table)

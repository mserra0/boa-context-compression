"""Evaluation orchestration: dataset x condition x scorer -> metrics.

Inference is delegated entirely to ``ModelRunner`` (the training-aligned
generation path) so evaluation reproduces exactly what the model saw during
training — no separate inference implementation, no train/eval gap.

The runner is scorer-agnostic: it works with any object exposing
``metric_keys`` and ``score(samples, preds) -> (per_sample, aggregate)``
(see :mod:`src.evaluation.scorers`).
"""

import os

from colorama import Fore
from rich.console import Console
from rich.table import Table

from ..utils.evaluate_utils import (
    _make_progress,
    load_results,
    results_exist,
    save_results,
)

_CONDITIONS = ("student", "teacher_context", "teacher_nocontext")


def _macro(dataset_metrics: dict, metric_keys) -> dict:
    """Mean of each metric key across the given datasets."""
    names = list(dataset_metrics)
    if not names:
        return {k: 0.0 for k in metric_keys}
    return {
        k: sum(dataset_metrics[n][k] for n in names) / len(names)
        for k in metric_keys
    }


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _macro_scalar(per_dataset: dict) -> float:
    """Mean of a single scalar (e.g. compression ratio) across datasets."""
    return _mean(per_dataset.values())


def build_summary(results: dict, metric_keys) -> dict:
    """Macro-average per group (in-domain / OOD) and overall across all datasets.

    ``results`` = ``{"in_domain": {ds: {metric: val}}, "out_of_domain": {...}}``.
    Overall is the macro average over ALL datasets, not the mean of group means
    (groups can have different dataset counts).
    """
    in_domain = results.get("in_domain", {})
    ood = results.get("out_of_domain", {})
    all_datasets = {**in_domain, **ood}
    return {
        "in_domain_avg": _macro(in_domain, metric_keys),
        "out_of_domain_avg": _macro(ood, metric_keys),
        "overall_avg": _macro(all_datasets, metric_keys),
    }


class EvalRunner:
    def __init__(self, raw_cfg: dict, model_runner, scorer):
        self.cfg = raw_cfg
        self.runner = model_runner
        self.scorer = scorer
        self.metric_keys = scorer.metric_keys
        self.scorer_tag = raw_cfg.get("scorer", "f1_em")
        self.console = Console()

        conditions_cfg = raw_cfg.get("conditions", {})
        self.conditions = [c for c in _CONDITIONS if conditions_cfg.get(c, c == "student")]

    def _cache_path(self, group: str, dataset_name: str, condition: str) -> str:
        fname = f"{group}_{dataset_name}_{condition}_{self.scorer_tag}.json"
        if condition == "student":
            return os.path.join("evals", "experiments", self.runner.run_name, fname)
        return os.path.join("evals", "baselines", self.runner.llm_name_safe, fname)

    def _summary_path(self) -> str:
        return os.path.join(
            "evals", "experiments", self.runner.run_name,
            f"summary_{self.scorer_tag}.json",
        )

    def _infer(self, condition: str, samples, batch_size, progress, task_id):
        """Returns (preds, compression_ratios). compression_ratios is None for
        the teacher conditions — only the student compresses its input."""
        if condition == "teacher_context":
            preds = self.runner.run_teacher(
                samples, use_context=True, batch_size=batch_size,
                progress=progress, task_id=task_id,
            )
            return preds, None
        if condition == "teacher_nocontext":
            preds = self.runner.run_teacher(
                samples, use_context=False, batch_size=batch_size,
                progress=progress, task_id=task_id,
            )
            return preds, None
        return self.runner.run_student(
            samples, batch_size=batch_size, progress=progress, task_id=task_id,
        )

    def _run_condition(self, group, dataset_name, samples, condition,
                       force, batch_size, progress) -> dict:
        path = self._cache_path(group, dataset_name, condition)
        if results_exist(path) and not force:
            progress.console.print(f"  [yellow][{condition}] cache:[/yellow] {path}")
            return load_results(path)

        infer_task = progress.add_task(
            f"[magenta]{condition}[/magenta] [dim]infer[/dim]", total=len(samples)
        )
        preds, compression_ratios = self._infer(condition, samples, batch_size, progress, infer_task)

        score_task = progress.add_task(
            f"[magenta]{condition}[/magenta] [dim]score[/dim]", total=len(samples)
        )
        per_sample, aggregate = self.scorer.score(
            samples, preds, progress=progress, task_id=score_task
        )

        ratios = compression_ratios if compression_ratios is not None else [None] * len(samples)
        result = {
            "dataset": dataset_name,
            "group": group,
            "condition": condition,
            "scorer": self.scorer_tag,
            "n_samples": len(samples),
            "metrics": aggregate,
            "samples": [
                {"qid": s.get("qid"), "question": s["question"], "gold": s["answers"],
                 "predicted": p, **ps,
                 **({"compression_ratio": cr} if cr is not None else {})}
                for s, p, ps, cr in zip(samples, preds, per_sample, ratios)
            ],
        }
        # Achieved compression varies per sample for ICAE (fixed n_mem slots
        # regardless of input length) unlike Mamba, where it tracks the
        # configured ratio exactly — report the mean actually achieved.
        if compression_ratios is not None:
            result["compression_ratio"] = _mean(compression_ratios)
        save_results(result, path)
        metric_str = "  ".join(f"{k}={aggregate[k]:.3f}" for k in self.metric_keys)
        if compression_ratios is not None:
            metric_str += f"  compression={result['compression_ratio']:.2f}x"
        progress.console.print(
            f"  [green][{condition}] {metric_str}[/green] -> {path}"
        )
        return result

    def run(self, groups: dict) -> dict:
        force_baseline = self.cfg.get("force_recompute_baselines", False)
        force_experiment = self.cfg.get("force_recompute_experiment", False)
        batch_size = self.cfg.get("inference", {}).get("batch_size", 8)

        # metrics keyed by condition -> group -> dataset
        results_by_condition = {}
        # compression ratio (student only) keyed the same way, kept separate
        # from metric_keys since it's an efficiency stat, not an accuracy score
        compression_by_condition = {}

        with _make_progress(self.console) as progress:
            for group in ("in_domain", "out_of_domain"):
                for dataset_name, samples in groups.get(group, {}).items():
                    progress.console.print(
                        f"\n[bold cyan]=== [{group}] {dataset_name} "
                        f"({len(samples)} samples) ===[/bold cyan]"
                    )
                    for condition in self.conditions:
                        force = force_experiment if condition == "student" else force_baseline
                        res = self._run_condition(
                            group, dataset_name, samples, condition,
                            force, batch_size, progress,
                        )
                        cond_res = results_by_condition.setdefault(
                            condition, {"in_domain": {}, "out_of_domain": {}}
                        )
                        cond_res[group][dataset_name] = res["metrics"]
                        if "compression_ratio" in res:
                            comp_res = compression_by_condition.setdefault(
                                condition, {"in_domain": {}, "out_of_domain": {}}
                            )
                            comp_res[group][dataset_name] = res["compression_ratio"]

        # Per-condition macro summaries so teacher-only (or any subset) runs are
        # reported, not just the student. Top-level keys mirror the "primary"
        # condition (student if run, else the first requested condition).
        per_condition = {}
        for cond, res in results_by_condition.items():
            entry = {**build_summary(res, self.metric_keys), "per_dataset": res}
            comp = compression_by_condition.get(cond)
            if comp:
                entry["compression_ratio"] = {
                    "in_domain_avg": _macro_scalar(comp.get("in_domain", {})),
                    "out_of_domain_avg": _macro_scalar(comp.get("out_of_domain", {})),
                    "overall_avg": _macro_scalar({**comp.get("in_domain", {}), **comp.get("out_of_domain", {})}),
                    "per_dataset": comp,
                }
            per_condition[cond] = entry

        primary = "student" if "student" in per_condition else (self.conditions[0] if self.conditions else None)
        primary_res = results_by_condition.get(primary, {"in_domain": {}, "out_of_domain": {}})

        summary = build_summary(primary_res, self.metric_keys)
        summary.update({
            "run_name": self.runner.run_name,
            "llm_model": self.runner.config["llm_model_name"],
            "scorer": self.scorer_tag,
            "conditions": self.conditions,
            "primary_condition": primary,
            "per_dataset": primary_res,
            "per_condition": per_condition,
        })
        if primary in per_condition and "compression_ratio" in per_condition[primary]:
            summary["compression_ratio"] = per_condition[primary]["compression_ratio"]
        save_results(summary, self._summary_path())
        self.console.print(f"[green]\nSummary saved to {self._summary_path()}[/green]")
        return summary

    def print_summary(self, summary: dict) -> None:
        # One table per condition so teacher baselines and the student are all shown.
        per_condition = summary.get("per_condition") or {
            summary.get("primary_condition", "student"): {
                "per_dataset": summary["per_dataset"],
                "in_domain_avg": summary["in_domain_avg"],
                "out_of_domain_avg": summary["out_of_domain_avg"],
                "overall_avg": summary["overall_avg"],
            }
        }
        for condition, cond in per_condition.items():
            comp = cond.get("compression_ratio")
            table = Table(
                title=f"MRQA {self.scorer_tag} [{condition}] — {self.runner.run_name}",
                show_header=True, header_style="bold cyan",
            )
            table.add_column("Group", style="dim")
            table.add_column("Dataset", style="bold")
            for k in self.metric_keys:
                table.add_column(k.upper(), justify="right")
            if comp:
                # Achieved compression, not the configured ratio: for ICAE this
                # varies per dataset since n_mem is fixed regardless of input length.
                table.add_column("COMPRESSION", justify="right")

            def _row(group_label, name, metrics, comp_val=None, bold=False):
                cells = [group_label, name]
                for k in self.metric_keys:
                    v = f"{metrics[k] * 100:.2f}"
                    cells.append(f"[bold]{v}[/bold]" if bold else v)
                if comp:
                    cv = f"{comp_val:.2f}x" if comp_val is not None else "-"
                    cells.append(f"[bold]{cv}[/bold]" if bold else cv)
                table.add_row(*cells)

            for group, label in (("in_domain", "in-dist"), ("out_of_domain", "ood")):
                for name, metrics in cond["per_dataset"].get(group, {}).items():
                    comp_val = comp["per_dataset"].get(group, {}).get(name) if comp else None
                    _row(label, name, metrics, comp_val)
            table.add_section()
            _row("in-dist", "[bold]avg[/bold]", cond["in_domain_avg"], comp["in_domain_avg"] if comp else None, bold=True)
            _row("ood", "[bold]avg[/bold]", cond["out_of_domain_avg"], comp["out_of_domain_avg"] if comp else None, bold=True)
            _row("all", "[bold]avg[/bold]", cond["overall_avg"], comp["overall_avg"] if comp else None, bold=True)
            self.console.print(table)

    def log_wandb(self, summary: dict) -> None:
        import wandb

        wandb_cfg = self.cfg.get("wandb", {})
        wandb.init(
            project=wandb_cfg.get("project", "boa"),
            entity=wandb_cfg.get("entity") or None,
            name=f"eval_{self.scorer_tag}_{self.runner.run_name}",
            tags=wandb_cfg.get("tags", ["eval", self.scorer_tag]),
        )
        metrics = {}
        for condition, cond in summary.get("per_condition", {}).items():
            for group in ("in_domain", "out_of_domain"):
                for ds, vals in cond["per_dataset"].get(group, {}).items():
                    for k in self.metric_keys:
                        metrics[f"{condition}/{group}/{ds}_{k}"] = vals[k]
            for avg_key in ("in_domain_avg", "out_of_domain_avg", "overall_avg"):
                for k in self.metric_keys:
                    metrics[f"{condition}/{avg_key}/{k}"] = cond[avg_key][k]

            comp = cond.get("compression_ratio")
            if comp:
                for group in ("in_domain", "out_of_domain"):
                    for ds, val in comp["per_dataset"].get(group, {}).items():
                        metrics[f"{condition}/{group}/{ds}_compression_ratio"] = val
                for avg_key in ("in_domain_avg", "out_of_domain_avg", "overall_avg"):
                    metrics[f"{condition}/{avg_key}/compression_ratio"] = comp[avg_key]
        wandb.log(metrics)
        wandb.finish()
        print(Fore.GREEN + f"W&B logged: eval_{self.scorer_tag}_{self.runner.run_name}")

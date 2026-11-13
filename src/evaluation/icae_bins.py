"""Pool ICAE eval results across checkpoints by ACHIEVED compression ratio.

Different ICAE checkpoints ("4x", "8x", "16x") only differ in n_mem (the fixed
memory-slot count); the ratio any single sample actually gets compressed to is
real_len / n_mem, which varies per sample and routinely lands far from the
checkpoint's nominal name (see per-sample "compression_ratio" already recorded
by ``src.evaluation.runner`` in evals/experiments/<run>/*.json). This module
does NOT run any new inference -- it re-reads those already-cached per-sample
results from however many checkpoints you've evaluated, buckets every
(checkpoint, sample) pair into whichever ratio bin it actually achieved, and
reports pooled metrics per bin. Frozen-LLM teacher predictions (which don't
depend on the compressor at all) are pulled from the same cache the main eval
already writes to (evals/baselines/<llm_name_safe>/...) and deduplicated per
question so a question landing in a bin under multiple checkpoints doesn't
over-weight the teacher's score.

Usage:
    uv run python -m src.evaluation.icae_bins \
        --checkpoints /path/to/4x_ckpt /path/to/8x_ckpt /path/to/16x_ckpt \
        --label ICAE_actual_ratio

Each run overwrites the bin output files from scratch based on whatever
checkpoint eval caches currently exist on disk -- rerun this after evaluating
more checkpoints (or more datasets) rather than expecting incremental state.

Bin width: each bin is [nominal * (1 - half_width), nominal * (1 + half_width))
for nominal in (4, 8, 16) -- default half_width=0.25 gives [3,5)/[6,10)/[12,20).
Pass --bin-half-width 0.5 for a wider band, e.g. [2,6)/[4,12)/[8,24).
"""

import argparse
import os
from pathlib import Path

from colorama import Fore, init as colorama_init
from rich.console import Console
from rich.table import Table

from .datasets import MRQA_INDOMAIN, MRQA_OOD
from ..utils.config import load_config
from ..utils.evaluate_utils import load_results, results_exist, save_results

colorama_init(autoreset=True)

_ALL_DATASETS = [("in_domain", name) for name in MRQA_INDOMAIN] + \
                [("out_of_domain", name) for name in MRQA_OOD]

# Half-open [lo, hi) — a sample with ratio exactly at an upper edge falls in
# the NEXT bin, not this one. Each bin is a band straddling its nominal ratio:
# lo = nominal * (1 - half_width), hi = nominal * (1 + half_width). Default
# half_width=0.25 gives [3,5)/[6,10)/[12,20).
_NOMINAL_RATIOS = [("4x", 4.0), ("8x", 8.0), ("16x", 16.0)]


def make_bins(half_width: float = 0.1) -> list[tuple[str, float, float]]:
    if not 0.0 < half_width < 1.0:
        raise ValueError(f"half_width must be in (0, 1), got {half_width!r}")
    return [(name, nominal * (1 - half_width), nominal * (1 + half_width))
            for name, nominal in _NOMINAL_RATIOS]


# Default bins, used by assign_bin()'s default and anything importing BINS
# directly. run() always builds its own from --bin-half-width and threads it
# through explicitly, so changing this constant is NOT required to change
# behavior via the CLI.
BINS = make_bins()


def assign_bin(ratio: float, bins: list[tuple[str, float, float]] = BINS) -> str | None:
    if ratio is None:
        return None
    for name, lo, hi in bins:
        if lo <= ratio < hi:
            return name
    return None


def _sample_key(sample: dict, idx: int) -> str:
    qid = sample.get("qid")
    return f"qid:{qid}" if qid else f"idx:{idx}"


def _student_cache_path(run_name: str, group: str, dataset: str, scorer: str) -> str:
    return os.path.join("evals", "experiments", run_name, f"{group}_{dataset}_student_{scorer}.json")


def _teacher_cache_path(llm_name_safe: str, group: str, dataset: str, condition: str, scorer: str) -> str:
    return os.path.join("evals", "baselines", llm_name_safe, f"{group}_{dataset}_{condition}_{scorer}.json")


class _CheckpointInfo:
    def __init__(self, checkpoint_dir: str):
        exp_path = Path(checkpoint_dir)
        self.run_name = exp_path.resolve().name
        config = load_config(str(exp_path / "config.yaml"))
        if config.get("compressor_type") != "icae":
            raise ValueError(
                f"{checkpoint_dir} has compressor_type={config.get('compressor_type')!r}, "
                "expected 'icae' — bin pooling only makes sense across ICAE checkpoints "
                "(the ratio-vs-n_mem relationship this script relies on is ICAE-specific)."
            )
        self.llm_model_name = config["llm_model_name"]
        self.llm_name_safe = self.llm_model_name.split("/")[-1]


def _load_student_samples(ckpt: _CheckpointInfo, group: str, dataset: str, scorer: str):
    path = _student_cache_path(ckpt.run_name, group, dataset, scorer)
    if not results_exist(path):
        return None
    return load_results(path)["samples"]


def _load_teacher_samples(llm_name_safe: str, group: str, dataset: str, condition: str, scorer: str):
    path = _teacher_cache_path(llm_name_safe, group, dataset, condition, scorer)
    if not results_exist(path):
        return None
    return load_results(path)["samples"]


def _pool_dataset(ckpts: list[_CheckpointInfo], group: str, dataset: str, scorer: str,
                  bins: list[tuple[str, float, float]], console: Console):
    """Returns (student_by_bin, teacher_by_bin, out_of_range) for one dataset.

    student_by_bin[bin] = list of {run_name, qid, question, gold, predicted, em, f1, compression_ratio}
    teacher_by_bin[bin][condition] = list of {question, gold, predicted, em, f1} (deduped by key)
    out_of_range[run_name] = {"below_min": n, "in_gap": n, "above_max": n, "total": n}
    """
    student_by_bin = {b: [] for b, _, _ in bins}
    # bin -> set of keys placed there by ANY checkpoint. The same question can
    # legitimately land in different bins under different checkpoints (same
    # document, different n_mem -> different achieved ratio) -- this must stay
    # a set-per-bin, not a single key->bin map, or a question claimed by one
    # checkpoint's bin silently shadows its (equally valid) placement in a
    # different bin under another checkpoint.
    bin_to_keys: dict[str, set] = {b: set() for b, _, _ in bins}
    out_of_range = {}

    min_lo = bins[0][1]
    max_hi = bins[-1][2]
    llm_name_safes_seen = set()

    for ckpt in ckpts:
        samples = _load_student_samples(ckpt, group, dataset, scorer)
        if samples is None:
            console.print(
                f"  [yellow]skip[/yellow] {ckpt.run_name}: no cached student result for "
                f"{group}/{dataset} (evaluate this checkpoint on this dataset first)"
            )
            continue
        llm_name_safes_seen.add(ckpt.llm_name_safe)

        below_min = in_gap = above_max = 0
        for idx, s in enumerate(samples):
            ratio = s.get("compression_ratio")
            key = _sample_key(s, idx)
            b = assign_bin(ratio, bins)
            if b is None:
                # Non-adjacent bins (e.g. [3,5)/[6,10)) leave gaps between
                # them, so a missed bin isn't just "too low" or "too high"
                # anymore -- track all three distinctly.
                if ratio is None:
                    pass
                elif ratio < min_lo:
                    below_min += 1
                elif ratio >= max_hi:
                    above_max += 1
                else:
                    in_gap += 1
                continue
            student_by_bin[b].append({
                "run_name": ckpt.run_name, "key": key,
                "qid": s.get("qid"), "question": s["question"], "gold": s["gold"],
                "predicted": s["predicted"], "em": s["em"], "f1": s["f1"],
                "compression_ratio": ratio,
            })
            # A set, not an overwrite: dedupes the same question landing in
            # this SAME bin under multiple checkpoints (so the teacher isn't
            # double-counted there) without blocking it from ALSO appearing
            # in a different bin under a different checkpoint.
            bin_to_keys[b].add(key)

        out_of_range[ckpt.run_name] = {
            "below_min": below_min, "in_gap": in_gap, "above_max": above_max, "total": len(samples),
        }

    # Pull teacher predictions for exactly the unique questions that landed in
    # each bin (a question can appear in multiple bins if different
    # checkpoints achieved different ratios on it), indexed the same way (qid
    # if available, else position — position is only trustworthy if the
    # teacher cache has the same sample count as the student caches it's
    # being paired against).
    teacher_by_bin = {b: {"teacher_context": [], "teacher_nocontext": []} for b, _, _ in bins}
    if len(llm_name_safes_seen) > 1:
        console.print(
            f"  [yellow]WARNING[/yellow] {group}/{dataset}: checkpoints span multiple base "
            f"LLMs {sorted(llm_name_safes_seen)} — teacher pairing uses each checkpoint's own "
            "baseline cache, so cross-checkpoint teacher dedup may mix answers from different LLMs."
        )

    for llm_name_safe in llm_name_safes_seen:
        for condition in ("teacher_context", "teacher_nocontext"):
            t_samples = _load_teacher_samples(llm_name_safe, group, dataset, condition, scorer)
            if t_samples is None:
                console.print(
                    f"  [yellow]skip[/yellow] teacher/{llm_name_safe}: no cached {condition} "
                    f"result for {group}/{dataset}"
                )
                continue
            # Build key -> sample for this teacher cache. Only fall back to
            # positional keys if the teacher cache is the same length as at
            # least one contributing student cache (checked implicitly: if a
            # qid-based key never matches, we skip it below rather than guess).
            by_key = {}
            for idx, s in enumerate(t_samples):
                by_key[_sample_key(s, idx)] = s

            for b, _, _ in bins:
                for key in bin_to_keys[b]:
                    t_s = by_key.get(key)
                    if t_s is None:
                        continue
                    teacher_by_bin[b][condition].append({
                        "question": t_s["question"], "gold": t_s["gold"],
                        "predicted": t_s["predicted"], "em": t_s["em"], "f1": t_s["f1"],
                    })

    return student_by_bin, teacher_by_bin, out_of_range


def _mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


def _metrics(entries: list) -> dict:
    return {"em": _mean([e["em"] for e in entries]), "f1": _mean([e["f1"] for e in entries])}


def run(checkpoint_dirs: list[str], label: str, scorer: str = "f1_em", half_width: float = 0.25) -> dict:
    console = Console()
    ckpts = [_CheckpointInfo(d) for d in checkpoint_dirs]
    bins = make_bins(half_width)
    console.print(
        f"[bold cyan]Pooling {len(ckpts)} ICAE checkpoint(s) by achieved compression ratio:[/bold cyan] "
        + ", ".join(c.run_name for c in ckpts)
    )
    console.print("[bold cyan]Bins:[/bold cyan] " + ", ".join(f"{n} [{lo:g},{hi:g})" for n, lo, hi in bins))

    out_root = os.path.join("evals", "experiments_binned", label)

    # bin -> group -> dataset -> {student: metrics+n, teacher_context: ..., teacher_nocontext: ...}
    per_bin_per_dataset = {b: {"in_domain": {}, "out_of_domain": {}} for b, _, _ in bins}
    out_of_range_all = {}  # run_name -> group -> dataset -> counts

    for group, dataset in _ALL_DATASETS:
        console.print(f"\n[bold]=== [{group}] {dataset} ===[/bold]")
        student_by_bin, teacher_by_bin, out_of_range = _pool_dataset(ckpts, group, dataset, scorer, bins, console)

        for run_name, counts in out_of_range.items():
            out_of_range_all.setdefault(run_name, {"in_domain": {}, "out_of_domain": {}})
            out_of_range_all[run_name][group][dataset] = counts

        for bin_name, _, _ in bins:
            s_entries = student_by_bin[bin_name]
            tc_entries = teacher_by_bin[bin_name]["teacher_context"]
            tnc_entries = teacher_by_bin[bin_name]["teacher_nocontext"]

            record = {
                "bin": bin_name, "group": group, "dataset": dataset,
                "student": {
                    "n_samples": len(s_entries), "metrics": _metrics(s_entries),
                    "contributing_runs": sorted(set(e["run_name"] for e in s_entries)),
                    "samples": s_entries,
                },
                "teacher_context": {"n_samples": len(tc_entries), "metrics": _metrics(tc_entries), "samples": tc_entries},
                "teacher_nocontext": {"n_samples": len(tnc_entries), "metrics": _metrics(tnc_entries), "samples": tnc_entries},
            }
            save_results(record, os.path.join(out_root, f"{group}_{dataset}_{bin_name}.json"))
            per_bin_per_dataset[bin_name][group][dataset] = {
                "student": record["student"]["metrics"] | {"n_samples": record["student"]["n_samples"]},
                "teacher_context": record["teacher_context"]["metrics"] | {"n_samples": record["teacher_context"]["n_samples"]},
                "teacher_nocontext": record["teacher_nocontext"]["metrics"] | {"n_samples": record["teacher_nocontext"]["n_samples"]},
            }

            if s_entries:
                console.print(
                    f"  [{bin_name} bin] n={len(s_entries):5d}  "
                    f"em={record['student']['metrics']['em']*100:5.2f}  f1={record['student']['metrics']['f1']*100:5.2f}  "
                    f"(from {', '.join(record['student']['contributing_runs'])})"
                )

    summary = {"label": label, "checkpoints": [c.run_name for c in ckpts], "bins": {}}
    for bin_name, lo, hi in bins:
        bin_summary = {"range": [lo, hi], "in_domain": {}, "out_of_domain": {}, "overall": {}}
        for condition in ("student", "teacher_context", "teacher_nocontext"):
            for group in ("in_domain", "out_of_domain"):
                ds_metrics = {
                    ds: per_bin_per_dataset[bin_name][group][ds][condition]
                    for ds in per_bin_per_dataset[bin_name][group]
                    if per_bin_per_dataset[bin_name][group][ds][condition]["n_samples"] > 0
                }
                bin_summary[group][condition] = {
                    "per_dataset": ds_metrics,
                    "macro_avg": {
                        "em": _mean([m["em"] for m in ds_metrics.values()]),
                        "f1": _mean([m["f1"] for m in ds_metrics.values()]),
                    } if ds_metrics else {"em": 0.0, "f1": 0.0},
                    "n_datasets_covered": len(ds_metrics),
                }
            all_ds = {**bin_summary["in_domain"][condition]["per_dataset"],
                      **bin_summary["out_of_domain"][condition]["per_dataset"]}
            bin_summary["overall"][condition] = {
                "macro_avg": {
                    "em": _mean([m["em"] for m in all_ds.values()]),
                    "f1": _mean([m["f1"] for m in all_ds.values()]),
                } if all_ds else {"em": 0.0, "f1": 0.0},
                "n_datasets_covered": len(all_ds),
            }
        summary["bins"][bin_name] = bin_summary

    summary["out_of_range"] = out_of_range_all
    save_results(summary, os.path.join(out_root, "summary_bins.json"))
    console.print(f"\n[green]Summary saved to {os.path.join(out_root, 'summary_bins.json')}[/green]")

    _print_summary(summary, console)
    return summary


def _print_summary(summary: dict, console: Console) -> None:
    table = Table(title=f"ICAE pooled-by-actual-ratio — {summary['label']}", show_header=True, header_style="bold cyan")
    table.add_column("Bin", style="bold")
    table.add_column("Scope")
    table.add_column("Student EM/F1", justify="right")
    table.add_column("Teacher+Ctx EM/F1", justify="right")
    table.add_column("Teacher-NoCtx EM/F1", justify="right")
    table.add_column("#datasets", justify="right")

    for bin_name, bin_data in summary["bins"].items():
        for scope in ("in_domain", "out_of_domain", "overall"):
            row_scope = {"in_domain": "in-dist", "out_of_domain": "ood", "overall": "all"}[scope]
            student = bin_data[scope]["student"]["macro_avg"]
            tc = bin_data[scope]["teacher_context"]["macro_avg"]
            tnc = bin_data[scope]["teacher_nocontext"]["macro_avg"]
            n_ds = bin_data[scope]["student"]["n_datasets_covered"]
            table.add_row(
                bin_name if scope == "in_domain" else "",
                row_scope,
                f"{student['em']*100:.2f} / {student['f1']*100:.2f}",
                f"{tc['em']*100:.2f} / {tc['f1']*100:.2f}",
                f"{tnc['em']*100:.2f} / {tnc['f1']*100:.2f}",
                str(n_ds),
            )
        table.add_section()
    console.print(table)

    console.print("\n[bold]Out-of-range samples (no bin), per checkpoint:[/bold]")
    for run_name, groups in summary["out_of_range"].items():
        below = sum(c["below_min"] for g in groups.values() for c in g.values())
        gap = sum(c["in_gap"] for g in groups.values() for c in g.values())
        above = sum(c["above_max"] for g in groups.values() for c in g.values())
        total = sum(c["total"] for g in groups.values() for c in g.values())
        console.print(
            f"  {run_name}: {below+gap+above}/{total} excluded "
            f"({below} below lowest bin, {gap} in a gap between bins, {above} at/above highest bin)"
        )


def main():
    parser = argparse.ArgumentParser(description="Pool ICAE MRQA eval results by achieved compression ratio")
    parser.add_argument("--checkpoints", nargs="+", required=True, help="ICAE checkpoint dirs (already evaluated)")
    parser.add_argument("--label", default="ICAE_actual_ratio", help="Output subdir under evals/experiments_binned/")
    parser.add_argument("--scorer", default="f1_em", help="Scorer tag matching the cached eval files (default f1_em)")
    parser.add_argument("--bin-half-width", type=float, default=0.25,
                        help="Bin = [nominal*(1-w), nominal*(1+w)) for nominal in 4/8/16. "
                             "Default 0.25 -> [3,5)/[6,10)/[12,20).")
    args = parser.parse_args()
    run(args.checkpoints, args.label, args.scorer, args.bin_half_width)


if __name__ == "__main__":
    main()

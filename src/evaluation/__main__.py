"""CLI: evaluate a Boa checkpoint with MRQA F1/EM (or judge).

    uv run python -m src.evaluation \
        --config configs/eval/config_eval_MRQA.yaml \
        --experiment-dir /path/to/checkpoints/<run_dir> \
        --device cuda

Scorer is chosen by ``scorer:`` in the config (``f1_em`` default, or
``llm_judge`` for the LLM-as-judge alternative, which needs a vLLM server).
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
from colorama import Fore, init as colorama_init

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.model_runner import ModelRunner
from src.evaluation.datasets import load_eval_datasets
from src.evaluation.scorers import LLMJudgeScorer, SpanMetricScorer
from src.evaluation.runner import EvalRunner

colorama_init(autoreset=True)


def _build_scorer(raw_cfg: dict):
    scorer_name = raw_cfg.get("scorer", "f1_em")
    if scorer_name == "f1_em":
        return SpanMetricScorer()
    if scorer_name == "llm_judge":
        judge_cfg = raw_cfg.get("judge")
        if not judge_cfg:
            print(Fore.RED + "ERROR: scorer 'llm_judge' requires a 'judge:' block in the config.")
            sys.exit(1)
        return LLMJudgeScorer(judge_cfg)
    print(Fore.RED + f"ERROR: unknown scorer '{scorer_name}' (use 'f1_em' or 'llm_judge').")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Boa with MRQA F1/EM")
    parser.add_argument("--config", required=True, help="Path to eval YAML config")
    parser.add_argument("--experiment-dir", required=True,
                        help="Checkpoint dir with best_model.pt and config.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-wandb", action="store_true", help="Skip W&B logging")
    args = parser.parse_args()

    with open(args.config) as f:
        raw_cfg = yaml.safe_load(f)

    exp_dir = Path(args.experiment_dir)
    if not (exp_dir / "best_model.pt").exists():
        print(Fore.RED + f"ERROR: best_model.pt not found in {exp_dir}")
        sys.exit(1)
    if not (exp_dir / "config.yaml").exists():
        print(Fore.RED + f"ERROR: config.yaml not found in {exp_dir}")
        sys.exit(1)

    ds_cfg = raw_cfg["datasets"]
    print(Fore.CYAN + "Loading MRQA dev sets...")
    groups = load_eval_datasets(
        root=ds_cfg["mrqa_data_dir"],
        in_domain=ds_cfg.get("in_domain", {}),
        out_of_domain=ds_cfg.get("out_of_domain", {}),
        max_samples=ds_cfg.get("max_samples"),
    )
    total = sum(len(v) for g in groups.values() for v in g.values())
    if total == 0:
        print(Fore.RED + "ERROR: no datasets enabled. Set at least one to true in the config.")
        sys.exit(1)
    for group, datasets in groups.items():
        for name, samples in datasets.items():
            print(Fore.GREEN + f"  [{group}] {name}: {len(samples)} samples")

    runner = ModelRunner(str(exp_dir), device=args.device)
    scorer = _build_scorer(raw_cfg)

    eval_runner = EvalRunner(raw_cfg, runner, scorer)
    summary = eval_runner.run(groups)
    eval_runner.print_summary(summary)
    if not args.no_wandb and raw_cfg.get("wandb", {}).get("enabled", True):
        eval_runner.log_wandb(summary)


if __name__ == "__main__":
    main()

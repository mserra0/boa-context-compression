"""Evaluate Activation Beacon on the repository's MRQA benchmark."""

from __future__ import annotations

import argparse
import os
import sys

import torch
import yaml
from colorama import Fore, init as colorama_init

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.baselines.activation_beacon import ActivationBeaconRunner
from src.utils.evaluate_utils import DatasetLoader, EvalRunner, LLMJudge

colorama_init(autoreset=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-config", required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    with open(args.eval_config) as handle:
        raw_cfg = yaml.safe_load(handle)

    metric = raw_cfg.get("evaluation", {}).get("metric", "judge_accuracy")
    judge = None
    if metric != "qa_f1":
        judge = LLMJudge(raw_cfg["judge"])
        if not judge.health_check():
            print(Fore.RED + f"ERROR: judge is unreachable at {raw_cfg['judge']['base_url']}")
            raise SystemExit(1)

    datasets_cfg = raw_cfg.get("datasets", {})
    datasets = DatasetLoader(datasets_cfg).load()
    if not datasets:
        enabled_flags = [
            key
            for key, value in datasets_cfg.items()
            if isinstance(value, bool) and value
        ]
        nested_enabled = {
            section: [key for key, value in (datasets_cfg.get(section) or {}).items() if value]
            for section in ("in_domain", "out_of_domain")
        }
        print(Fore.RED + "ERROR: no evaluation datasets are enabled")
        print(Fore.YELLOW + f"Config read: {args.eval_config}")
        print(Fore.YELLOW + f"datasets keys: {sorted(datasets_cfg)}")
        print(Fore.YELLOW + f"enabled flat flags: {enabled_flags}")
        print(Fore.YELLOW + f"enabled nested flags: {nested_enabled}")
        raise SystemExit(1)

    runner = ActivationBeaconRunner(raw_cfg["activation_beacon"], device=args.device)
    evaluator = EvalRunner(raw_cfg, runner, judge)
    summary = evaluator.run(datasets)
    evaluator._log_wandb(summary)
    evaluator._print_summary(summary)


if __name__ == "__main__":
    main()

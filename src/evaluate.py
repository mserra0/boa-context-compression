import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
from colorama import Fore, init as colorama_init

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.model_runner import ModelRunner
from src.utils.evaluate_utils import DatasetLoader, LLMJudge, EvalRunner

colorama_init(autoreset=True)

### Main entry point ###
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Boa with LLM-as-judge"
    )
    parser.add_argument("--eval-config", required=True, help="Path to eval YAML config")
    parser.add_argument(
        "--experiment-dir", required=True,
        help="Experiment directory containing best_model.pt and config.yaml",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    with open(args.eval_config) as f:
        raw_cfg = yaml.safe_load(f)

    judge_cfg = raw_cfg["judge"]
    datasets_cfg = raw_cfg["datasets"]

    exp_dir = Path(args.experiment_dir)
    if not (exp_dir / "best_model.pt").exists():
        print(Fore.RED + f"ERROR: best_model.pt not found in {exp_dir}")
        sys.exit(1)
    if not (exp_dir / "config.yaml").exists():
        print(Fore.RED + f"ERROR: config.yaml not found in {exp_dir}")
        sys.exit(1)

    print(Fore.CYAN + f"Checking vLLM server at {judge_cfg['base_url']} ...")
    judge_probe = LLMJudge(judge_cfg)
    if not judge_probe.health_check():
        print(Fore.RED + f"ERROR: vLLM server not reachable at {judge_cfg['base_url']}")
        import urllib.parse
        parsed = urllib.parse.urlparse(judge_cfg["base_url"])
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        print(Fore.RED + f"Start the server with: uv run vllm serve {judge_cfg['model']} --port {port}")
        sys.exit(1)
    print(Fore.GREEN + "vLLM server is reachable.")

    print(Fore.CYAN + "\nLoading datasets...")
    loader = DatasetLoader(datasets_cfg)
    datasets = loader.load()
    if not datasets:
        print(Fore.RED + "ERROR: No datasets enabled in config. Set at least one to true.")
        sys.exit(1)
    for name, samples in datasets.items():
        print(Fore.GREEN + f"  {name}: {len(samples)} samples")

    runner = ModelRunner(str(exp_dir), device=args.device)

    judge = LLMJudge(judge_cfg)

    eval_runner = EvalRunner(raw_cfg, runner, judge)
    summary = eval_runner.run(datasets)
    eval_runner._log_wandb(summary)
    eval_runner._print_summary(summary)


if __name__ == "__main__":
    main()

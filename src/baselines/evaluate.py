"""Standalone QA evaluation for plain HF causal-LM baselines (no compressor).

Evaluates a bare Hugging Face model (repo id or local checkpoint dir) on the same
MRQA dev sets and LLM-as-judge used by the compressor eval, under two conditions:

  - context:   gold context + question in the prompt (RAG-style upper bound)
  - nocontext: question only (closed-book floor)

Prompts are built with ``src.baselines.dataset.build_prompt_text`` so the format
is identical to FullFT SFT training — apples-to-apples for both the base model and
its fine-tuned variant.

Per-dataset results are cached under ``evals/baselines/<name>/<dataset>_<condition>.json``
(same JSON schema as the compressor teacher baselines) and skipped on re-run unless
``--force`` is passed.

Usage:
  CUDA_VISIBLE_DEVICES=0 uv run python -m src.baselines.evaluate \
      --eval-config configs/eval/config_eval.yaml \
      --model Qwen/Qwen3-1.7B

  CUDA_VISIBLE_DEVICES=0 uv run python -m src.baselines.evaluate \
      --eval-config configs/eval/config_eval.yaml \
      --model /path/to/checkpoints/qwen3_1_7b_fullft/final \
      --name Qwen3-1.7B-FullFT
"""
import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from colorama import Fore, init as colorama_init
from rich.console import Console
from rich.table import Table
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.baselines.dataset import build_prompt_text, _truncate_context
from src.utils.evaluate_utils import (
    DatasetLoader, LLMJudge, _make_progress,
    compute_accuracy, compute_macro_average,
    save_results, load_results, results_exist,
)
from src.evaluation.scorers import SpanMetricScorer

colorama_init(autoreset=True)

# Zero-shot span-only directive. Fine-tuned models learn bare-span output from their
# answer targets; a non-fine-tuned base model needs to be told, or it answers in full
# sentences (correct content, but EM~0 and low F1). Applied only when --zero-shot is set.
ZERO_SHOT_ANSWER_INSTRUCTION = (
    "Answer the question using the shortest possible answer copied verbatim from the "
    "context. Output only that answer span — no explanation, no full sentence, no "
    "markdown, no trailing punctuation."
)

_CONDITIONS = ("context", "nocontext")
_CONDITION_STYLE = {
    "context":   "[bold blue]Baseline+Ctx  [/bold blue]",
    "nocontext": "[bold yellow]Baseline-NoCtx[/bold yellow]",
}


class HFBaselineRunner:
    """Loads a plain HF causal LM and runs greedy QA generation over samples."""

    def __init__(self, model_src: str, device: str = "cuda",
                 max_seq_len: int = 1024, max_new_tokens: int = 80,
                 enable_thinking: bool = True, answer_instruction: str | None = None):
        self.device = torch.device(device)
        self.max_seq_len = max_seq_len
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking
        self.answer_instruction = answer_instruction

        self.tokenizer = AutoTokenizer.from_pretrained(model_src)
        self.tokenizer.padding_side = "left"  # left-pad for batched generation
        self.model = AutoModelForCausalLM.from_pretrained(
            model_src, torch_dtype=torch.bfloat16,
        ).to(self.device).eval()

        _pid = self.model.config.pad_token_id
        if _pid is None:
            _pid = self.tokenizer.pad_token_id
        if _pid is None:
            eos = self.model.config.eos_token_id
            _pid = eos[0] if isinstance(eos, (list, tuple)) else eos
        self.pad_token_id = _pid

    @torch.no_grad()
    def generate(self, samples: list, use_context: bool, batch_size: int,
                 progress=None, task_id=None) -> list:
        answers = []
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            all_ids = []
            for sample in batch:
                context = sample.get("context", "") if use_context else ""
                if context:
                    context = _truncate_context(
                        self.tokenizer, context, sample["question"],
                        self.max_new_tokens, self.max_seq_len,
                        answer_instruction=self.answer_instruction,
                    )
                prompt = build_prompt_text(self.tokenizer, context, sample["question"],
                                           enable_thinking=self.enable_thinking,
                                           answer_instruction=self.answer_instruction)
                ids = self.tokenizer(prompt, add_special_tokens=False,
                                     return_tensors="pt")["input_ids"]
                all_ids.append(ids)

            batch_max = max(x.shape[1] for x in all_ids)
            input_ids = torch.cat(
                [F.pad(x, (batch_max - x.shape[1], 0), value=self.pad_token_id) for x in all_ids],
                dim=0,
            ).to(self.device)
            attention_mask = (input_ids != self.pad_token_id).long()

            out = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.pad_token_id,
            )
            for row in out:
                answers.append(
                    self.tokenizer.decode(row[batch_max:], skip_special_tokens=True)
                )

            if progress is not None and task_id is not None:
                progress.update(task_id, advance=len(batch))

        return answers


def _build_result(model: str, dataset: str, condition: str,
                  samples: list, preds: list, scores: list) -> dict:
    return {
        "model": model,
        "dataset": dataset,
        "condition": condition,
        "accuracy": compute_accuracy(scores),
        "n_samples": len(samples),
        "n_invalid": scores.count(-1),
        "samples": [
            {"question": s["question"], "gold": s["answer"], "predicted": p, "score": sc}
            for s, p, sc in zip(samples, preds, scores)
        ],
    }


def _baseline_path(name: str, dataset_name: str, condition: str,
                   scorer: str = "judge") -> str:
    # judge keeps the historical filename (no suffix) so existing caches stay
    # valid; f1_em writes a distinct file so both scorers can coexist per model.
    suffix = "" if scorer == "judge" else f"_{scorer}"
    return os.path.join("evals", "baselines", name,
                        f"{dataset_name}_{condition}{suffix}.json")


def _load_full_gold_datasets(datasets_cfg: dict) -> dict:
    """Load enabled MRQA sets keeping the FULL gold list (for F1/EM max-over-golds).

    Reuses the same config toggles + files as the judge path, but reads
    ``qa['answers']`` (all acceptable spans) instead of a single detected span.
    Returns ``{dataset_name: [{question, context, answers, qid}]}``.
    """
    from src.dataset import MRQA_FILENAME
    from src.evaluation.datasets import mrqa_eval_generator

    if datasets_cfg.get("squad"):
        raise ValueError("f1_em scorer supports MRQA sets only; disable 'squad' in the config.")

    max_samples = datasets_cfg.get("max_samples")
    mrqa_dir = datasets_cfg.get("mrqa_data_dir")
    result = {}
    for key, filename in MRQA_FILENAME.items():
        if not datasets_cfg.get(key):
            continue
        if not mrqa_dir:
            raise ValueError(f"Dataset '{key}' is enabled but 'mrqa_data_dir' is not set.")
        path = os.path.join(mrqa_dir, filename)
        rows = []
        for i, row in enumerate(mrqa_eval_generator(path)):
            if max_samples is not None and i >= max_samples:
                break
            rows.append(row)
        result[key] = rows
    return result


def _macro_f1_em(per_dataset: dict) -> dict:
    """Macro-average em/f1 across datasets, per condition."""
    macro = {}
    for condition in _CONDITIONS:
        vals = [m[condition] for m in per_dataset.values() if condition in m]
        if not vals:
            continue
        macro[condition] = {
            k: sum(v[k] for v in vals) / len(vals) for k in ("em", "f1")
        }
    return macro


def _run_f1_em(args, name: str, runner, datasets: dict, batch_size: int, console) -> None:
    """Span-metric (official MRQA F1/EM) evaluation — no vLLM judge needed."""
    scorer = SpanMetricScorer()
    per_dataset = {}
    with _make_progress(console) as progress:
        for ds_name, samples in datasets.items():
            progress.console.print(
                f"\n[bold cyan]=== {ds_name} ({len(samples)} samples) ===[/bold cyan]"
            )
            cond_metrics = {}
            for condition in _CONDITIONS:
                path = _baseline_path(name, ds_name, condition, scorer="f1_em")
                if results_exist(path) and not args.force:
                    progress.console.print(f"  [yellow][{condition}] cache:[/yellow] {path}")
                    cond_metrics[condition] = load_results(path)["metrics"]
                    continue

                label = _CONDITION_STYLE[condition]
                infer_task = progress.add_task(f"{label} [dim]inference[/dim]", total=len(samples))
                preds = runner.generate(
                    samples, use_context=(condition == "context"),
                    batch_size=batch_size, progress=progress, task_id=infer_task,
                )
                score_task = progress.add_task(f"{label} [dim]scoring  [/dim]", total=len(samples))
                per_sample, aggregate = scorer.score(
                    samples, preds, progress=progress, task_id=score_task
                )
                result = {
                    "model": args.model,
                    "dataset": ds_name,
                    "condition": condition,
                    "scorer": "f1_em",
                    "n_samples": len(samples),
                    "metrics": aggregate,
                    "samples": [
                        {"question": s["question"], "gold": s["answers"],
                         "predicted": p, **ps}
                        for s, p, ps in zip(samples, preds, per_sample)
                    ],
                }
                save_results(result, path)
                cond_metrics[condition] = aggregate
                progress.console.print(
                    f"  [green][{condition}] EM={aggregate['em']:.3f} "
                    f"F1={aggregate['f1']:.3f}[/green] — saved to {path}"
                )
            per_dataset[ds_name] = cond_metrics

    macro = _macro_f1_em(per_dataset)
    summary = {
        "model": args.model,
        "name": name,
        "scorer": "f1_em",
        "datasets_evaluated": list(datasets.keys()),
        "per_dataset": per_dataset,
        "macro_average": macro,
    }
    summary_path = os.path.join("evals", "baselines", name, "summary_f1_em.json")
    save_results(summary, summary_path)

    table = Table(title=f"Baseline F1/EM — {name}", show_header=True, header_style="bold cyan")
    table.add_column("Dataset", style="bold")
    table.add_column("Ctx EM", justify="right")
    table.add_column("Ctx F1", justify="right")
    table.add_column("NoCtx EM", justify="right")
    table.add_column("NoCtx F1", justify="right")

    def _cells(m):
        ctx, noctx = m.get("context", {}), m.get("nocontext", {})
        return (
            f"{ctx.get('em', 0) * 100:.2f}", f"{ctx.get('f1', 0) * 100:.2f}",
            f"{noctx.get('em', 0) * 100:.2f}", f"{noctx.get('f1', 0) * 100:.2f}",
        )

    for ds_name, m in per_dataset.items():
        table.add_row(ds_name, *_cells(m))
    table.add_section()
    table.add_row("[bold]Average[/bold]", *(f"[bold]{c}[/bold]" for c in _cells(macro)))
    console.print(table)
    console.print(f"[green]Summary saved to {summary_path}[/green]")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a plain HF causal-LM baseline on MRQA with LLM-as-judge"
    )
    parser.add_argument("--eval-config", required=True, help="Path to eval YAML config")
    parser.add_argument("--model", required=True, help="HF repo id or local checkpoint dir")
    parser.add_argument("--name", default=None,
                        help="Output dir name under evals/baselines/ (default: model basename)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--no-think", action="store_true",
                        help="Disable Qwen3 thinking (use for non-fine-tuned models so the "
                             "generation budget isn't spent on a <think> block)")
    parser.add_argument("--force", action="store_true", help="Recompute cached results")
    parser.add_argument("--scorer", default="judge", choices=["judge", "f1_em"],
                        help="judge = LLM-as-judge accuracy (needs vLLM); "
                             "f1_em = official MRQA span F1/EM (no server, full golds)")
    parser.add_argument("--zero-shot", action="store_true",
                        help="Add a span-only answer instruction to the system prompt so a "
                             "non-fine-tuned base model emits bare spans (fair F1/EM). Do NOT "
                             "use for a fine-tuned model — it was trained on the plain prompt.")
    args = parser.parse_args()

    name = args.name or Path(args.model.rstrip("/")).name

    with open(args.eval_config) as f:
        raw_cfg = yaml.safe_load(f)
    judge_cfg = raw_cfg.get("judge")
    datasets_cfg = raw_cfg["datasets"]
    infer_cfg = raw_cfg.get("inference", {})
    batch_size = infer_cfg.get("batch_size", 8)
    max_new_tokens = infer_cfg.get("max_new_tokens", 80)

    console = Console()

    if args.scorer == "judge":
        if not judge_cfg:
            print(Fore.RED + "ERROR: scorer 'judge' requires a 'judge:' block in the config.")
            sys.exit(1)
        print(Fore.CYAN + f"Checking vLLM judge at {judge_cfg['base_url']} ...")
        if not LLMJudge(judge_cfg).health_check():
            print(Fore.RED + f"ERROR: vLLM server not reachable at {judge_cfg['base_url']}")
            sys.exit(1)
        print(Fore.GREEN + "vLLM judge reachable.")

    print(Fore.CYAN + "\nLoading datasets...")
    if args.scorer == "f1_em":
        datasets = _load_full_gold_datasets(datasets_cfg)
    else:
        datasets = DatasetLoader(datasets_cfg).load()
    if not datasets:
        print(Fore.RED + "ERROR: No datasets enabled in config.")
        sys.exit(1)
    for ds_name, samples in datasets.items():
        print(Fore.GREEN + f"  {ds_name}: {len(samples)} samples")

    print(Fore.CYAN + f"\nLoading model '{args.model}' (baseline name: {name}) ...")
    runner = HFBaselineRunner(
        args.model, device=args.device,
        max_seq_len=args.max_seq_len, max_new_tokens=max_new_tokens,
        enable_thinking=not args.no_think,
        answer_instruction=ZERO_SHOT_ANSWER_INSTRUCTION if args.zero_shot else None,
    )

    if args.scorer == "f1_em":
        _run_f1_em(args, name, runner, datasets, batch_size, console)
        return

    judge = LLMJudge(judge_cfg)

    per_dataset = {}
    with _make_progress(console) as progress:
        for ds_name, samples in datasets.items():
            progress.console.print(
                f"\n[bold cyan]=== {ds_name} ({len(samples)} samples) ===[/bold cyan]"
            )
            cond_acc = {}
            for condition in _CONDITIONS:
                path = _baseline_path(name, ds_name, condition)
                if results_exist(path) and not args.force:
                    progress.console.print(
                        f"  [yellow][{condition}] cache:[/yellow] {path}"
                    )
                    cond_acc[condition] = load_results(path)["accuracy"]
                    continue

                label = _CONDITION_STYLE[condition]
                infer_task = progress.add_task(f"{label} [dim]inference[/dim]", total=len(samples))
                preds = runner.generate(
                    samples, use_context=(condition == "context"),
                    batch_size=batch_size, progress=progress, task_id=infer_task,
                )
                judge_task = progress.add_task(f"{label} [dim]judging  [/dim]", total=len(samples))
                scores = judge.score(
                    [(s["question"], s.get("context", ""), s["answer"], p)
                     for s, p in zip(samples, preds)],
                    progress=progress, task_id=judge_task,
                )
                n_invalid = scores.count(-1)
                if n_invalid:
                    progress.console.print(
                        f"  [yellow]WARNING:[/yellow] {n_invalid} invalid judge responses excluded"
                    )
                result = _build_result(args.model, ds_name, condition, samples, preds, scores)
                save_results(result, path)
                progress.console.print(
                    f"  [green][{condition}] Accuracy: {result['accuracy']:.3f}[/green] — saved to {path}"
                )
                cond_acc[condition] = result["accuracy"]

            per_dataset[ds_name] = cond_acc

    macro = compute_macro_average(per_dataset)
    summary = {
        "model": args.model,
        "name": name,
        "datasets_evaluated": list(datasets.keys()),
        "per_dataset": per_dataset,
        "macro_average": macro,
    }
    summary_path = os.path.join("evals", "baselines", name, "summary.json")
    save_results(summary, summary_path)

    table = Table(title=f"Baseline Evaluation — {name}", show_header=True, header_style="bold cyan")
    table.add_column("Dataset", style="bold")
    table.add_column("+Context", justify="right")
    table.add_column("No-Context", justify="right")
    for ds_name, vals in per_dataset.items():
        table.add_row(ds_name, f"{vals['context']:.3f}", f"{vals['nocontext']:.3f}")
    table.add_section()
    table.add_row("[bold]Average[/bold]",
                  f"[bold]{macro['context']:.3f}[/bold]",
                  f"[bold]{macro['nocontext']:.3f}[/bold]")
    console.print(table)
    console.print(f"[green]Summary saved to {summary_path}[/green]")


if __name__ == "__main__":
    main()

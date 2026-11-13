"""Sweep top-K teacher probability coverage over MRQA dev sets.

Free-running greedy generation (same path as dump_logits.py); at each generated
answer position we measure how much softmax mass the top-K tokens hold, for a grid
of K. Emits per-split JSON + coverage-vs-K plots to pick the K for the KD dump.
"""
import json
import os
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from datasets import Dataset as HFDataset

from src.dataset import mrqa_generator
from src.baselines.dump_logits import build_prompt, first_eos_length
from src.baselines.utils import (
    make_console, load_config, quiet_external_logging, setup_distributed, align_model_tokens,
)

console = make_console()


def _log(msg, style="green"):
    console.print(msg, style=style)


def coverage_at_grid(probs: torch.Tensor, k_grid: list[int]) -> torch.Tensor:
    """Cumulative top-K softmax mass at each K in k_grid. probs [..., V] -> [..., len(k_grid)]."""
    V = probs.shape[-1]
    sorted_probs, _ = probs.sort(dim=-1, descending=True)
    cum = sorted_probs.cumsum(dim=-1)
    idx = torch.tensor([min(k, V) - 1 for k in k_grid], device=probs.device)
    return cum.index_select(-1, idx)


def distribution_entropy(probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Natural-log entropy over the last dim. probs [..., V] -> [...]."""
    return -(probs * (probs + eps).log()).sum(dim=-1)


class CoverageAccumulator:
    """Collect per-valid-position coverage[K] and entropy, reduce to a metric table."""

    def __init__(self, k_grid: list[int]):
        self.k_grid = list(k_grid)
        self._cov_chunks: list[torch.Tensor] = []  # each [n, K]
        self._ent_chunks: list[torch.Tensor] = []  # each [n]

    def update(self, cov: torch.Tensor, ent: torch.Tensor, valid: torch.Tensor) -> None:
        mask = valid.reshape(-1)
        self._cov_chunks.append(cov.reshape(-1, cov.shape[-1])[mask].detach().cpu())
        self._ent_chunks.append(ent.reshape(-1)[mask].detach().cpu())

    def summary(self) -> dict:
        cov = torch.cat(self._cov_chunks, dim=0).float()  # [N, K]
        ent = torch.cat(self._ent_chunks, dim=0).float()  # [N]
        return {
            "k_grid": self.k_grid,
            "n_positions": int(cov.shape[0]),
            "mean_coverage": cov.mean(0).tolist(),
            "p10": cov.quantile(0.10, dim=0).tolist(),
            "p50": cov.quantile(0.50, dim=0).tolist(),
            "p90": cov.quantile(0.90, dim=0).tolist(),
            "frac_ge_99": (cov >= 0.99).float().mean(0).tolist(),
            "mean_entropy": float(ent.mean()),
        }


def write_summary_json(summary: dict, path: str) -> None:
    """Write coverage summary dict to indented JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)


def plot_coverage(summary: dict, path: str, title: str) -> None:
    """Plot coverage-vs-K with mean line and p10-p90 shaded band."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    k = summary["k_grid"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.fill_between(k, summary["p10"], summary["p90"], alpha=0.2,
                    label="p10-p90")
    ax.plot(k, summary["mean_coverage"], marker="o", label="mean coverage")
    ax.set_xscale("log")
    ax.set_xlabel("Top-K")
    ax.set_ylabel("Probability coverage")
    ax.set_ylim(0.0, 1.01)
    ax.set_title(f"{title} (n={summary['n_positions']})")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def load_single_mrqa(file_path: str, seed: int) -> HFDataset:
    """Load ONE MRQA dev file (not the blended six) as a shuffled HFDataset.

    load_mrqa_dataset interleaves all six sets into one; for per-dataset coverage we
    read a single <Split>.jsonl.gz directly via the shared mrqa_generator, then shuffle
    with the config seed so max_samples is a reproducible subset.
    """
    ds = HFDataset.from_generator(mrqa_generator, gen_kwargs={"file_path": file_path})
    return ds.shuffle(seed=seed)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None, help="override paths.final_checkpoint_dir")
    args = parser.parse_args()
    cfg = load_config(args.config)

    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    is_main = rank == 0
    quiet_external_logging()

    cov_cfg = cfg["coverage"]
    data_cfg = cfg["data"]
    model_max_len = cfg["model"]["max_seq_len"]
    k_grid = list(cov_cfg["k_grid"])
    max_new_tokens = cov_cfg["max_new_tokens"]
    batch_size = cov_cfg.get("batch_size", 16)
    out_root = cov_cfg["output_dir"]
    use_context = data_cfg.get("use_context", True)

    ckpt = args.checkpoint or cfg["paths"]["final_checkpoint_dir"]
    _log(f"Loading model from {ckpt}")
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id
    model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.bfloat16).to(device).eval()
    align_model_tokens(model, tokenizer)

    gen_eos = model.generation_config.eos_token_id
    eos_ids = sorted({*([gen_eos] if isinstance(gen_eos, int) else list(gen_eos)),
                      tokenizer.eos_token_id})

    model_name = os.path.basename(os.path.normpath(ckpt))
    out_dir = os.path.join(out_root, model_name)

    for split in cov_cfg["splits"]:
        file_path = os.path.join(data_cfg["data_dir"], f"{split}.jsonl.gz")
        _log(f"[{split}] loading from {file_path}")
        ds = load_single_mrqa(file_path, seed=data_cfg.get("seed", 42))
        max_samples = data_cfg.get("max_samples")
        n_total = min(max_samples, len(ds)) if max_samples else len(ds)
        my_indices = list(range(rank, n_total, world_size))

        acc = CoverageAccumulator(k_grid)
        for b_start in range(0, len(my_indices), batch_size):
            idx_batch = my_indices[b_start:b_start + batch_size]
            prompts = [build_prompt(tokenizer, ds[i], use_context, model_max_len, max_new_tokens)
                       for i in idx_batch]
            enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                            max_length=model_max_len, add_special_tokens=False).to(device)
            ctx_len = enc["input_ids"].shape[1]
            out = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
                return_dict_in_generate=True, output_scores=True, pad_token_id=pad_id,
            )
            logits = torch.stack(out.scores, dim=1)          # [B, Ga, V]
            gen_ids = out.sequences[:, ctx_len:]             # [B, Ga]
            Ga = gen_ids.shape[1]

            probs = F.softmax(logits.float(), dim=-1)        # [B, Ga, V]
            cov = coverage_at_grid(probs, k_grid)            # [B, Ga, K]
            ent = distribution_entropy(probs)                # [B, Ga]
            length = first_eos_length(gen_ids, eos_ids)      # [B]
            valid = torch.arange(Ga, device=device)[None, :] < length[:, None]
            acc.update(cov, ent, valid)

        # Gather per-rank chunks onto rank 0 for a single summary.
        if world_size > 1:
            local_cov = torch.cat(acc._cov_chunks, dim=0) if acc._cov_chunks else torch.empty(0, len(k_grid))
            local_ent = torch.cat(acc._ent_chunks, dim=0) if acc._ent_chunks else torch.empty(0)
            gathered_cov = [None] * world_size
            gathered_ent = [None] * world_size
            dist.all_gather_object(gathered_cov, local_cov)
            dist.all_gather_object(gathered_ent, local_ent)
            if is_main:
                acc._cov_chunks = list(gathered_cov)
                acc._ent_chunks = list(gathered_ent)

        if is_main:
            summary = acc.summary()
            summary["split"] = split
            summary["model"] = ckpt
            json_path = os.path.join(out_dir, f"coverage_{split}.json")
            png_path = os.path.join(out_dir, f"coverage_{split}.png")
            write_summary_json(summary, json_path)
            plot_coverage(summary, png_path, title=f"{model_name} / {split}")
            _log(f"[{split}] n={summary['n_positions']}  "
                 f"mean@[{k_grid[0]},{k_grid[len(k_grid)//2]},{k_grid[-1]}]="
                 f"{summary['mean_coverage'][0]:.3f},"
                 f"{summary['mean_coverage'][len(k_grid)//2]:.3f},"
                 f"{summary['mean_coverage'][-1]:.3f}  -> {json_path}", "yellow")

        if world_size > 1:
            dist.barrier()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

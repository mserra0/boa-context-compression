"""Dump offline top-k teacher logits over MRQA for later KD.

Stores per generated-answer position the top-k RAW logits + indices (raw keeps
temperature tunable; top-k of logits == top-k of softmax). Multi-GPU via torchrun:
each rank dumps a disjoint slice, rank 0 merges manifest.json. Single-GPU also works.

    torchrun --nproc_per_node=<N> -m src.baselines.dump_logits --config <cfg.yaml>
"""
import os
import json
import glob
import argparse

import torch
import torch.nn.functional as F
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.dataset import load_mrqa_dataset
from src.baselines.dataset import build_prompt_text, _truncate_context
from src.baselines.utils import (
    make_console, load_config, quiet_external_logging, setup_distributed, align_model_tokens,
)

console = make_console()


def log(msg, style="green"):
    console.print(msg, style=style)


def build_prompt(tokenizer, sample, use_context, max_length, max_new_tokens):
    """Prompt-only string (no gold answer), context trimmed to leave room to generate."""
    question = sample["question"]
    context = sample["context"] if use_context else ""
    if context:
        context = _truncate_context(tokenizer, context, question, max_new_tokens, max_length)
    return build_prompt_text(tokenizer, context, question)


def first_eos_length(generated_ids: torch.Tensor, eos_ids) -> torch.Tensor:
    """[B] tokens up to and including the first eos per row (else full G)."""
    G = generated_ids.shape[1]
    eos = torch.tensor(list(eos_ids), device=generated_ids.device)
    is_eos = (generated_ids.unsqueeze(-1) == eos).any(-1)  # [B, G]
    has_eos = is_eos.any(dim=1)
    first = torch.argmax(is_eos.int(), dim=1)              # first True idx (0 if none)
    return torch.where(has_eos, first + 1, torch.full_like(first, G))


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

    dump_cfg = cfg["dump"]
    data_cfg = cfg["data"]
    model_max_len = cfg["model"]["max_seq_len"]
    k = dump_cfg["top_k"]
    max_new_tokens = dump_cfg["max_new_tokens"]
    batch_size = dump_cfg.get("batch_size", 16)
    shard_size = dump_cfg["shard_size"]
    out_dir = dump_cfg["output_dir"]
    use_context = data_cfg.get("use_context", True)

    ckpt = args.checkpoint or cfg["paths"]["final_checkpoint_dir"]
    log(f"Loading model from {ckpt}")
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    tokenizer.padding_side = "left"  # required for batched generate
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id
    model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.bfloat16).to(device).eval()
    align_model_tokens(model, tokenizer)

    # Stop set (Qwen3 = [<|im_end|>, <|endoftext|>]).
    gen_eos = model.generation_config.eos_token_id
    eos_ids = sorted({*( [gen_eos] if isinstance(gen_eos, int) else list(gen_eos) ),
                      tokenizer.eos_token_id})

    os.makedirs(out_dir, exist_ok=True)
    log(f"Writing dump to {out_dir}")

    for split in dump_cfg["splits"]:
        split_dir = os.path.join(data_cfg["data_dir"], split)
        log(f"[{split}] loading from {split_dir}")
        ds = load_mrqa_dataset(split_dir, seed=data_cfg.get("seed", 42))
        max_samples = data_cfg.get("max_samples")
        n_total = min(max_samples, len(ds)) if max_samples else len(ds)

        my_indices = list(range(rank, n_total, world_size))  # disjoint slice per rank

        buf = {"example_idx": [], "generated_ids": [], "topk_values": [],
               "topk_indices": [], "length": []}
        shard_id = 0
        coverage_sum, coverage_count = 0.0, 0

        def flush():
            nonlocal shard_id, buf
            if not buf["example_idx"]:
                return
            path = os.path.join(out_dir, f"{split}_shard_r{rank:02d}_{shard_id:05d}.pt")
            torch.save({
                "example_idx": torch.cat(buf["example_idx"], dim=0),
                "generated_ids": torch.cat(buf["generated_ids"], dim=0),
                "topk_values": torch.cat(buf["topk_values"], dim=0),
                "topk_indices": torch.cat(buf["topk_indices"], dim=0),
                "length": torch.cat(buf["length"], dim=0),
            }, path)
            shard_id += 1
            for v in buf.values():
                v.clear()

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
            logits = torch.stack(out.scores, dim=1)                       # [B, Ga, V]
            gen_ids = out.sequences[:, ctx_len:]                          # [B, Ga]
            Ga = gen_ids.shape[1]

            topk_v, topk_i = torch.topk(logits, k, dim=-1)               # [B, Ga, k]

            # top-k probability coverage (diagnostic): sum of top-k softmax probs.
            probs = F.softmax(logits.float(), dim=-1)
            cov = probs.topk(k, dim=-1).values.sum(-1)                    # [B, Ga]
            length = first_eos_length(gen_ids, eos_ids)                   # [B]
            valid = (torch.arange(Ga, device=device)[None, :] < length[:, None])
            coverage_sum += float((cov * valid).sum().item())
            coverage_count += int(valid.sum().item())

            # pad gen dim to max_new_tokens so shards stack uniformly.
            pad = max_new_tokens - Ga
            if pad > 0:
                gen_ids = F.pad(gen_ids, (0, pad), value=pad_id)
                topk_v = F.pad(topk_v, (0, 0, 0, pad), value=0.0)
                topk_i = F.pad(topk_i, (0, 0, 0, pad), value=0)

            buf["example_idx"].append(torch.tensor(idx_batch, dtype=torch.long))
            buf["generated_ids"].append(gen_ids.cpu())
            buf["topk_values"].append(topk_v.float().cpu())
            buf["topk_indices"].append(topk_i.cpu())
            buf["length"].append(length.cpu())

            if sum(t.shape[0] for t in buf["generated_ids"]) >= shard_size:
                flush()

        flush()

        mean_cov = coverage_sum / max(coverage_count, 1)
        meta = {"rank": rank, "split": split, "n_examples": len(my_indices),
                "shards": shard_id, "mean_topk_coverage": mean_cov}
        with open(os.path.join(out_dir, f".meta_{split}_r{rank:02d}.json"), "w") as f:
            json.dump(meta, f)
        log(f"[{split}] rank {rank}: {len(my_indices)} ex, {shard_id} shards "
            f"-> {os.path.join(out_dir, f'{split}_shard_r{rank:02d}_*.pt')}, "
            f"mean top-{k} coverage {mean_cov:.4f}", "yellow")

        if world_size > 1:
            dist.barrier()

        if is_main:
            metas = [json.load(open(p)) for p in
                     sorted(glob.glob(os.path.join(out_dir, f".meta_{split}_r*.json")))]
            total_ex = sum(m["n_examples"] for m in metas)
            total_shards = sum(m["shards"] for m in metas)
            weighted_cov = sum(m["mean_topk_coverage"] * m["n_examples"] for m in metas) / max(total_ex, 1)
            manifest = {
                "model": ckpt,
                "split": split,
                "top_k": k,
                "max_new_tokens": max_new_tokens,
                "world_size": world_size,
                "total_examples": total_ex,
                "total_shards": total_shards,
                "mean_topk_coverage": weighted_cov,
                "shard_glob": f"{split}_shard_r*.pt",
            }
            manifest_path = os.path.join(out_dir, f"manifest_{split}.json")
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
            log(f"[{split}] manifest written -> {manifest_path}: {total_ex} ex, "
                f"{total_shards} shards, mean top-{k} coverage {weighted_cov:.4f}")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

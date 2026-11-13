"""Pretrain Activation Beacon parameters on FineWeb-Edu with an exact token budget."""

from __future__ import annotations

import argparse
import copy
import os
import random
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import yaml
from accelerate import Accelerator, DataLoaderConfiguration
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.baselines.activation_beacon.pretraining import (
    FineWebTokenDataset,
    cap_batch_to_token_budget,
    collate_causal_lm,
    discover_parquet_files,
    token_lr_scale,
    validate_pretraining_config,
)
from src.baselines.activation_beacon.training import (
    ensure_beacon_memory_dtype,
    load_beacon_tokenizer,
    load_trainable_beacon_model,
    save_training_metadata,
)

RESTART_STATE_FILE = "pretraining_state.pt"
RANK_STATE_TEMPLATE = "pretraining_rank_{rank:05d}.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help=(
            "Checkpoint directory containing activation beacon weights and "
            f"{RESTART_STATE_FILE}. Overrides training.resume_from_checkpoint."
        ),
    )
    return parser.parse_args()


def _unwrap_model_for_ddp(model, accelerator: Accelerator):
    try:
        return accelerator.unwrap_model(model)
    except ModuleNotFoundError as error:
        if error.name != "cpuinfo":
            raise
        return getattr(model, "module", model)


def _torch_load(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _rank_state_path(checkpoint_dir: Path, rank: int) -> Path:
    return checkpoint_dir / RANK_STATE_TEMPLATE.format(rank=int(rank))


def _config_for_resume(config: dict, resume_checkpoint: str | None) -> dict:
    if not resume_checkpoint:
        return config
    resumed = copy.deepcopy(config)
    resumed["model"]["base_model_name_or_path"] = str(Path(resume_checkpoint).expanduser())
    return resumed


def _collect_rng_state(model, accelerator: Accelerator) -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    unwrapped = _unwrap_model_for_ddp(model, accelerator)
    memory = getattr(unwrapped, "memory", None)
    rng = getattr(memory, "rng", None)
    if rng is not None:
        state["beacon_memory_rng"] = rng.bit_generator.state
    return state


def _restore_rng_state(state: dict | None, model, accelerator: Accelerator) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])
    unwrapped = _unwrap_model_for_ddp(model, accelerator)
    memory = getattr(unwrapped, "memory", None)
    rng = getattr(memory, "rng", None)
    if rng is not None and state.get("beacon_memory_rng") is not None:
        rng.bit_generator.state = state["beacon_memory_rng"]


def _load_resume_states(checkpoint_dir: Path, accelerator: Accelerator) -> tuple[dict, dict]:
    main_state_path = checkpoint_dir / RESTART_STATE_FILE
    if not main_state_path.is_file():
        raise FileNotFoundError(
            f"Missing restart state: {main_state_path}. "
            "Only checkpoints saved with training.save_restart_state=true can resume exactly."
        )
    rank_state_path = _rank_state_path(checkpoint_dir, accelerator.process_index)
    if not rank_state_path.is_file():
        raise FileNotFoundError(
            f"Missing rank-local restart state for rank {accelerator.process_index}: "
            f"{rank_state_path}. Resume with the same world size/rank layout used to save."
        )
    main_state = _torch_load(main_state_path)
    rank_state = _torch_load(rank_state_path)
    expected_world_size = int(main_state.get("world_size", accelerator.num_processes))
    if expected_world_size != accelerator.num_processes:
        raise ValueError(
            f"Checkpoint was saved with world_size={expected_world_size}, "
            f"but this run has world_size={accelerator.num_processes}. Exact dataset "
            "resume requires the same DDP world size."
        )
    if int(rank_state.get("rank", accelerator.process_index)) != accelerator.process_index:
        raise ValueError(
            f"Rank state {rank_state_path} does not belong to rank {accelerator.process_index}."
        )
    return main_state, rank_state


def _save_restart_state(
    destination: Path,
    *,
    model,
    optimizer,
    accelerator: Accelerator,
    dataset: FineWebTokenDataset,
    train_state: dict,
    base_lr: float,
    warmup_tokens: int,
    token_budget: int,
) -> None:
    rank_state = {
        "version": 1,
        "rank": accelerator.process_index,
        "world_size": accelerator.num_processes,
        "dataset_state": dataset.state_dict(),
        "rng_state": _collect_rng_state(model, accelerator),
    }
    torch.save(rank_state, _rank_state_path(destination, accelerator.process_index))
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        main_state = dict(train_state)
        main_state.update(
            {
                "version": 1,
                "world_size": accelerator.num_processes,
                "optimizer_state": optimizer.state_dict(),
                "lr_state": {
                    "kind": "token_linear_warmup_decay",
                    "base_lr": base_lr,
                    "warmup_tokens": warmup_tokens,
                    "token_budget": token_budget,
                    "current_lrs": [group["lr"] for group in optimizer.param_groups],
                },
            }
        )
        torch.save(main_state, destination / RESTART_STATE_FILE)
    accelerator.wait_for_everyone()


def save_checkpoint(
    model,
    tokenizer,
    accelerator,
    output_dir: str,
    config: dict,
    stats: dict,
    tokens_processed: int,
    optimizer_step: int,
    name: str,
    *,
    optimizer=None,
    dataset: FineWebTokenDataset | None = None,
    train_state: dict | None = None,
    save_restart_state: bool = False,
    base_lr: float = 0.0,
    warmup_tokens: int = 0,
    token_budget: int = 0,
) -> None:
    accelerator.wait_for_everyone()
    destination = Path(output_dir) / name
    if accelerator.is_main_process:
        destination.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    if save_restart_state:
        if optimizer is None or dataset is None or train_state is None:
            raise ValueError("Restart checkpoints require optimizer, dataset, and train_state.")
        _save_restart_state(
            destination,
            model=model,
            optimizer=optimizer,
            accelerator=accelerator,
            dataset=dataset,
            train_state=train_state,
            base_lr=base_lr,
            warmup_tokens=warmup_tokens,
            token_budget=token_budget,
        )

    if not accelerator.is_main_process:
        return
    unwrapped = _unwrap_model_for_ddp(model, accelerator)
    unwrapped.save_pretrained(
        destination,
        state_dict=accelerator.get_state_dict(model),
        safe_serialization=True,
    )
    tokenizer.save_pretrained(destination)
    metadata = dict(stats)
    metadata.update(
        {
            "tokens_processed": tokens_processed,
            "optimizer_step": optimizer_step,
            "restart_state": bool(save_restart_state),
        }
    )
    save_training_metadata(destination, config, metadata)


def _training_state_snapshot(
    *,
    tokens_processed: int,
    optimizer_step: int,
    micro_step: int,
    sequences_processed: int,
    weighted_loss: float,
    weighted_loss_tokens: int,
    next_log: int,
    next_save: int,
    elapsed_seconds: float,
) -> dict:
    return {
        "tokens_processed": int(tokens_processed),
        "optimizer_step": int(optimizer_step),
        "micro_step": int(micro_step),
        "sequences_processed": int(sequences_processed),
        "weighted_loss": float(weighted_loss),
        "weighted_loss_tokens": int(weighted_loss_tokens),
        "next_log": int(next_log),
        "next_save": int(next_save),
        "elapsed_seconds": float(elapsed_seconds),
    }


def main() -> None:
    args = parse_args()
    with open(args.config) as handle:
        config = yaml.safe_load(handle)
    training = config["training"]
    if args.resume_from_checkpoint:
        training["resume_from_checkpoint"] = args.resume_from_checkpoint
    resume_checkpoint = training.get("resume_from_checkpoint")
    save_restart_state = bool(training.get("save_restart_state", bool(resume_checkpoint)))
    validate_pretraining_config(config)
    files = discover_parquet_files(
        config["data"]["data_dir"], config["data"].get("file_pattern", "**/*.parquet")
    )
    if args.validate_only:
        print(
            f"Activation Beacon pretraining configuration is valid "
            f"({len(files)} Parquet files)."
        )
        return

    accelerator = Accelerator(
        gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", 1)),
        mixed_precision=training.get("mixed_precision", "bf16"),
        log_with="wandb" if config.get("wandb", {}).get("enabled", False) else None,
        # Each rank streams its own disjoint dataset shard (FineWebTokenDataset
        # self-shards by process_index). dispatch_batches=True would make rank0
        # read + torch.cat variable-length batches across ranks and crash.
        dataloader_config=DataLoaderConfiguration(dispatch_batches=False),
    )
    # token_budget is the GLOBAL supervised-token target. Each rank streams its
    # own shard and processes a different number of tokens per step, so the stop
    # condition is driven by an all-reduced global count (see the loop) — never a
    # per-rank count, which would let ranks exit at different steps and deadlock
    # DDP at the next collective.
    token_budget = int(training["token_budget"])
    world_size = accelerator.num_processes
    resume_main_state = resume_rank_state = None
    if resume_checkpoint:
        resume_checkpoint = str(Path(resume_checkpoint).expanduser())
        resume_main_state, resume_rank_state = _load_resume_states(
            Path(resume_checkpoint), accelerator
        )

    if accelerator.is_main_process and config.get("wandb", {}).get("enabled", False):
        accelerator.init_trackers(
            config["wandb"].get("project", "boa"),
            config=config,
            init_kwargs={"wandb": {"name": config["wandb"].get("name")}},
        )

    model_config = _config_for_resume(config, resume_checkpoint)
    tokenizer = load_beacon_tokenizer(model_config)
    dataset = FineWebTokenDataset(
        files,
        tokenizer,
        text_column=config["data"].get("text_column", "text"),
        min_length=int(config["data"]["min_length"]),
        max_length=int(config["data"]["max_length"]),
        seed=int(config["data"].get("seed", 42)),
        tokenizer_batch_size=int(config["data"].get("tokenizer_batch_size", 64)),
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
    )
    if resume_rank_state is not None:
        dataset.load_state_dict(resume_rank_state.get("dataset_state"))
    loader = DataLoader(
        dataset,
        batch_size=int(training.get("batch_size", 1)),
        num_workers=int(config["data"].get("num_workers", 0)),
        persistent_workers=int(config["data"].get("num_workers", 0)) > 0,
        collate_fn=partial(
            collate_causal_lm, pad_token_id=tokenizer.pad_token_id
        ),
    )
    # Do NOT accelerator.prepare(loader): FineWebTokenDataset already self-shards
    # by process_index. accelerate wraps an IterableDataset in IterableDatasetShard,
    # which re-shards per process — double-sharding to ~1/(world_size^2) of the data
    # and draining each rank's stream almost immediately (StopIteration -> DDP
    # deadlock). Keep the self-shard and move batches to the device by hand. The
    # loop uses manual gradient accumulation, so no accelerator.accumulate() context
    # is needed on the loader.
    iterator = iter(loader)

    model, tokenizer, stats = load_trainable_beacon_model(model_config, tokenizer=tokenizer)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(training.get("learning_rate", 5e-5)),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    model, optimizer = accelerator.prepare(model, optimizer)
    # from_pretrained() leaves models in eval mode. The official Beacon forward
    # disables autograd whenever self.training is false.
    model.train()
    accumulation_steps = int(training.get("gradient_accumulation_steps", 1))
    optimizer.zero_grad(set_to_none=True)

    base_lr = float(training.get("learning_rate", 5e-5))
    warmup_tokens = int(training.get("warmup_tokens", 0))
    logging_tokens = int(training.get("logging_tokens", 1_000_000))
    save_every_tokens = int(training.get("save_every_tokens", 100_000_000))

    if resume_main_state is not None:
        optimizer.load_state_dict(resume_main_state["optimizer_state"])
        tokens_processed = int(resume_main_state["tokens_processed"])
        optimizer_step = int(resume_main_state["optimizer_step"])
        micro_step = int(resume_main_state.get("micro_step", optimizer_step * accumulation_steps))
        sequences_processed = int(resume_main_state.get("sequences_processed", 0))
        weighted_loss = float(resume_main_state.get("weighted_loss", 0.0))
        weighted_loss_tokens = int(resume_main_state.get("weighted_loss_tokens", 0))
        next_log = int(resume_main_state.get("next_log", logging_tokens))
        next_save = int(resume_main_state.get("next_save", save_every_tokens))
        elapsed_before_resume = float(resume_main_state.get("elapsed_seconds", 0.0))
        _restore_rng_state(resume_rank_state.get("rng_state"), model, accelerator)
    else:
        initial_scale = token_lr_scale(0, token_budget, warmup_tokens)
        for group in optimizer.param_groups:
            group["lr"] = base_lr * initial_scale
        tokens_processed = 0
        optimizer_step = 0
        micro_step = 0
        sequences_processed = 0
        weighted_loss = 0.0
        weighted_loss_tokens = 0
        next_log = logging_tokens
        next_save = save_every_tokens
        elapsed_before_resume = 0.0
    started = time.perf_counter() - elapsed_before_resume

    if accelerator.is_main_process:
        ratio = stats["trainable"] / stats["total"]
        print(f"FineWeb-Edu Parquet files: {len(files)}")
        print(f"Token budget: {token_budget:,} global across {world_size} ranks")
        if resume_checkpoint:
            print(
                f"Resuming from {resume_checkpoint} at "
                f"{tokens_processed:,} tokens / optimizer_step={optimizer_step:,}"
            )
        if save_restart_state:
            print("Exact restart state is enabled; data.num_workers must remain 0.")
        print(
            f"Trainable parameters: {stats['trainable']:,}/{stats['total']:,} "
            f"({ratio:.3%})"
        )
    progress = tqdm(
        total=token_budget,
        initial=min(tokens_processed, token_budget),
        unit="tok",
        unit_scale=True,
        dynamic_ncols=True,
        desc="Beacon pretraining",
        disable=not accelerator.is_main_process,
    )

    # tokens_processed is a GLOBAL count (summed across ranks each step). Every
    # rank evaluates the same loop condition, so all ranks run an identical number
    # of iterations — required for DDP collectives to stay in lockstep.
    while tokens_processed < token_budget:
        try:
            batch = next(iterator)
        except StopIteration as error:
            # The dataset is an infinite stream, so this should never fire. Keep it
            # as a defensive guard: if it ever does, fail loudly rather than hang.
            raise RuntimeError(
                f"FineWeb-Edu was exhausted after {tokens_processed:,} supervised tokens, "
                f"before the requested {token_budget:,}."
            ) from error
        # The loader is not accelerator.prepare()'d (avoids double-sharding), so
        # batches arrive on CPU — move them to this rank's device explicitly.
        batch = {key: value.to(accelerator.device) for key, value in batch.items()}

        # Cap each rank to a fair share of the remaining global budget. The +1
        # keeps keep>=1 on every rank near the end, so no rank produces a
        # zero-token batch and control flow stays uniform across ranks (a per-rank
        # skip would desync DDP).
        remaining = max(0, token_budget - tokens_processed)
        per_rank_remaining = remaining // world_size + 1
        batch, batch_tokens = cap_batch_to_token_budget(batch, per_rank_remaining)
        unwrapped = _unwrap_model_for_ddp(model, accelerator)
        ensure_beacon_memory_dtype(unwrapped)
        unwrapped.memory.reset()
        outputs = model(**batch)
        loss = outputs.loss
        if not loss.requires_grad:
            raise RuntimeError(
                "Activation Beacon loss is detached from trainable parameters. "
                "Ensure model.train() is active and sequences create Beacon tokens."
            )
        accelerator.backward(loss / accumulation_steps)
        micro_step += 1
        # Sum this step's supervised tokens across ranks. This all-reduce is a
        # collective every rank hits once per iteration, keeping them in lockstep.
        step_tokens = int(
            accelerator.reduce(
                torch.tensor(batch_tokens, device=accelerator.device),
                reduction="sum",
            ).item()
        )
        tokens_after_batch = tokens_processed + step_tokens
        should_step = (
            micro_step % accumulation_steps == 0 or tokens_after_batch >= token_budget
        )
        if should_step:
            # Apply the token-based schedule before the optimizer update. At this
            # point tokens_processed includes earlier micro-batches in the group,
            # avoiding both a zero-LR first step and a zero-LR final step.
            scale = token_lr_scale(tokens_processed, token_budget, warmup_tokens)
            for group in optimizer.param_groups:
                group["lr"] = base_lr * scale
            accumulation_in_group = micro_step % accumulation_steps or accumulation_steps
            if accumulation_in_group != accumulation_steps:
                correction = accumulation_steps / accumulation_in_group
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(correction)
            accelerator.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                float(training.get("max_grad_norm", 1.0)),
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        tokens_processed = tokens_after_batch
        sequences_processed += int(batch["input_ids"].shape[0])
        weighted_loss += loss.detach().float().item() * batch_tokens
        weighted_loss_tokens += batch_tokens
        if should_step:
            optimizer_step += 1

        elapsed = max(time.perf_counter() - started, 1e-6)
        average_loss = weighted_loss / max(weighted_loss_tokens, 1)
        tokens_per_second = tokens_processed / elapsed
        if accelerator.is_main_process:
            progress.update(step_tokens)
            progress.set_postfix(
                loss=f"{loss.detach().float().item():.4f}",
                avg=f"{average_loss:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                step=optimizer_step,
                tok_s=f"{tokens_per_second:,.0f}",
            )

        if tokens_processed >= next_log or tokens_processed >= token_budget:
            accelerator.log(
                {
                    "train/loss": average_loss,
                    "train/tokens_processed": tokens_processed,
                    "train/token_budget_progress": tokens_processed / token_budget,
                    "train/tokens_per_second": tokens_per_second,
                    "train/sequences_processed": sequences_processed,
                    "train/optimizer_step": optimizer_step,
                    "train/learning_rate": optimizer.param_groups[0]["lr"],
                },
                step=optimizer_step,
            )
            while next_log <= tokens_processed:
                next_log += logging_tokens

        if should_step and tokens_processed >= next_save and tokens_processed < token_budget:
            while next_save <= tokens_processed:
                next_save += save_every_tokens
            elapsed = max(time.perf_counter() - started, 1e-6)
            save_checkpoint(
                model,
                tokenizer,
                accelerator,
                config["paths"]["output_dir"],
                config,
                stats,
                tokens_processed,
                optimizer_step,
                f"checkpoint-tokens-{tokens_processed}",
                optimizer=optimizer,
                dataset=dataset,
                train_state=_training_state_snapshot(
                    tokens_processed=tokens_processed,
                    optimizer_step=optimizer_step,
                    micro_step=micro_step,
                    sequences_processed=sequences_processed,
                    weighted_loss=weighted_loss,
                    weighted_loss_tokens=weighted_loss_tokens,
                    next_log=next_log,
                    next_save=next_save,
                    elapsed_seconds=elapsed,
                ),
                save_restart_state=save_restart_state,
                base_lr=base_lr,
                warmup_tokens=warmup_tokens,
                token_budget=token_budget,
            )

    progress.close()
    elapsed = max(time.perf_counter() - started, 1e-6)
    save_checkpoint(
        model,
        tokenizer,
        accelerator,
        config["paths"]["output_dir"],
        config,
        stats,
        tokens_processed,
        optimizer_step,
        "final",
        optimizer=optimizer,
        dataset=dataset,
        train_state=_training_state_snapshot(
            tokens_processed=tokens_processed,
            optimizer_step=optimizer_step,
            micro_step=micro_step,
            sequences_processed=sequences_processed,
            weighted_loss=weighted_loss,
            weighted_loss_tokens=weighted_loss_tokens,
            next_log=next_log,
            next_save=next_save,
            elapsed_seconds=elapsed,
        ),
        save_restart_state=save_restart_state,
        base_lr=base_lr,
        warmup_tokens=warmup_tokens,
        token_budget=token_budget,
    )
    accelerator.end_training()


if __name__ == "__main__":
    main()

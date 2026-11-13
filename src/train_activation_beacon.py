"""Fine-tune Activation Beacon parameters on this repository's MRQA data."""

from __future__ import annotations

import argparse
import math
import os
import sys
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import torch
import yaml
from accelerate import Accelerator, DataLoaderConfiguration
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.baselines.activation_beacon.training import (
    build_encoded_datasets,
    collate_beacon_batch,
    create_beacon_dataloader,
    ensure_beacon_memory_dtype,
    load_beacon_tokenizer,
    load_trainable_beacon_model,
    save_training_metadata,
    validate_training_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate configuration without loading data or model weights.",
    )
    return parser.parse_args()


def _move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _unwrap_model_for_ddp(model, accelerator: Accelerator):
    try:
        return accelerator.unwrap_model(model)
    except ModuleNotFoundError as error:
        if error.name != "cpuinfo":
            raise
        return getattr(model, "module", model)


def evaluate(model, dataloader, accelerator: Accelerator) -> float:
    if dataloader is None:
        return float("nan")
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in dataloader:
            batch = _move_batch_to_device(batch, accelerator.device)
            unwrapped = _unwrap_model_for_ddp(model, accelerator)
            ensure_beacon_memory_dtype(unwrapped)
            unwrapped.memory.reset()
            outputs = model(**batch)
            losses.append(accelerator.gather_for_metrics(outputs.loss.detach()).mean())
    model.train()
    return torch.stack(losses).mean().item() if losses else float("nan")


def save_checkpoint(model, tokenizer, accelerator, output_dir, config, stats, name):
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return
    destination = Path(output_dir) / name
    unwrapped = _unwrap_model_for_ddp(model, accelerator)
    unwrapped.save_pretrained(
        destination,
        state_dict=accelerator.get_state_dict(model),
        safe_serialization=True,
    )
    tokenizer.save_pretrained(destination)
    save_training_metadata(destination, config, stats)


def main() -> None:
    args = parse_args()
    with open(args.config) as handle:
        config = yaml.safe_load(handle)
    validate_training_config(config)
    if args.validate_only:
        print("Activation Beacon training configuration is valid.")
        return

    training = config["training"]
    accelerator = Accelerator(
        gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", 1)),
        mixed_precision=training.get("mixed_precision", "bf16"),
        log_with="wandb" if config.get("wandb", {}).get("enabled", False) else None,
        dataloader_config=DataLoaderConfiguration(dispatch_batches=False),
        # Dataloaders are manually rank-sharded below. Without this flag,
        # Accelerate wraps the scheduler and advances it once per process when
        # split_batches=False, making the LR hit zero too early on multi-GPU runs.
        step_scheduler_with_optimizer=False,
    )
    if accelerator.is_main_process and config.get("wandb", {}).get("enabled", False):
        accelerator.init_trackers(
            config["wandb"].get("project", "boa"),
            config=config,
            init_kwargs={"wandb": {"name": config["wandb"].get("name")}},
        )

    # Prepare and cache tokenized data before materializing 7B model weights.
    # This keeps Arrow worker processes small and avoids inheriting the loaded model.
    tokenizer = load_beacon_tokenizer(config)
    with accelerator.main_process_first():
        train_dataset, validation_dataset, data_stats = build_encoded_datasets(
            tokenizer,
            config["data"],
            beacon_window=int(config["model"]["beacon_window"]),
        )
    model, tokenizer, stats = load_trainable_beacon_model(config, tokenizer=tokenizer)
    stats.update(data_stats)
    if accelerator.is_main_process:
        ratio = stats["trainable"] / stats["total"]
        print(f"Beacon-active examples: {data_stats['retained']}/{data_stats['records']}")
        print(f"Trainable parameters: {stats['trainable']:,}/{stats['total']:,} ({ratio:.3%})")

    collate = partial(collate_beacon_batch, pad_token_id=tokenizer.pad_token_id)
    data_seed = int(config["data"].get("seed", 42))
    train_loader, train_sampler = create_beacon_dataloader(
        train_dataset,
        batch_size=int(training.get("batch_size", 1)),
        shuffle=True,
        num_workers=int(training.get("num_workers", 0)),
        collate_fn=collate,
        seed=data_seed,
        accelerator=accelerator,
    )
    validation_loader = None
    if validation_dataset is not None:
        validation_loader, _ = create_beacon_dataloader(
            validation_dataset,
            batch_size=int(training.get("batch_size", 1)),
            shuffle=False,
            num_workers=int(training.get("num_workers", 0)),
            collate_fn=collate,
            seed=data_seed,
            accelerator=accelerator,
        )

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(training.get("learning_rate", 1e-5)),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    epochs = int(training.get("num_epochs", 1))
    accumulation = int(training.get("gradient_accumulation_steps", 1))
    update_steps = math.ceil(len(train_loader) / accumulation) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(training.get("warmup_steps", 0)),
        num_training_steps=update_steps,
    )

    # Dataloaders are explicitly rank-sharded above, matching the pretraining
    # script's no-double-sharding pattern. Keep them out of accelerator.prepare
    # and move batches to the local rank device inside the loops.
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    global_step = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        progress = tqdm(
            train_loader,
            total=len(train_loader),
            desc=f"Epoch {epoch}/{epochs}",
            unit="batch",
            dynamic_ncols=True,
            disable=not accelerator.is_main_process,
        )
        for batch_index, batch in enumerate(progress, start=1):
            batch = _move_batch_to_device(batch, accelerator.device)
            unwrapped = _unwrap_model_for_ddp(model, accelerator)
            ensure_beacon_memory_dtype(unwrapped)
            unwrapped.memory.reset()
            is_last_batch = batch_index == len(train_loader)
            should_step = batch_index % accumulation == 0 or is_last_batch
            sync_context = nullcontext() if should_step else accelerator.no_sync(model)
            with sync_context:
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss / accumulation)
            if should_step:
                accumulation_in_group = batch_index % accumulation or accumulation
                if accumulation_in_group != accumulation:
                    correction = accumulation / accumulation_in_group
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(correction)
                accelerator.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    float(training.get("max_grad_norm", 1.0)),
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            current_loss = loss.detach().float().item()
            running_loss += current_loss
            average = running_loss / batch_index
            if accelerator.is_main_process:
                progress.set_postfix(
                    loss=f"{current_loss:.4f}",
                    avg_loss=f"{average:.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                    step=global_step,
                    refresh=True,
                )
            if (
                should_step
                and accelerator.is_main_process
                and global_step % int(training.get("logging_steps", 10)) == 0
            ):
                accelerator.log(
                    {
                        "train/loss": current_loss,
                        "train/loss_step": current_loss,
                        "train/loss_epoch_avg": average,
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/epoch": epoch,
                        "train/global_step": global_step,
                    },
                    step=global_step,
                )
        progress.close()

        validation_loss = evaluate(model, validation_loader, accelerator)
        if accelerator.is_main_process:
            tqdm.write(f"epoch={epoch} validation_loss={validation_loss:.6f}")
            accelerator.log({"validation/loss": validation_loss}, step=global_step)
        save_checkpoint(
            model,
            tokenizer,
            accelerator,
            config["paths"]["output_dir"],
            config,
            stats,
            f"checkpoint-epoch-{epoch}",
        )

    save_checkpoint(
        model,
        tokenizer,
        accelerator,
        config["paths"]["output_dir"],
        config,
        stats,
        "final",
    )
    accelerator.end_training()


if __name__ == "__main__":
    main()

import math
import time
import signal
from itertools import cycle
from contextlib import nullcontext

import torch
import wandb
from rich.console import Console
from rich.progress import (
    Progress, BarColumn, TextColumn, TimeElapsedColumn,
    TimeRemainingColumn, MofNCompleteColumn, SpinnerColumn,
)
import torch.distributed as dist

from .utils.utils import save_checkpoint


def _move(d, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in d.items()}


def _batch_token_counts(batch):
    """(total_tokens, target_tokens) for one micro-batch, computed on the raw CPU
    batch (no GPU sync). Handles both backends: mamba2 packs context lengths in
    'lengths', mamba1 carries an 'attention_mask'."""
    tgt = int(batch["target_inputs"]["target_mask"].sum().item())
    comp = batch["compressor_inputs"]
    if "lengths" in comp:
        ctx = int(comp["lengths"].sum().item())
    elif "attention_mask" in comp:
        ctx = int(comp["attention_mask"].sum().item())
    else:
        ctx = 0
    return ctx + tgt, tgt


@torch.no_grad()
def evaluate_lm(model, dataloader, device):
    """Mean continuation CE + perplexity over a held-out loader.

    Always reports the STUDENT cross-entropy on Y via ce_readout, regardless of the
    training objective (LM-CE or KD). A KD-pretrain run's training loss is a KL and is
    not comparable across runs, so validation uses this shared CE metric so the two
    objectives can be compared directly."""
    model.eval()
    core = model.module if hasattr(model, "module") else model
    total = torch.tensor(0.0, device=device)
    nb = torch.tensor(0.0, device=device)
    for batch in dataloader:
        tgt = _move(batch["target_inputs"], device)
        comp = _move(batch["compressor_inputs"], device)
        with torch.no_grad():
            loss = core.ce_readout(tgt, comp)
        total += loss.detach().to(device)
        nb += 1
    if dist.is_initialized():
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        dist.all_reduce(nb, op=dist.ReduceOp.SUM)
    ce = (total / nb.clamp_min(1.0)).item()
    return ce, math.exp(ce)


def pretrain(config, device, train_loader, val_loader, model, optimizer, scheduler):
    """Single-pass, step-based LM-pretraining driver. Counts GLOBAL STEPS instead of
    epochs: exactly one pass over the corpus, so log/eval/checkpoint are step-triggered
    and there is no epoch-level early stopping."""
    rank = config.get("global_rank", 0)
    max_steps = config["max_steps"]
    grad_accum = config.get("gradient_accumulation_steps", 1)
    log_interval = config.get("log_interval", 50)
    eval_interval = config.get("eval_interval", 500)
    save_interval = config.get("save_interval", 1000)
    best_ce = float("inf")
    # Train-side loss label follows the objective: KD's forward loss is a KL, not a CE,
    # and its exp() is not a perplexity. LM-CE keeps ce/ppl. (val_ce below is always the
    # real student CE via ce_readout, regardless of objective — a cross-run yardstick.)
    loss_tag = "kl" if config.get("pretrain_loss", "ce") == "kd" else "ce"

    config["dataloader_len"] = len(train_loader)
    model.train()
    global_step = 0
    running = 0.0
    tokens_seen = 0
    target_tokens_seen = 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    t_start = time.perf_counter()

    console = Console()

    if rank == 0:
        wandb.define_metric("tokens_seen")
        wandb.define_metric(f"loss_vs_tokens/train_{loss_tag}", step_metric="tokens_seen")
        wandb.define_metric("loss_vs_tokens/val_ce", step_metric="tokens_seen")

    try:
      with Progress(
        SpinnerColumn(), TextColumn("[bold cyan]{task.description}"), BarColumn(),
        MofNCompleteColumn(), TimeElapsedColumn(), TimeRemainingColumn(),
        TextColumn("[yellow]{task.fields[stats]}"), console=console, disable=(rank != 0),
      ) as progress:
        task_id = progress.add_task("Pretrain", total=max_steps, stats="")
        if rank == 0:
            console.log(f"[green]LM pretrain · {max_steps} steps · "
                        f"eval every {eval_interval}[/green]")

        batch_iter = cycle(train_loader) if config.get("overfit") else train_loader
        for micro, batch in enumerate(batch_iter):
            ttl_tok, tgt_tok = _batch_token_counts(batch)
            tokens_seen += ttl_tok * world
            target_tokens_seen += tgt_tok * world
            batch_size = batch["target_inputs"]["target_mask"].shape[0]

            tgt = _move(batch["target_inputs"], device)
            comp = _move(batch["compressor_inputs"], device)
            # KD collates add teacher_inputs (plaintext prefix for the teacher pass); the
            # LM-CE path has no such key and calls the 2-arg forward.
            teacher = batch.get("teacher_inputs")
            if teacher is not None:
                teacher = _move(teacher, device)

            is_boundary = ((micro + 1) % grad_accum == 0)
            sync_ctx = nullcontext() if (is_boundary or not hasattr(model, "no_sync")) else model.no_sync()
            with sync_ctx:
                raw = model(tgt, comp, teacher) if teacher is not None else model(tgt, comp)
                loss = raw / grad_accum
                loss.backward()

            grad_norm = None
            if is_boundary:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=config["gradient_clip_max_norm"])
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
                global_step += 1

                step_loss = loss.item() * grad_accum
                running += step_loss
                progress.update(task_id, advance=1,
                                stats=f"{loss_tag}={step_loss:.4f} avg={running / global_step:.4f}")

                if rank == 0 and global_step % log_interval == 0:
                    elapsed = max(time.perf_counter() - t_start, 1e-9)
                    log = {
                        f"train/{loss_tag}": step_loss,
                        f"train/avg_{loss_tag}": running / global_step,
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "train/step": global_step,
                        "train/tokens_seen": tokens_seen,
                        "train/target_tokens_seen": target_tokens_seen,
                        "throughput/tokens_per_sec": tokens_seen / elapsed,
                        "throughput/steps_per_sec": global_step / elapsed,
                        "data/avg_context_len": (ttl_tok - tgt_tok) / batch_size,
                        "data/avg_target_len": tgt_tok / batch_size,
                        "tokens_seen": tokens_seen,
                        f"loss_vs_tokens/train_{loss_tag}": step_loss,
                    }
                    # ppl = exp(CE) only; exp(KL) is not a perplexity, so skip it for KD.
                    if loss_tag == "ce":
                        log["train/ppl"] = math.exp(min(step_loss, 20.0))
                    if grad_norm is not None:
                        log["train/grad_norm"] = float(grad_norm)
                    wandb.log(log, step=global_step)

                if global_step % eval_interval == 0:
                    val_ce, val_ppl = evaluate_lm(model, val_loader, device)
                    model.train()
                    is_best = val_ce < best_ce
                    best_ce = min(best_ce, val_ce)
                    if rank == 0:
                        marker = " · [bold green]✓ new best[/bold green]" if is_best else ""
                        console.log(
                            f"step {global_step:>6} · val_ce {val_ce:.4f} · ppl {val_ppl:8.2f}{marker}")
                        wandb.log({
                            "val/ce": val_ce,
                            "val/ppl": val_ppl,
                            "val/best_ce": best_ce,
                            "val/tokens_seen": tokens_seen,
                            "tokens_seen": tokens_seen,
                            "loss_vs_tokens/val_ce": val_ce,
                        }, step=global_step)
                    if is_best:
                        path = save_checkpoint(model=model, optimizer=optimizer, scheduler=scheduler,
                                        epoch=1, train_loss=running / global_step, val_loss=val_ce,
                                        config=config, best_val_loss=best_ce, total_steps=global_step,
                                        is_interrupted=False, step_in_epoch=global_step, is_best=True,
                                        quiet=True)
                        if rank == 0 and path:
                            console.log(f"[dim]best checkpoint → {path}[/dim]")

                if global_step % save_interval == 0:
                    path = save_checkpoint(model=model, optimizer=optimizer, scheduler=scheduler,
                                    epoch=1, train_loss=running / global_step, val_loss=float("inf"),
                                    config=config, best_val_loss=best_ce, total_steps=global_step,
                                    is_interrupted=False, step_in_epoch=global_step, is_best=False,
                                    quiet=True)
                    if rank == 0 and path:
                        console.log(f"[dim]checkpoint → {path}[/dim]")

                if global_step >= max_steps:
                    break
        else:
            # for-loop fell through without `break` -> train_loader was exhausted
            # (single pass over the corpus) before reaching max_steps.
            if rank == 0:
                console.log(
                    f"[bold yellow]⚠ Dataloader exhausted at step {global_step}/{max_steps} "
                    f"after one full pass ({len(train_loader):,} batches/rank). "
                    f"max_steps not reached — corpus yields fewer optimizer steps per pass. "
                    f"Lower max_steps or enable multi-epoch to train longer.[/bold yellow]")

    except KeyboardInterrupt:
        old_handler = signal.signal(signal.SIGINT, signal.SIG_IGN) if rank == 0 else None
        try:
            if rank == 0:
                console.log(f"[bold red]Interrupt at step {global_step} — saving checkpoint...[/bold red]")
            path = save_checkpoint(model=model, optimizer=optimizer, scheduler=scheduler,
                                   epoch=1, train_loss=(running / global_step if global_step else 0.0),
                                   val_loss=float("inf"), config=config, best_val_loss=best_ce,
                                   total_steps=global_step, is_interrupted=True,
                                   step_in_epoch=global_step, quiet=True)
            if rank == 0 and path:
                console.log(f"[red]interrupted checkpoint → {path}[/red]")
        except Exception as save_err:
            if rank == 0:
                console.log(f"[bold red]Failed to save interrupted checkpoint: {save_err}[/bold red]")
        finally:
            if old_handler is not None:
                signal.signal(signal.SIGINT, old_handler)
        raise

    if rank == 0:
        console.log(f"[bold green]Pretraining complete · best val CE {best_ce:.4f}[/bold green]")
        wandb.finish()

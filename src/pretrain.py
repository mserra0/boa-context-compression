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

from .utils.utils import save_checkpoint, load_checkpoint, get_latest_checkpoint


def _move(d, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in d.items()}


def _forward(model, batch, device, return_logits: bool = False):
    """Dispatch to the wrapped model's forward with whichever batch shape the active
    collate function produced: Mamba's collate_fn_pretrain/collate_fn_pretrain_mamba1
    (LMPretrainer.forward(target_inputs, compressor_inputs)) or ICAE's collate_fn
    (ICAEDestiller.forward(llm_inputs, compressor_inputs, continuation_inputs=...)),
    mirroring train_epoch's calling convention for the latter (src/train.py).
    """
    compressor_inputs = _move(batch["compressor_inputs"], device)
    if "target_inputs" in batch:
        target_inputs = _move(batch["target_inputs"], device)
        return model(target_inputs, compressor_inputs, return_logits=return_logits)

    llm_inputs = _move(batch["llm_inputs"], device)
    extra_kwargs = {}
    if "continuation_inputs" in batch:
        extra_kwargs["continuation_inputs"] = _move(batch["continuation_inputs"], device)
    if "answer_inputs" in batch:
        extra_kwargs["answer_inputs"] = _move(batch["answer_inputs"], device)
    return model(llm_inputs, compressor_inputs, **extra_kwargs, return_logits=return_logits)


def _batch_token_counts(batch):
    """(total_tokens, target_tokens) for one micro-batch, computed on the raw CPU
    batch (no GPU sync). Handles all backends: mamba2 packs context lengths in
    'lengths', mamba1 carries an 'attention_mask', and ICAE's continuation target
    lives under 'continuation_inputs' instead of 'target_inputs'."""
    if "target_inputs" in batch:
        tgt = int(batch["target_inputs"]["target_mask"].sum().item())
    else:
        tgt = int(batch["continuation_inputs"]["attention_mask"].sum().item())
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
    """Mean continuation-LM CE + perplexity over a held-out loader.

    Reports ONLY the continuation-prediction component of the loss — the same
    quantity Mamba's LMPretrainer computes (src/pretrain_model.py) and what the
    ICAE paper reports — never a blend with ICAE's autoencoding term. For
    objective='ae_lm', the wrapped model's forward() sets
    `last_loss_components = {"ae_loss", "lm_loss"}` as a side effect (src/model.py);
    we pull `lm_loss` out of that instead of using the blended scalar `loss`
    itself, so val_ce/val_ppl (and therefore best-checkpoint selection) stay
    comparable across compressor backends instead of being dragged down by the
    much-easier AE reconstruction term. ae_loss is returned too, purely for
    diagnostic logging — it has no influence on ce/ppl/best_ce.
    """
    model.eval()
    underlying = model.module if hasattr(model, "module") else model
    total_lm = torch.tensor(0.0, device=device)
    total_ae = torch.tensor(0.0, device=device)
    has_ae = torch.tensor(0.0, device=device)
    nb = torch.tensor(0.0, device=device)
    for batch in dataloader:
        loss = _forward(model, batch, device)
        components = getattr(underlying, "last_loss_components", None)
        if components is not None:
            total_lm += components["lm_loss"].detach().to(device)
            total_ae += components["ae_loss"].detach().to(device)
            has_ae.fill_(1.0)
        else:
            total_lm += loss.detach().to(device)
        nb += 1
    if dist.is_initialized():
        dist.all_reduce(total_lm, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_ae, op=dist.ReduceOp.SUM)
        dist.all_reduce(nb, op=dist.ReduceOp.SUM)
        dist.all_reduce(has_ae, op=dist.ReduceOp.MAX)
    ce = (total_lm / nb.clamp_min(1.0)).item()
    ae_ce = (total_ae / nb.clamp_min(1.0)).item() if has_ae.item() > 0 else None
    return ce, math.exp(ce), ae_ce


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

    config["dataloader_len"] = len(train_loader)
    model.train()
    global_step = 0
    running = 0.0
    # Denominator for the running-average display: steps taken in THIS process launch.
    # global_step carries over across resumes (needed for max_steps/checkpointing), but
    # the checkpoint only stores the last train_loss, not a cumulative sum, so a true
    # all-time average can't be reconstructed — average over post-resume steps instead.
    steps_this_run = 0
    tokens_seen = 0
    target_tokens_seen = 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    t_start = time.perf_counter()

    console = Console()
    underlying_model = model.module if hasattr(model, "module") else model

    # No epoch/mid-epoch bookkeeping needed here (unlike train.py's resume): a step
    # count is all pretrain() tracks, so resuming is just restoring global_step +
    # best_ce and continuing to consume the (freshly shuffled) train_loader.
    if config.get('resume_from'):
        resume_path = config['resume_from']
        if resume_path.lower() == "latest":
            resume_path = get_latest_checkpoint(config)
        if resume_path:
            resume_info = load_checkpoint(resume_path, model, optimizer, scheduler, device, is_resume=True)
            if resume_info:
                global_step = resume_info['total_steps']
                best_ce = resume_info['best_val_loss']
                if rank == 0:
                    console.log(f"[green]Resuming from step {global_step}, best val CE {best_ce:.4f}[/green]")
        elif rank == 0:
            console.log("[yellow]Could not resolve 'latest' checkpoint or path invalid.[/yellow]")
    elif config.get('start_from'):
        start_path = config['start_from']
        if start_path.lower() == "latest":
            start_path = get_latest_checkpoint(config)
        if start_path:
            load_checkpoint(start_path, model, optimizer=None, scheduler=None, device=device, is_resume=False)
        elif rank == 0:
            console.log("[yellow]Could not resolve 'start_from' checkpoint path.[/yellow]")

    if rank == 0:
        wandb.define_metric("tokens_seen")
        wandb.define_metric("loss_vs_tokens/train_ce", step_metric="tokens_seen")
        wandb.define_metric("loss_vs_tokens/val_ce", step_metric="tokens_seen")

    try:
      with Progress(
        SpinnerColumn(), TextColumn("[bold cyan]{task.description}"), BarColumn(),
        MofNCompleteColumn(), TimeElapsedColumn(), TimeRemainingColumn(),
        TextColumn("[yellow]{task.fields[stats]}"), console=console, disable=(rank != 0),
      ) as progress:
        task_id = progress.add_task("Pretrain", total=max_steps, stats="", completed=global_step)
        if rank == 0:
            console.log(f"[green]LM pretrain · {max_steps} steps · "
                        f"eval every {eval_interval}[/green]")

        batch_iter = cycle(train_loader) if config.get("overfit") else train_loader
        for micro, batch in enumerate(batch_iter):
            ttl_tok, tgt_tok = _batch_token_counts(batch)
            tokens_seen += ttl_tok * world
            target_tokens_seen += tgt_tok * world
            if "target_inputs" in batch:
                batch_size = batch["target_inputs"]["target_mask"].shape[0]
            else:
                batch_size = batch["continuation_inputs"]["attention_mask"].shape[0]

            is_boundary = ((micro + 1) % grad_accum == 0)
            sync_ctx = nullcontext() if (is_boundary or not hasattr(model, "no_sync")) else model.no_sync()
            with sync_ctx:
                loss = _forward(model, batch, device) / grad_accum
                loss.backward()
            # Diagnostic only (mirrors src/train.py): None unless the objective blends
            # losses (ICAE's ae_lm), in which case `loss` above is the blended scalar
            # actually driving backprop — this split is logged separately below and
            # never substituted into train/ce so the optimized objective stays visible.
            step_components = getattr(underlying_model, "last_loss_components", None)

            grad_norm = None
            if is_boundary:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=config["gradient_clip_max_norm"])
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
                global_step += 1
                steps_this_run += 1

                step_loss = loss.item() * grad_accum
                running += step_loss
                progress.update(task_id, advance=1,
                                stats=f"ce={step_loss:.4f} avg={running / steps_this_run:.4f}")

                if rank == 0 and global_step % log_interval == 0:
                    elapsed = max(time.perf_counter() - t_start, 1e-9)
                    its = steps_this_run / elapsed
                    remaining_steps = max(max_steps - global_step, 0)
                    eta_min = (remaining_steps / its) / 60 if its > 0 else float("inf")
                    console.log(
                        f"step {global_step:>6}/{max_steps} · loss {step_loss:.4f} · "
                        f"avg {running / steps_this_run:.4f} · {its:.2f} it/s · "
                        f"{tokens_seen / elapsed:,.0f} tok/s · tokens seen {tokens_seen:,} · "
                        f"ETA {eta_min:.1f} min"
                    )

                    log = {
                        "train/ce": step_loss,
                        "train/avg_ce": running / steps_this_run,
                        "train/ppl": math.exp(min(step_loss, 20.0)),
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "train/step": global_step,
                        "train/tokens_seen": tokens_seen,
                        "train/target_tokens_seen": target_tokens_seen,
                        "throughput/tokens_per_sec": tokens_seen / elapsed,
                        "throughput/steps_per_sec": global_step / elapsed,
                        "data/avg_context_len": (ttl_tok - tgt_tok) / batch_size,
                        "data/avg_target_len": tgt_tok / batch_size,
                        "tokens_seen": tokens_seen,
                        "loss_vs_tokens/train_ce": step_loss,
                    }
                    if grad_norm is not None:
                        log["train/grad_norm"] = float(grad_norm)
                    if step_components is not None:
                        log["train/ae_ce"] = step_components["ae_loss"].item()
                        log["train/lm_ce"] = step_components["lm_loss"].item()
                    wandb.log(log, step=global_step)

                if global_step % eval_interval == 0:
                    val_ce, val_ppl, val_ae_ce = evaluate_lm(model, val_loader, device)
                    model.train()
                    is_best = val_ce < best_ce
                    best_ce = min(best_ce, val_ce)
                    if rank == 0:
                        marker = " · [bold green]✓ new best[/bold green]" if is_best else ""
                        console.log(
                            f"step {global_step:>6} · val_ce (lm) {val_ce:.4f} · ppl {val_ppl:8.2f}{marker}")
                        val_log = {
                            "val/ce": val_ce,
                            "val/ppl": val_ppl,
                            "val/best_ce": best_ce,
                            "val/tokens_seen": tokens_seen,
                            "tokens_seen": tokens_seen,
                            "loss_vs_tokens/val_ce": val_ce,
                        }
                        if val_ae_ce is not None:
                            val_log["val/ae_ce"] = val_ae_ce
                        wandb.log(val_log, step=global_step)
                    if is_best:
                        path = save_checkpoint(model=model, optimizer=optimizer, scheduler=scheduler,
                                        epoch=1, train_loss=running / steps_this_run, val_loss=val_ce,
                                        config=config, best_val_loss=best_ce, total_steps=global_step,
                                        is_interrupted=False, step_in_epoch=global_step, is_best=True,
                                        quiet=True)
                        if rank == 0 and path:
                            console.log(f"[dim]best checkpoint → {path}[/dim]")

                if global_step % save_interval == 0:
                    path = save_checkpoint(model=model, optimizer=optimizer, scheduler=scheduler,
                                    epoch=1, train_loss=running / steps_this_run, val_loss=float("inf"),
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
                                   epoch=1, train_loss=(running / steps_this_run if steps_this_run else 0.0),
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

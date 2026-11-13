import os
import signal
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from rich.progress import (
    Progress, BarColumn, TextColumn, TimeElapsedColumn,
    TimeRemainingColumn, MofNCompleteColumn, SpinnerColumn,
)
from colorama import Fore
from .utils.utils import save_checkpoint, load_checkpoint, get_latest_checkpoint, batch_token_counts
from .utils.artifacts import save_artifact
import torch.distributed as dist


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, config,
                base_steps=0, start_step=0, training_state=None,
                tokens_seen=0, context_tokens_seen=0):
    model.train()
    total_loss = 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    
    disabled = config.get('global_rank', 0) != 0
    skipped_steps = 0
    final_global_step = base_steps

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("[yellow]{task.fields[stats]}"),
        disable=disabled,
    ) as progress:
        task_id = progress.add_task(f"Epoch {epoch}", total=len(dataloader), stats="")

        for batch_idx, batch in enumerate(dataloader):
            # Update shared state for interruption handling
            if training_state is not None:
                training_state['step_in_epoch'] = batch_idx

            # SKIP steps if resuming mid-epoch
            if batch_idx < start_step:
                skipped_steps += 1
                if batch_idx == start_step - 1 and config.get('global_rank', 0) == 0:
                    print(f"Skipped {skipped_steps} steps to resume...")
                progress.update(task_id, advance=1)
                continue

            # Count real tokens on the CPU batch (×world for the global DDP total)
            # before moving anything to the GPU.
            ttl_tok, ctx_tok = batch_token_counts(batch)
            tokens_seen += ttl_tok * world
            context_tokens_seen += ctx_tok * world

            # question_ids (B-ctx) is a python list-of-lists, not a tensor — guard the device move.
            compressor_inputs = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                                 for k, v in batch["compressor_inputs"].items()}

            llm_inputs = {k: v.to(device, non_blocking=True) for k, v in batch["llm_inputs"].items()}

            save_artifacts = (batch_idx == 0 and config.get('artifact_every_n_epochs') is not None and (epoch - 1) % config.get('artifact_every_n_epochs') == 0)

            grad_accum = config.get('gradient_accumulation_steps', 1)
            is_last_accum = ((batch_idx + 1) % grad_accum == 0)
            # Skip DDP all-reduce on non-boundary accumulation micro-steps — a single
            # all-reduce at the boundary is mathematically equivalent and saves (K-1)/K
            # of inter-GPU communication per optimizer step.
            sync_ctx = nullcontext() if (is_last_accum or not hasattr(model, 'no_sync')) else model.no_sync()
            with sync_ctx:
                if save_artifacts:
                    loss, teacher_logits, student_logits = model(llm_inputs, compressor_inputs, return_logits=True)
                    if config['global_rank'] == 0:
                        artifact_path = f"{config['checkpoint_dir']}/artifacts/epoch_{epoch}"
                        t_logits_detached = teacher_logits.detach().cpu()
                        s_logits_detached = student_logits.detach().cpu()
                        save_artifact(t_logits_detached, s_logits_detached, config, artifact_path)
                else:
                    loss = model(llm_inputs, compressor_inputs, return_logits=False)

                # Scale loss by accumulation steps
                loss = loss / grad_accum
                loss.backward()

            grad_norm = None
            if is_last_accum:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['gradient_clip_max_norm'])
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()

            # Log unscaled loss for tracking
            training_loss = loss.item() * grad_accum
            total_loss += training_loss

            # Calculate avg loss based on ACTUAL processed steps
            steps_processed = (batch_idx + 1) - start_step
            if steps_processed > 0:
                avg_loss = total_loss / steps_processed
            else:
                avg_loss = 0

            final_global_step = base_steps + batch_idx + 1

            progress.update(task_id, advance=1, stats=f"loss={training_loss:.4f} avg={avg_loss:.4f}")

            log_interval = config.get('log_interval') or len(dataloader)
            if batch_idx % log_interval == 0 and config['global_rank'] == 0:
                log_dict = {
                    'train/loss': training_loss,
                    'train/avg_loss': avg_loss,
                    'train/epoch': epoch,
                    'train/lr': optimizer.param_groups[0]['lr'],
                    'train/tokens_seen': tokens_seen,
                    'train/context_tokens_seen': context_tokens_seen,
                    'tokens_seen': tokens_seen,
                    'loss_vs_tokens/train_loss': training_loss,
                }
                if grad_norm is not None:
                    log_dict['train/grad_norm'] = grad_norm.item() if hasattr(grad_norm, 'item') else float(grad_norm)
                wandb.log(log_dict, step=final_global_step)

            # Update loss in state for saving
            if training_state is not None:
                training_state['train_loss'] = avg_loss

    if 'avg_loss' not in locals():
        avg_loss = 0

    return avg_loss, final_global_step, tokens_seen, context_tokens_seen


def validate(config: dict, model: nn.Module, dataloader: DataLoader, device: torch.device, epoch: int) -> float:
    model.eval()
    total_loss = torch.tensor(0.0, device=device)
    num_batches = torch.tensor(0.0, device=device)
    
    disabled = config.get('global_rank', 0) != 0

    with torch.no_grad():
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold magenta]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn("[yellow]{task.fields[stats]}"),
            disable=disabled,
        ) as progress:
            task_id = progress.add_task("Validation", total=len(dataloader), stats="")

            for batch in dataloader:
                compressor_inputs = {k: (v.to(device) if torch.is_tensor(v) else v)
                                     for k, v in batch["compressor_inputs"].items()}
                llm_inputs = {k: v.to(device) for k, v in batch["llm_inputs"].items()}

                loss = model(llm_inputs, compressor_inputs, return_logits=False)

                total_loss += loss
                num_batches += 1

                local_val_loss = (total_loss / num_batches).item()
                progress.update(task_id, advance=1, stats=f"val_loss={local_val_loss:.4f}")
    
    # [DDP] Sum the losses and batch counts across all GPUs
    if dist.is_initialized():
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_batches, op=dist.ReduceOp.SUM)
        
    global_avg_loss = (total_loss / num_batches).item()
    return global_avg_loss


def train(config: dict, device: torch.device, train_loader: DataLoader, val_loader: DataLoader, 
          model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler):
    
    start_epoch = 1
    best_val_loss = float('inf')
    step_offset = 0 # This will track cumulative steps from previous runs or epochs
    start_step_in_epoch = 0 # Resume from specific step
    patience = config.get('patience', 10)
    patience_counter = 0

    if config.get('resume_from'):
        resume_path = config['resume_from']
        if resume_path.lower() == "latest":
             resume_path = get_latest_checkpoint(config)
        
        if resume_path:
            resume_info = load_checkpoint(resume_path, model, optimizer, scheduler, device, is_resume=True)
            if resume_info:
                start_epoch = resume_info['start_epoch']
                best_val_loss = resume_info['best_val_loss']
                
                total_steps_in_ckpt = resume_info['total_steps']
                start_step_in_epoch = resume_info.get('step_in_epoch', 0)
                
                if start_step_in_epoch > 0:
                    # We are resuming mid-epoch
                    step_offset = total_steps_in_ckpt - start_step_in_epoch
                else:
                    # We are starting a fresh epoch (or resumed from end of epoch)
                    step_offset = total_steps_in_ckpt
                
                print(f"{Fore.GREEN}Resuming from epoch {Fore.BLUE}{start_epoch}{Fore.RESET}, step {Fore.BLUE}{start_step_in_epoch}{Fore.RESET}, best val loss: {Fore.BLUE}{best_val_loss:.4f}{Fore.RESET}")
                print(f"{Fore.GREEN}Global step offset initialized to: {Fore.BLUE}{step_offset}{Fore.RESET}")
        else:
            print(f"{Fore.YELLOW}Could not resolve 'latest' checkpoint or path invalid.{Fore.RESET}")
    elif config.get('start_from'):
        start_path = config['start_from']
        if start_path.lower() == "latest":
             start_path = get_latest_checkpoint(config)
             
        if start_path:
            print(f"{Fore.GREEN}Initializing model weights from checkpoint: {Fore.BLUE}{start_path}{Fore.RESET}")
            load_checkpoint(start_path, model, optimizer=None, scheduler=None, device=device, is_resume=False)
        else:
            print(f"{Fore.YELLOW}Could not resolve 'start_from' checkpoint path.{Fore.RESET}")

    # Pass dataloader length to config for saving in utils
    config['dataloader_len'] = len(train_loader)
    
    print(f"\n{Fore.GREEN}Starting training...")
    print(f"{Fore.GREEN}Artifacts: {Fore.BLUE}{'Enabled' if config.get('artifact_every_n_epochs', False) else f'{Fore.RED}Disabled{Fore.BLUE}'}{Fore.RESET}")
    print(f"{Fore.GREEN}Starting epoch: {start_epoch}, Base global step: {step_offset}{Fore.RESET}")

    # Cumulative real-token counters across ALL epochs (the point: total tokens
    # consumed over a multi-epoch run). Reset to 0 on a fresh resume.
    tokens_seen = 0
    context_tokens_seen = 0
    if config.get('global_rank', 0) == 0:
        wandb.define_metric("tokens_seen")
        wandb.define_metric("loss_vs_tokens/train_loss", step_metric="tokens_seen")

    for epoch in range(start_epoch, config['num_epochs'] + 1):
        # [DDP] Essential for shuffling data differently each epoch.
        # DataLoader always auto-wraps a plain `sampler=` in a torch BatchSampler
        # (which has no set_epoch), so `batch_sampler or sampler` would wrongly pick
        # the auto-wrapper and silently skip the real sampler's set_epoch. Custom
        # batch samplers (e.g. token-budget packing) are NOT torch.utils.data.BatchSampler,
        # so checking that type tells the two cases apart.
        if isinstance(train_loader.batch_sampler, torch.utils.data.BatchSampler):
            epoch_obj = train_loader.sampler
        else:
            epoch_obj = train_loader.batch_sampler
        if hasattr(epoch_obj, 'set_epoch'):
            epoch_obj.set_epoch(epoch)

        print(f"\n{'='*50}\nEpoch {epoch}/{config['num_epochs']}\n{'='*50}")
        
        # If resuming mid-epoch, only run the remaining portion
        current_start_step = start_step_in_epoch if epoch == start_epoch else 0
        
        # Track state for interruption handling
        training_state = {'step_in_epoch': current_start_step, 'train_loss': 0.0}
        
        try:
            train_loss, total_steps, tokens_seen, context_tokens_seen = train_epoch(
                model, train_loader, optimizer, scheduler, device, epoch, config,
                base_steps=step_offset, start_step=current_start_step, training_state=training_state,
                tokens_seen=tokens_seen, context_tokens_seen=context_tokens_seen
            )
        except KeyboardInterrupt:
            # Mask SIGINT on rank 0 immediately — torchrun fires a second SIGINT the
            # moment the sibling rank exits, which would abort the save if we wait.
            old_handler = signal.signal(signal.SIGINT, signal.SIG_IGN) if config.get('global_rank', 0) == 0 else None
            try:
                if config.get('global_rank', 0) == 0:
                    print(f"\n{Fore.RED}KeyboardInterrupt detected! Saving checkpoint...{Fore.RESET}")

                current_batch_idx = training_state['step_in_epoch']
                current_total_steps = step_offset + current_batch_idx + 1
                current_loss = training_state['train_loss']

                if config.get('global_rank', 0) == 0:
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        train_loss=current_loss,
                        val_loss=float('inf'),
                        config=config,
                        best_val_loss=best_val_loss,
                        total_steps=current_total_steps,
                        is_interrupted=True,
                        step_in_epoch=current_batch_idx
                    )
            except Exception as save_err:
                if config.get('global_rank', 0) == 0:
                    print(f"{Fore.RED}Failed to save interrupted checkpoint: {save_err}{Fore.RESET}")
                    import traceback; traceback.print_exc()
            finally:
                if old_handler is not None:
                    signal.signal(signal.SIGINT, old_handler)
            raise

        step_offset = total_steps
        start_step_in_epoch = 0 
        
        val_loss = validate(config, model, val_loader, device, epoch)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 1. Update best loss tracking outside the Rank 0 guard so all ranks agree on the state
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        # 2. Call save_checkpoint globally (it has an internal barrier and rank check)
        is_save_epoch = config.get('save_interval') and epoch % config.get('save_interval') == 0
        
        if is_best or is_save_epoch:
            save_checkpoint(
                model=model, optimizer=optimizer, scheduler=scheduler, 
                epoch=epoch, train_loss=train_loss, val_loss=val_loss, 
                config=config, best_val_loss=best_val_loss, total_steps=total_steps, 
                is_interrupted=False, step_in_epoch=0, is_best=is_best
            )

        # 3. Handle Rank 0 only logging and printing
        if config['global_rank'] == 0:
            print(f"\nTrain: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {current_lr:.6f}")
            wandb.log({
                'train/lr': current_lr,
                'epoch': epoch,
                'val/loss': val_loss,
                'val/epoch': epoch
            }, step=total_steps)

            if is_best:
                wandb.log({'best_val_loss': best_val_loss, 'best_epoch': epoch}, step=total_steps)
            else:
                print(f"\n{Fore.YELLOW}Early stopping counter: {patience_counter}/{patience}{Fore.RESET}")
                
        # 4. Check for early stopping globally so all ranks exit together
        if patience_counter >= patience:
            if config['global_rank'] == 0:
                print(f"\n{Fore.YELLOW}Early stopping triggered! No improvement for {patience} epochs.{Fore.RESET}")
            break

    if config['global_rank'] == 0:
        print(f"\n{'='*50}\nTraining completed! Best val loss: {best_val_loss:.4f}\n{'='*50}")
        wandb.finish()
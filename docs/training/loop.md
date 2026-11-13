# Training Loop

## Overview and Entry Point

Training is executed via `main.py` which initializes DDP environment variables, loads model and dataloaders, and calls `train(...)` in `src/train.py`.

```bash
torchrun --nproc_per_node=4 main.py --config configs/train/config_01Comp_QC_Mamba2L_bctx.yaml
```

---

## Distributed Data Parallel (DDP) Setup

### Initialization and Synchronization

DDP process group initialization occurs in `src/utils/utils.py:init_main()`:

```python
dist.init_process_group(backend="nccl")
torch.cuda.set_device(local_rank)
```

Within `src/train.py`:
- `world = dist.get_world_size() if dist.is_initialized() else 1` accounts for total active processes.
- Non-tensor items in batch dictionaries (e.g., Python lists like `question_ids` in B-ctx mode) are guarded during GPU transfer:
  ```python
  compressor_inputs = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                       for k, v in batch["compressor_inputs"].items()}
  ```
- Validation losses are aggregated across all DDP ranks using `dist.all_reduce`:
  ```python
  if dist.is_initialized():
      dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
      dist.all_reduce(num_batches, op=dist.ReduceOp.SUM)
  global_avg_loss = (total_loss / num_batches).item()
  ```

---

## Gradient Accumulation and DDP No-Sync Optimization

Gradient accumulation steps are specified by `config['gradient_accumulation_steps']`.

To prevent redundant inter-GPU all-reduce communication on non-boundary micro-steps, the training loop uses `model.no_sync()`:

```python
grad_accum = config.get('gradient_accumulation_steps', 1)
is_last_accum = ((batch_idx + 1) % grad_accum == 0)

# Disable gradient all-reduce during intermediate accumulation steps
sync_ctx = nullcontext() if (is_last_accum or not hasattr(model, 'no_sync')) else model.no_sync()
with sync_ctx:
    loss = model(llm_inputs, compressor_inputs, return_logits=False)
    loss = loss / grad_accum
    loss.backward()

if is_last_accum:
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['gradient_clip_max_norm'])
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if scheduler is not None:
        scheduler.step()
```

This optimization reduces DDP gradient synchronization overhead by $(K-1)/K$ per optimizer step for accumulation factor $K$.

---

## Precision and AMP Status

Currently, standard forward and backward operations in `src/train.py` execute directly in the configured model precision (e.g. `bfloat16` or `float32` as loaded). Explicit `torch.cuda.amp.GradScaler` wrapper objects are not used, as modern bfloat16 training on recent GPUs does not require dynamic loss scaling.

---

## Dataset Iteration and Sampler Epoch Syncing

At the start of each epoch, the dataset sampler must be notified of the current epoch to re-seed shuffling. Because custom batch samplers (such as `DistributedTokenBudgetSampler`) wrap indices differently than standard PyTorch `BatchSampler` instances, `src/train.py` detects the sampler type explicitly:

```python
if isinstance(train_loader.batch_sampler, torch.utils.data.BatchSampler):
    epoch_obj = train_loader.sampler
else:
    epoch_obj = train_loader.batch_sampler

if hasattr(epoch_obj, 'set_epoch'):
    epoch_obj.set_epoch(epoch)
```

---

## Token Counting

Real token counts are accumulated across all DDP ranks:

```python
ttl_tok, ctx_tok = batch_token_counts(batch)
tokens_seen += ttl_tok * world
context_tokens_seen += ctx_tok * world
```

Both total tokens and context tokens are logged to W&B as custom step metrics for loss vs. token tracking.

---

## Checkpointing and Mid-Epoch Resume

### Saving Checkpoints (`src/utils/utils.py:save_checkpoint`)

Checkpoints save the compressor backbone state dict, optimizer state, scheduler state, current epoch, step indices, and metrics:

```python
state = {
    'model': core.compressor.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict(),
    'epoch': epoch,
    'step_in_epoch': step_in_epoch,
    'total_steps': total_steps,
    'best_val_loss': best_val_loss,
    'config': config,
    'compressor_type': config['compressor_type'],
}
```

Saving is atomic: written to a `.tmp` file before renaming. Only Rank 0 writes files, while all ranks synchronize at DDP barriers.

### Resuming Workflows (`train()` in `src/train.py`)

1. **Full Resume (`resume_from`):** Restores model weights, optimizer, scheduler, `start_epoch`, `step_in_epoch`, and global step offsets.
2. **Weights Only (`start_from`):** Loads model weights from a specified or latest checkpoint while starting epoch count and step counts from scratch.
3. **Mid-Epoch Skipping:** When resuming mid-epoch, `train_epoch` iterates through `dataloader` and skips steps where `batch_idx < start_step`:
   ```python
   if batch_idx < start_step:
       skipped_steps += 1
       progress.update(task_id, advance=1)
       continue
   ```

---

## Keyboard Interrupt (`SIGINT`) Recovery

If a `KeyboardInterrupt` occurs during `train_epoch`:

1. Rank 0 immediately ignores further `SIGINT` signals (`signal.signal(signal.SIGINT, signal.SIG_IGN)`) to prevent `torchrun` child termination races from interrupting save operations.
2. An emergency interrupted checkpoint (`interrupted_model.pt`) is written with current `step_in_epoch` and `train_loss`.
3. Signal handlers are restored, and the exception is re-raised for clean shutdown.

---

## Early Stopping and Validation

Validation runs globally via `validate(...)`. All processes participate in validation forward passes and loss all-reduction:

```python
is_best = val_loss < best_val_loss
if is_best:
    best_val_loss = val_loss
    patience_counter = 0
else:
    patience_counter += 1

if patience_counter >= patience:
    if config['global_rank'] == 0:
        print(f"Early stopping triggered! No improvement for {patience} epochs.")
    break
```

Because `val_loss` is synchronized via `dist.all_reduce`, all ranks reach identical early stopping decisions simultaneously.

---

## WandB Metric Logging

Rank 0 logs metrics to Weights & Biases at regular `log_interval` step intervals and after each validation pass:

```python
if batch_idx % log_interval == 0 and config['global_rank'] == 0:
    wandb.log({
        'train/loss': training_loss,
        'train/avg_loss': avg_loss,
        'train/epoch': epoch,
        'train/lr': optimizer.param_groups[0]['lr'],
        'train/tokens_seen': tokens_seen,
        'train/context_tokens_seen': context_tokens_seen,
        'tokens_seen': tokens_seen,
        'loss_vs_tokens/train_loss': training_loss,
        'train/grad_norm': grad_norm,
    }, step=final_global_step)
```

---

## Artifact Generation

When `artifact_every_n_epochs` is configured, `train_epoch` triggers `model(..., return_logits=True)` on batch 0 of scheduled epochs. Rank 0 detaches teacher and student logit tensors to CPU and saves them via `save_artifact(...)` under `{checkpoint_dir}/artifacts/epoch_{epoch}`.

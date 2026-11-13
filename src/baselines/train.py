"""Full fine-tune a Qwen3 teacher on MRQA with HuggingFace Trainer. DDP is automatic
via torchrun (Trainer reads WORLD_SIZE/LOCAL_RANK). Final model saved to
`paths.final_checkpoint_dir`, consumed by `src.baselines.dump_logits`.

    torchrun --nproc_per_node=<N> -m src.baselines.train --config <cfg.yaml>
"""
import os
import math
import time
import signal
import argparse

import torch
import wandb
import torch.distributed as dist
from rich.console import Console
from rich.table import Table
from rich.progress import (
    Progress, BarColumn, TextColumn, MofNCompleteColumn,
    TimeElapsedColumn, TimeRemainingColumn,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_callback import PrinterCallback

from src.dataset import load_mrqa_dataset
from src.baselines.dataset import MRQASFTDataset, build_sft_collate
from src.baselines.utils import (
    is_main_process, make_console, load_config, quiet_external_logging,
    setup_distributed, resolve_resume, align_model_tokens,
)

console = make_console()


def log(msg: str, style: str = "green"):
    console.print(msg, style=style)


class SaveOnSignalCallback(TrainerCallback):
    """1st SIGINT/SIGTERM -> checkpoint + stop at the next step (resumable).
    2nd -> abort immediately without saving (os._exit bypasses Trainer cleanup)."""

    def __init__(self):
        self._stop = False
        signal.signal(signal.SIGINT, self._handle)
        signal.signal(signal.SIGTERM, self._handle)

    def _handle(self, signum, frame):
        if not self._stop:
            self._stop = True
            log(f"Signal {signum} — checkpointing and stopping after this step. "
                "Press Ctrl-C again to abort NOW without saving.", "bold red")
        else:
            log("Second signal — aborting immediately, NOT saving.", "bold red")
            os._exit(130)  # 128 + SIGINT; skips save/cleanup on every rank

    def on_step_end(self, args, state, control, **kwargs):
        if self._stop:
            control.should_save = True
            control.should_training_stop = True
        return control


class RichProgressCallback(TrainerCallback):
    """Per-epoch progress bar + one summary line per epoch (rank 0). Replaces PrinterCallback."""

    def __init__(self, cons: Console, eval_total: int | None = None):
        self.c = cons
        self.progress = None
        self.task = None
        self.eval_task = None
        self.eval_total = eval_total  # eval batches per process (for the validation bar)
        self.t0 = None
        self.metrics = {}

    def _epoch(self, state, args):
        return int(state.epoch or 0) + 1, int(args.num_train_epochs)

    def on_train_begin(self, args, state, control, **kwargs):
        self.progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("{task.fields[info]}"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=self.c,
            transient=True,
        )
        self.progress.start()

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.t0 = time.time()
        per_epoch = max(1, round(state.max_steps / max(args.num_train_epochs, 1)))
        e, total = self._epoch(state, args)
        desc = f"epoch {e}/{total}"
        if self.task is None:
            self.task = self.progress.add_task(desc, total=per_epoch, info="")
        else:
            self.progress.reset(self.task, total=per_epoch, description=desc, info="")

    def on_step_end(self, args, state, control, **kwargs):
        if self.task is not None:
            self.progress.advance(self.task, 1)

    def on_prediction_step(self, args, state, control, **kwargs):
        # Eval has no per-step Trainer logging -> the train bar looks frozen while the
        # full dev set is scored. Show a separate validation bar that advances per batch.
        if self.progress is None:
            return
        if self.eval_task is None:
            self.eval_task = self.progress.add_task(
                "validating", total=self.eval_total, info="")
        self.progress.advance(self.eval_task, 1)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        self.metrics.update(logs)
        if self.task is not None and "loss" in logs:
            info = f"loss={logs['loss']:.3f}"
            if "grad_norm" in logs:
                info += f" gn={logs['grad_norm']:.2f}"
            if "learning_rate" in logs:
                info += f" lr={logs['learning_rate']:.1e}"
            self.progress.update(self.task, info=info)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if self.eval_task is not None:  # eval done -> drop the validation bar
            self.progress.remove_task(self.eval_task)
            self.eval_task = None
        m = metrics or {}
        dt = time.time() - (self.t0 or time.time())
        # Header reports step + fractional epoch — correct under both step- and epoch-based eval.
        parts = [f"[bold]step {state.global_step}[/] (epoch {state.epoch:.2f})"]
        if "loss" in self.metrics:
            parts.append(f"train_loss={self.metrics['loss']:.4f}")
        if "eval_loss" in m:
            parts.append(f"eval_loss={m['eval_loss']:.4f}")
        if "learning_rate" in self.metrics:
            parts.append(f"lr={self.metrics['learning_rate']:.2e}")
        if "grad_norm" in self.metrics:
            parts.append(f"grad_norm={self.metrics['grad_norm']:.2f}")
        parts.append(f"{dt:.0f}s")
        self.c.log("  ".join(parts))

    def on_train_end(self, args, state, control, **kwargs):
        if self.progress is not None:
            self.progress.stop()
            self.progress = None
        best = getattr(state, "best_metric", None)
        if best is not None:
            self.c.log(f"[green]Best {args.metric_for_best_model} = {best:.4f}[/]")


class LoggingEarlyStoppingCallback(EarlyStoppingCallback):
    """EarlyStoppingCallback that announces (console + wandb) when it fires."""

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        was_stopping = control.should_training_stop
        super().on_evaluate(args, state, control, metrics=metrics, **kwargs)
        if control.should_training_stop and not was_stopping:
            log(f"⏹  EARLY STOPPING at step {state.global_step} (epoch {state.epoch:.2f}): "
                f"no improvement on {args.metric_for_best_model} for "
                f"{self.early_stopping_patience} evals.", "bold yellow")
            if is_main_process() and wandb.run is not None:
                wandb.log({"early_stopping/triggered": 1,
                           "early_stopping/epoch": state.epoch,
                           "early_stopping/step": state.global_step})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    quiet_external_logging()  # we render our own compact output (rank 0 only)
    setup_distributed()       # pre-init NCCL with device_id; Trainer reuses it (mutes barrier warning)

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = dict(cfg["training"])
    paths_cfg = cfg["paths"]
    wandb_cfg = cfg.get("wandb", {})

    model_name = model_cfg["llm_model_name"]
    max_seq_len = model_cfg["max_seq_len"]

    if wandb_cfg.get("project"):
        os.environ.setdefault("WANDB_PROJECT", wandb_cfg["project"])
    run_name = wandb_cfg.get("name")

    # resume_from / patience / start_from are not TrainingArguments fields — pop first.
    resume_from = train_cfg.pop("resume_from", None)
    patience = train_cfg.pop("patience", None)
    start_from = train_cfg.pop("start_from", None)  # weights-only warm start
    output_dir = train_cfg["output_dir"]
    resume_ckpt = resolve_resume(resume_from, output_dir)

    # Resume the same wandb run on restart (stable id from run_name).
    if run_name:
        os.environ.setdefault("WANDB_RUN_ID", run_name.replace(" ", "_"))
        if resume_ckpt:
            os.environ.setdefault("WANDB_RESUME", "allow")

    # resume restores weights from the checkpoint; start_from ignored when resuming.
    model_src = model_name if resume_ckpt else (start_from or model_name)
    log(f"Loading tokenizer + model: {model_src}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_src, torch_dtype=torch.bfloat16)
    align_model_tokens(model, tokenizer)  # pre-align pad/bos -> no auto-align warning

    # train/ and dev/ are subdirs under data_dir (mirrors main.py).
    data_dir = data_cfg["data_dir"]
    train_dir = os.path.join(data_dir, "train")
    dev_dir = data_cfg.get("dev_dir") or os.path.join(data_dir, "dev")
    seed = data_cfg.get("seed", 42)
    max_samples = data_cfg.get("max_samples")
    max_eval_samples = data_cfg.get("max_eval_samples")  # cap dev independently of train
    use_context = data_cfg.get("use_context", True)

    log(f"Loading MRQA train split from {train_dir}")
    train_raw = load_mrqa_dataset(train_dir, seed=seed)
    log(f"Loading MRQA dev split from {dev_dir}")
    eval_raw = load_mrqa_dataset(dev_dir, seed=seed)
    n_train_full = len(train_raw)
    if max_samples:
        train_raw = train_raw.shuffle(seed=seed).select(range(min(max_samples, n_train_full)))
    # max_eval_samples wins; else fall back to scaling eval with the train cap (legacy behavior).
    eval_cap = max_eval_samples or (max(1, round(max_samples * len(eval_raw) / n_train_full))
                                    if max_samples else None)
    if eval_cap:
        eval_raw = eval_raw.shuffle(seed=seed).select(range(min(eval_cap, len(eval_raw))))

    train_ds = MRQASFTDataset(train_raw, tokenizer, max_length=max_seq_len, use_context=use_context)
    eval_ds = MRQASFTDataset(eval_raw, tokenizer, max_length=max_seq_len, use_context=use_context)

    training_args = TrainingArguments(
        run_name=run_name,
        seed=seed,
        **train_cfg,
    )

    # Compact run summary (rank 0).
    world = int(os.environ.get("WORLD_SIZE", 1))
    eff_batch = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * world
    summary = Table(show_header=False, box=None, pad_edge=False)
    summary.add_column(style="dim")
    summary.add_column(style="bold")
    summary.add_row("model", model_src)
    summary.add_row("train / eval samples", f"{len(train_ds)} / {len(eval_ds)}")
    summary.add_row("effective batch", f"{eff_batch}  ({training_args.per_device_train_batch_size}×{training_args.gradient_accumulation_steps}×{world} gpu)")
    summary.add_row("epochs / lr", f"{int(training_args.num_train_epochs)} / {training_args.learning_rate:.1e}")
    summary.add_row("max_seq_len / clip", f"{max_seq_len} / {training_args.max_grad_norm}")
    if resume_ckpt:
        summary.add_row("resume", resume_ckpt)
    console.print(summary)

    # Eval batches per process (DDP splits the dev set across ranks) -> validation bar total.
    eval_total = math.ceil(len(eval_ds) / (training_args.per_device_eval_batch_size * world))
    callbacks = [SaveOnSignalCallback(), RichProgressCallback(console, eval_total=eval_total)]
    if patience:
        callbacks.append(LoggingEarlyStoppingCallback(early_stopping_patience=patience))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=build_sft_collate(tokenizer, max_seq_len),
        processing_class=tokenizer,
        callbacks=callbacks,
    )
    trainer.remove_callback(PrinterCallback)  # our RichProgressCallback owns console output

    log("Starting training" + (f" (resuming from {resume_ckpt})" if resume_ckpt else ""))
    trainer.train(resume_from_checkpoint=resume_ckpt)

    final_dir = paths_cfg["final_checkpoint_dir"]
    log(f"Saving final model to {final_dir}")
    trainer.save_model(final_dir)
    if is_main_process():
        tokenizer.save_pretrained(final_dir)
    log("Done.")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

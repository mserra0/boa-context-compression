"""GPU memory profiling helpers for diagnosing OOMs in the pretrain loop.

Two complementary, opt-in tools (both no-ops unless explicitly enabled, so normal
runs pay zero cost):

  - `StageMemoryProfiler` — cheap, always-on-when-enabled. Prints allocated/reserved/
    peak deltas at named checkpoints inside a single forward/backward pass. Answers
    "roughly which stage (Mamba2 backbone / kv_projector / LLM forward / backward)
    grows memory, and by how much".

  - `SnapshotRecorder` — heavier, run for a handful of steps only. Wraps
    torch.cuda.memory._record_memory_history / _dump_snapshot to produce a .pickle
    with the full allocation-site timeline, loadable at https://pytorch.org/memory_viz.
    Answers "which exact tensor/call is occupying memory" when the stage breakdown
    isn't precise enough.
"""
import os

import torch
import torch.distributed as dist


def _is_main_process() -> bool:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return os.environ.get("RANK", "0") == "0"


def unwrap_model(model):
    """DDP-wrapped modules hide the real model under `.module`."""
    return model.module if hasattr(model, "module") else model


class StageMemoryProfiler:
    """Call `.reset()` once per training step, then `.stage(tag)` at each checkpoint
    inside forward/backward to print incremental CUDA memory usage. No-op if
    `enabled=False`, no CUDA, or not rank 0 (avoids log spam under DDP)."""

    def __init__(self, enabled: bool = False, device: torch.device = None):
        self.enabled = bool(enabled) and torch.cuda.is_available()
        self.device = device
        self._last_allocated = 0.0

    def reset(self):
        if not self.enabled:
            return
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        self._last_allocated = torch.cuda.memory_allocated(self.device) / 1e6

    def stage(self, tag: str):
        if not self.enabled or not _is_main_process():
            return
        torch.cuda.synchronize(self.device)
        allocated = torch.cuda.memory_allocated(self.device) / 1e6
        reserved = torch.cuda.memory_reserved(self.device) / 1e6
        peak = torch.cuda.max_memory_allocated(self.device) / 1e6
        delta = allocated - self._last_allocated
        print(f"[mem] {tag:<28} allocated={allocated:9.1f}MB "
              f"(delta {delta:+9.1f}MB)  reserved={reserved:9.1f}MB  peak={peak:9.1f}MB",
              flush=True)
        self._last_allocated = allocated


class SnapshotRecorder:
    """Records a detailed CUDA allocator history for steps [start_step, stop_step]
    and dumps it to `path`. Call `.maybe_start(step)` before the step's forward and
    `.maybe_stop_and_dump(step)` after the step's optimizer.step()."""

    def __init__(self, enabled: bool, start_step: int, stop_step: int, path: str,
                 max_entries: int = 200_000):
        self.enabled = bool(enabled) and torch.cuda.is_available() and _is_main_process()
        self.start_step = start_step
        self.stop_step = stop_step
        self.path = path
        self.max_entries = max_entries
        self._recording = False

    def maybe_start(self, step: int):
        if self.enabled and not self._recording and step == self.start_step:
            torch.cuda.memory._record_memory_history(max_entries=self.max_entries)
            self._recording = True
            print(f"[mem] recording detailed allocator history from step {step}...", flush=True)

    def maybe_stop_and_dump(self, step: int):
        if self.enabled and self._recording and step >= self.stop_step:
            out_dir = os.path.dirname(self.path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            torch.cuda.memory._dump_snapshot(self.path)
            torch.cuda.memory._record_memory_history(enabled=None)
            self._recording = False
            print(f"[mem] snapshot dumped -> {self.path} "
                  f"(load at https://pytorch.org/memory_viz)", flush=True)

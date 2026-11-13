"""Sweep loop + timing for the latency benchmark.

Per (adapter, seq_len) cell it times three stages with CUDA-synced wall clock:

    t_compress : encoder cost (SSM forward for `ssm`, ~0 for `full_ctx`)
    t_ttft     : prefill + first token   (generate with max_new_tokens=1)
    t_decode   : t_total_gen - t_ttft    (amortised per-token decode)
    t_total    : t_compress + t_total_gen (end-to-end, the headline number)

Each stage is measured over `warmup` discarded runs + `repeats` timed runs, and
reported as the median (robust to scheduler noise). Peak GPU memory is captured
per cell. A CUDA OOM records NaN and continues -- the baseline OOMing before our
method is itself a result worth logging.
"""

import gc
import math
import time

import torch

from .adapters import ADAPTERS


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def _median(xs):
    return float("nan") if not xs else sorted(xs)[len(xs) // 2]


def _time_ms(fn, device, warmup, repeats):
    """Median wall-clock ms of fn() over `repeats` timed runs after `warmup`."""
    for _ in range(warmup):
        fn()
    _sync(device)
    samples = []
    for _ in range(repeats):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return _median(samples)


def _run_cell(model, adapter, ctx, seq_len, max_new_tokens, warmup, repeats):
    device = ctx["device"]
    prepare, compress, generate = adapter["prepare"], adapter["compress"], adapter["generate"]

    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    state = prepare(model, ctx, seq_len)

    # gen_state is deterministic in shape and not mutated by generate(), so we
    # build it once and reuse it across both the compress and generate timings.
    gen_state = compress(model, state)

    t_compress = _time_ms(lambda: compress(model, state), device, warmup, repeats)
    t_ttft = _time_ms(lambda: generate(model, gen_state, 1), device, warmup, repeats)
    t_gen_full = _time_ms(lambda: generate(model, gen_state, max_new_tokens), device, warmup, repeats)

    gen_tokens = generate(model, gen_state, max_new_tokens)
    t_decode = t_gen_full - t_ttft

    # Generating max_new_tokens cannot be faster than generating one, so a
    # negative decode means the two stages were not measured under comparable
    # conditions -- host-load jitter hitting a partly CPU-bound method, or a
    # method that early-stops. The number is unusable either way, so mark the
    # row rather than let a negative decode reach a plot.
    suspect = t_decode < 0
    peak_mem_mb = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        if torch.device(device).type == "cuda" else float("nan")
    )

    return {
        "adapter": None,  # filled by caller
        "seq_len": seq_len,
        "compressed_tokens": adapter["compressed_tokens"](gen_state, seq_len),
        "gen_tokens": gen_tokens,
        "t_compress_ms": t_compress,
        "t_ttft_ms": t_ttft,
        "t_decode_ms": t_decode,
        "t_total_ms": t_compress + t_gen_full,
        "peak_mem_mb": peak_mem_mb,
        "oom": False,
        "suspect": suspect,
    }


def _global_warmup(model, adapter, ctx, seq_len, max_new_tokens):
    """One untimed full pass so one-time CUDA/cuBLAS/kernel-autotune allocations
    happen OUTSIDE any cell's measurement window. Without this the first measured
    cell absorbs those costs and reports inflated peak memory."""
    with torch.no_grad():
        state = adapter["prepare"](model, ctx, seq_len)
        gen_state = adapter["compress"](model, state)
        adapter["generate"](model, gen_state, 1)
        adapter["generate"](model, gen_state, max_new_tokens)


def _group_by_loader(adapter_names):
    """Adapters that share a model, keyed by loader in first-appearance order."""
    groups = {}
    for name in adapter_names:
        groups.setdefault(ADAPTERS[name]["loader"], []).append(name)
    return groups


def run_sweep(loaders, bench_cfg, log=print, on_row=None):
    """Run every (adapter, seq_len) cell. Returns a list of result-row dicts.

    `loaders` maps a loader name to a zero-arg callable returning `(model, ctx)`.
    Only one loader's model is held at a time: methods bring their own backbone
    (Activation Beacon does), and a competing model resident on the same device
    would inflate every `peak_mem_mb` reading.

    `on_row` is called with each completed row. Only OOM is recoverable: any
    other CUDA fault (mamba2's Triton kernels raise an illegal memory access
    once a packed sequence overflows int32 indexing) poisons the context, so
    every later cell on that device would be garbage and the sweep must die.
    Streaming rows out as they complete means a fault costs one cell, not the
    whole run.
    """
    rows = []
    max_new_tokens = bench_cfg["max_new_tokens"]
    warmup = bench_cfg.get("warmup", 2)
    repeats = bench_cfg.get("repeats", 5)
    min_seq_len = min(bench_cfg["seq_lens"])

    for loader_name, names in _group_by_loader(bench_cfg["adapters"]).items():
        model, ctx = loaders[loader_name]()
        is_cuda = torch.device(ctx["device"]).type == "cuda"
        try:
            for name in names:
                adapter = ADAPTERS[name]
                _global_warmup(model, adapter, ctx, min_seq_len, max_new_tokens)
                if is_cuda:
                    torch.cuda.empty_cache()
                for seq_len in bench_cfg["seq_lens"]:
                    try:
                        with torch.no_grad():
                            row = _run_cell(model, adapter, ctx, seq_len, max_new_tokens,
                                            warmup, repeats)
                        row["adapter"] = name
                        flag = "  [SUSPECT: negative decode]" if row["suspect"] else ""
                        log(f"  {name:>9} L={seq_len:>6}  "
                            f"compress={row['t_compress_ms']:.1f}ms  ttft={row['t_ttft_ms']:.1f}ms  "
                            f"total={row['t_total_ms']:.1f}ms  mem={row['peak_mem_mb']:.0f}MB{flag}")
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        row = {
                            "adapter": name, "seq_len": seq_len, "compressed_tokens": math.nan,
                            "gen_tokens": math.nan, "t_compress_ms": math.nan,
                            "t_ttft_ms": math.nan, "t_decode_ms": math.nan,
                            "t_total_ms": math.nan, "peak_mem_mb": math.nan, "oom": True,
                            "suspect": False,
                        }
                        log(f"  {name:>9} L={seq_len:>6}  OOM")
                    rows.append(row)
                    if on_row is not None:
                        on_row(row)
                    if is_cuda:
                        torch.cuda.empty_cache()
        finally:
            del model
            gc.collect()
            if is_cuda:
                # A poisoned context makes this raise too; the original fault is
                # what matters, so do not let cleanup mask it.
                try:
                    torch.cuda.empty_cache()
                except RuntimeError:
                    pass
    return rows

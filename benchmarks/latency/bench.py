"""CLI entry point for the end-to-end latency benchmark.

Measures Compression Time + Generation Time across sequence lengths, contrasting
our linear-time SSM compressor against feeding the full context to the frozen LLM.

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python -m benchmarks.latency.bench \
        --config configs/train/config_01CompDIST.yaml \
        --checkpoint /path/to/checkpoint \
        --bench-config benchmarks/latency/config.yaml
"""

import argparse
import json
import os
import sys

import torch
import yaml
from colorama import Fore, Style, init as colorama_init

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils.inference_utils import load_resources
from benchmarks.latency.beacon import load_beacon_model
from benchmarks.latency.icae import load_icae_model
from benchmarks.latency.runner import run_sweep
from benchmarks.latency.sac import load_sac_model

colorama_init(autoreset=True)


def _log(msg):
    print(msg, flush=True)


def _build_ctx(model, llm_tokenizer, compressor_tokenizer, device, question_len):
    pid = model.llm.config.pad_token_id
    pad_token_id = pid if pid is not None else model.llm.config.eos_token_id
    if isinstance(pad_token_id, (list, tuple)):
        pad_token_id = pad_token_id[0]
    return {
        "device": device,
        "llm_vocab": len(llm_tokenizer),
        "compressor_vocab": len(compressor_tokenizer),
        "pad_token_id": pad_token_id,
        "question_len": question_len,
    }


def _open_results(out_path, meta):
    """Open the results file and write the meta line.

    Rows are appended and flushed as they complete rather than written at the
    end: a fault in one cell (a 256k mamba2 packed sequence overflows int32
    indexing in the Triton kernel and kills the process) must not discard the
    hours of cells that already succeeded.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    handle = open(out_path, "w")
    handle.write(json.dumps({"meta": meta}) + "\n")
    handle.flush()
    return handle


def _beacon_window(checkpoint_dir):
    """Beacon window size from a checkpoint's backend metadata, or None."""
    if not checkpoint_dir:
        return None
    path = os.path.join(checkpoint_dir, "activation_beacon_backend.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f).get("beacon_window")


def _print_table(rows):
    hdr = f"{'adapter':>9} {'L':>7} {'C':>6} {'compress':>10} {'ttft':>9} {'decode':>9} {'total':>10} {'mem_MB':>9}"
    print(f"\n{Fore.CYAN}{hdr}{Style.RESET_ALL}")
    print("-" * len(hdr))
    for r in rows:
        if r["oom"]:
            print(f"{r['adapter']:>9} {r['seq_len']:>7} {'OOM':>6}")
            continue
        print(f"{r['adapter']:>9} {r['seq_len']:>7} {r['compressed_tokens']:>6} "
              f"{r['t_compress_ms']:>9.1f} {r['t_ttft_ms']:>8.1f} {r['t_decode_ms']:>8.1f} "
              f"{r['t_total_ms']:>9.1f} {r['peak_mem_mb']:>9.0f}")


def _load_boa(args):
    """Our compressor + frozen LLM, from a training checkpoint."""
    if not (args.config and args.checkpoint):
        raise SystemExit("--config and --checkpoint are required for the ssm/full_ctx adapters.")
    print(f"{Fore.YELLOW}Loading model from {args.checkpoint}...{Style.RESET_ALL}")
    model, llm_tokenizer, compressor_tokenizer, _, _ = load_resources(
        args.config, args.checkpoint, args.device
    )
    model.eval()
    if args.dtype == "bf16":
        model.to(torch.bfloat16)
        print(f"{Fore.YELLOW}Cast model to bfloat16 for inference.{Style.RESET_ALL}")
    return model, _build_ctx(model, llm_tokenizer, compressor_tokenizer, args.device,
                             args.question_len)


def _load_beacon(args):
    """Activation Beacon, which ships its own Qwen3 backbone."""
    if not args.beacon_checkpoint:
        raise SystemExit("--beacon-checkpoint is required for the beacon adapter.")
    print(f"{Fore.YELLOW}Loading Activation Beacon from {args.beacon_checkpoint}...{Style.RESET_ALL}")
    model, _ = load_beacon_model(
        args.beacon_checkpoint,
        base_model_dir=args.beacon_base,
        beacon_ratio=args.beacon_ratio,
        beacon_window=args.beacon_window,
        device=args.device,
        dtype=args.dtype,
    )
    print(f"{Fore.YELLOW}Beacon window={model.config.beacon_window} "
          f"ratio={model.config.beacon_ratio}{Style.RESET_ALL}")
    return model, {"device": args.device, "beacon_vocab": int(model.config.vocab_size)}


def _load_sac(args):
    """SAC, which ships its own LoRA encoder + frozen decoder pair."""
    if not (args.sac_checkpoint or args.sac_base):
        raise SystemExit("--sac-checkpoint or --sac-base is required for the sac adapter.")
    source = args.sac_checkpoint or args.sac_base
    print(f"{Fore.YELLOW}Loading SAC from {source}...{Style.RESET_ALL}")
    model, task_config, adapter_path = load_sac_model(
        checkpoint_dir=args.sac_checkpoint,
        base_model_dir=args.sac_base,
        compress_ratio=args.sac_ratio,
        chunk_size=args.sac_chunk,
        device=args.device,
        dtype=args.dtype,
    )
    print(f"{Fore.YELLOW}SAC chunk={task_config['chunk_size']} ratio={task_config['compress_ratio']} "
          f"mem/chunk={task_config['mem_size']} adapter={adapter_path}{Style.RESET_ALL}")
    return model, {"device": args.device, "sac_vocab": int(model.vocab_size),
                   "question_len": args.question_len}


def _load_icae(args):
    """ICAE: one base model shared between a LoRA encoder and a frozen decoder."""
    if not args.icae_base:
        raise SystemExit("--icae-base is required for the icae adapter.")
    print(f"{Fore.YELLOW}Loading ICAE from {args.icae_base}...{Style.RESET_ALL}")
    model = load_icae_model(
        args.icae_base,
        checkpoint=args.icae_checkpoint,
        lora_r=args.icae_lora_r,
        device=args.device,
        dtype=args.dtype,
    )
    print(f"{Fore.YELLOW}ICAE ratio={args.icae_ratio} lora_r={args.icae_lora_r} "
          f"(n_mem = ceil(L / ratio), set per cell){Style.RESET_ALL}")
    return model, {
        "device": args.device,
        "icae_vocab": int(model.llm.config.vocab_size),
        "icae_ratio": int(args.icae_ratio),
    }


def main():
    parser = argparse.ArgumentParser(description="End-to-end latency benchmark")
    parser.add_argument("--config", type=str, help="Train config (as in inference.py)")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--beacon-checkpoint", type=str,
                        help="Activation Beacon checkpoint dir (for the beacon adapter)")
    parser.add_argument("--beacon-base", type=str,
                        help="Override for Beacon's frozen backbone weights. Needed when the "
                             "checkpoint's own model.safetensors is not readable; the backbone "
                             "never trains, so any copy of the same base is equivalent.")
    parser.add_argument("--beacon-ratio", type=int, default=8,
                        help="Single compression ratio. Checkpoints train on a sampled list, "
                             "which would make compression rate and latency nondeterministic.")
    parser.add_argument("--beacon-window", type=int,
                        help="Override the checkpoint's trained compression window (and stride). "
                             "Valid for LATENCY only: no beacon tensor's shape depends on the "
                             "window, so the cost is that of a checkpoint trained at this window, "
                             "but the generated text is meaningless. Results are stamped "
                             "beacon_window_overridden=true.")
    parser.add_argument("--sac-checkpoint", type=str,
                        help="SAC run directory (holds config.json plus adapter.pt / "
                             "instruction_adapter.pt, either at the root or under output/). "
                             "Optional: the trained tensors do not change any matmul shape, "
                             "so latency is the same without them.")
    parser.add_argument("--sac-base", type=str,
                        help="Base model for SAC. Read from the checkpoint's config.json when "
                             "omitted; required when no checkpoint is given.")
    parser.add_argument("--sac-ratio", type=int, default=8,
                        help="SAC compression ratio. Each chunk keeps chunk_size // ratio slots.")
    parser.add_argument("--sac-chunk", type=int, default=512,
                        help="SAC chunk size, i.e. how much context one encoder pass sees.")
    parser.add_argument("--icae-base", type=str,
                        help="Base model for ICAE (encoder and decoder share these weights).")
    parser.add_argument("--icae-checkpoint", type=str,
                        help="Optional ICAE state dict. Its [MEM] bank is fixed at the trained "
                             "n_mem, so only cells with a matching slot count can use it.")
    parser.add_argument("--icae-ratio", type=int, default=8,
                        help="ICAE compression ratio; n_mem = ceil(seq_len / ratio) per cell.")
    parser.add_argument("--icae-lora-r", type=int, default=128,
                        help="LoRA rank on the ICAE encoder's q_proj/v_proj.")
    parser.add_argument("--question-len", type=int, default=32,
                        help="Length of the literal question block, for the methods that keep "
                             "the question uncompressed on the decoder side (ssm_kv, sac). One "
                             "value for both so their prefill costs stay comparable.")
    parser.add_argument("--bench-config", type=str, default="benchmarks/latency/config.yaml")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32"],
                        help="Inference dtype. bf16 casts the whole model (mamba2 kernels follow "
                             "param dtype, so a plain .to() is cleaner than autocast).")
    parser.add_argument("--out", type=str, default="benchmarks/latency/outputs/results.jsonl")
    args = parser.parse_args()

    with open(args.bench_config) as f:
        bench_cfg = yaml.safe_load(f)

    loaders = {
        "boa": lambda: _load_boa(args),
        "beacon": lambda: _load_beacon(args),
        "sac": lambda: _load_sac(args),
        "icae": lambda: _load_icae(args),
    }

    meta = {
        "checkpoint": args.checkpoint,
        "beacon_checkpoint": args.beacon_checkpoint,
        "beacon_base": args.beacon_base,
        # Two checkpoints of the same backbone differ only by their window, so
        # recording it is what lets a results file be attributed to one of them.
        # This is the EFFECTIVE window -- what was measured -- and the companion
        # flag says whether it came from the checkpoint or from --beacon-window.
        # A figure must never present an overridden run as a trained one.
        "beacon_window": args.beacon_window or _beacon_window(args.beacon_checkpoint),
        "beacon_window_overridden": args.beacon_window is not None,
        "beacon_ratio": args.beacon_ratio,
        "sac_checkpoint": args.sac_checkpoint,
        "sac_ratio": args.sac_ratio,
        "sac_chunk": args.sac_chunk,
        "icae_base": args.icae_base,
        "icae_checkpoint": args.icae_checkpoint,
        "icae_ratio": args.icae_ratio,
        "question_len": args.question_len,
        "device": args.device,
        "dtype": args.dtype,
        "max_new_tokens": bench_cfg["max_new_tokens"],
        "warmup": bench_cfg.get("warmup", 2),
        "repeats": bench_cfg.get("repeats", 5),
    }

    print(f"{Fore.YELLOW}Running sweep: adapters={bench_cfg['adapters']} "
          f"seq_lens={bench_cfg['seq_lens']}{Style.RESET_ALL}")
    handle = _open_results(args.out, meta)
    rows = []

    def _record(row):
        # `rows` is accumulated here rather than from run_sweep's return value:
        # if the sweep dies mid-cell it never returns, and the summary must still
        # reflect everything that actually completed.
        rows.append(row)
        handle.write(json.dumps(row) + "\n")
        handle.flush()

    try:
        run_sweep(loaders, bench_cfg, log=_log, on_row=_record)
    finally:
        handle.close()
        print(f"{Fore.GREEN}Wrote {len(rows)} rows -> {args.out}{Style.RESET_ALL}")
        _print_table(rows)


if __name__ == "__main__":
    main()

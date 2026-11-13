"""Shared helpers for the Qwen3 FullFT baseline scripts (train + dump)."""
import os

import yaml
import torch
import datasets
import torch.distributed as dist
from rich.console import Console
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import logging as hf_logging


def is_main_process() -> bool:
    return int(os.environ.get("LOCAL_RANK", 0)) == 0


def make_console() -> Console:
    """Rank-0 console; quiet elsewhere. log_path=False drops rich's file:line suffix."""
    return Console(quiet=not is_main_process(), log_path=False)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def quiet_external_logging() -> None:
    """Silence HF + datasets chatter so our rich output is the only console source."""
    (hf_logging.set_verbosity_warning if is_main_process() else hf_logging.set_verbosity_error)()
    hf_logging.disable_progress_bar()
    datasets.disable_progress_bars()


def setup_distributed():
    """(rank, world_size, local_rank); initializes the NCCL group under torchrun.
    device_id binds collectives to this rank's GPU (mutes the barrier() device warning)."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device_id = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else None
        dist.init_process_group(backend="nccl", device_id=device_id)
        return dist.get_rank(), dist.get_world_size(), local_rank
    return 0, 1, 0


def resolve_resume(resume_cfg, output_dir):
    """resume_from -> checkpoint: null->None; "latest"/"auto"/true->newest in dir; path->path."""
    if not resume_cfg:
        return None
    if resume_cfg in (True, "latest", "auto"):
        return get_last_checkpoint(output_dir) if os.path.isdir(output_dir) else None
    return resume_cfg


def align_model_tokens(model, tokenizer):
    """Align config + generation_config pad/bos to the tokenizer (avoids HF auto-align
    warning). EOS untouched — Qwen3's multi-id gen eos is needed for correct stopping."""
    for cfg in (model.config, getattr(model, "generation_config", None)):
        if cfg is not None:
            cfg.pad_token_id = tokenizer.pad_token_id
            cfg.bos_token_id = tokenizer.bos_token_id
    return model

# Adapted from Activation Beacon (https://github.com/FlagOpen/FlagEmbedding, research/Long_LLM/activation_beacon),
# Copyright (c) 2022 staoxiao, MIT License. Modified for the Boa baseline reproduction; see THIRD_PARTY_NOTICES.md.
"""Helpers required by the vendored Activation Beacon Qwen3 backend.

Copied from `src/baselines/activation_beacon/training.py` (see the `baseline/activation-beacon`
branch); the helpers match that branch's versions apart from docstring wording. Only these
three model-side helpers are vendored:
the rest of that module pulls in the MRQA data pipeline, which the latency
benchmark does not need.
"""

from __future__ import annotations

import torch
from transformers import AutoTokenizer


def ensure_beacon_memory_dtype(model) -> torch.dtype:
    """Keep Beacon Memory compatible with torch.finfo after wrappers/config IO."""
    unwrapped = getattr(model, "module", model)
    dtype = next(unwrapped.parameters()).dtype
    unwrapped.config.torch_dtype = dtype
    if hasattr(unwrapped, "memory"):
        unwrapped.memory.config.torch_dtype = dtype
    return dtype


def freeze_non_beacon_parameters(model) -> dict[str, int]:
    trainable = total = 0
    trainable_names = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad = "beacon_" in name
        total += parameter.numel()
        if parameter.requires_grad:
            trainable += parameter.numel()
            trainable_names.append(name)
    if trainable == 0:
        raise RuntimeError("No beacon parameters were found in the initialized model.")
    return {"trainable": trainable, "total": total, "names": trainable_names}


def load_beacon_tokenizer(config: dict):
    settings = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        settings["base_model_name_or_path"],
        padding_side="left",
        trust_remote_code=True,
        local_files_only=bool(settings.get("local_files_only", False)),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

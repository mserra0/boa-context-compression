# Adapted from Activation Beacon (https://github.com/FlagOpen/FlagEmbedding, research/Long_LLM/activation_beacon),
# Copyright (c) 2022 staoxiao, MIT License. Modified for the Boa baseline reproduction; see THIRD_PARTY_NOTICES.md.
"""Official Qwen2/Qwen2.5 remote-code Activation Beacon backend."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from .training import (
    ensure_beacon_memory_dtype,
    freeze_non_beacon_parameters,
    load_beacon_tokenizer,
)


def build_beacon_config(base_config, template_config, settings: dict):
    """Create the remote Beacon Qwen config using Qwen2.5 architecture values."""
    config_dict = base_config.to_dict()
    config_dict["auto_map"] = template_config.auto_map
    config_dict["architectures"] = ["Qwen2ForCausalLM"]
    beacon_config = type(template_config).from_dict(config_dict)
    for name in (
        "beacon_window",
        "beacon_stride",
        "beacon_attn",
        "beacon_ratio",
        "beacon_ratio_mix",
        "beacon_param",
        "beacon_embed_init",
        "beacon_sink_size",
        "beacon_attend_prev",
        "beacon_pos",
        "beacon_parallel_window",
    ):
        setattr(beacon_config, name, settings[name])
    # The custom model owns a one-row beacon embedding; a vocabulary pad index is
    # invalid for that embedding. Training labels already mask padded positions.
    beacon_config.pad_token_id = None
    beacon_config.use_cache = True
    return beacon_config

def load_trainable_qwen2_remote_beacon_model(config: dict, tokenizer=None):
    settings = config["model"]
    compat_root = Path(__file__).with_name("compat")
    if str(compat_root) not in sys.path:
        sys.path.insert(0, str(compat_root))

    if tokenizer is None:
        tokenizer = load_beacon_tokenizer(config)

    base_config = AutoConfig.from_pretrained(
        settings["base_model_name_or_path"],
        trust_remote_code=True,
        local_files_only=bool(settings.get("local_files_only", False)),
    )
    template_config = AutoConfig.from_pretrained(
        settings["beacon_code_model"],
        trust_remote_code=True,
        local_files_only=bool(settings.get("local_files_only", False)),
    )
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[settings.get("dtype", "bf16")]
    beacon_config = build_beacon_config(base_config, template_config, settings)
    # PretrainedConfig.to_dict() serializes torch.dtype as a string. The custom
    # Beacon Memory calls torch.finfo(config.torch_dtype), so it requires the
    # concrete dtype object rather than "bfloat16"/"float16".
    beacon_config.torch_dtype = dtype
    model = AutoModelForCausalLM.from_pretrained(
        settings["base_model_name_or_path"],
        config=beacon_config,
        trust_remote_code=True,
        torch_dtype=dtype,
        attn_implementation=settings.get("attn_implementation", "sdpa"),
        local_files_only=bool(settings.get("local_files_only", False)),
    )
    # Some Transformers versions deserialize config values again during loading.
    # Memory retains the same config object, but set both references defensively.
    model.config.torch_dtype = dtype
    model.memory.config.torch_dtype = dtype
    ensure_beacon_memory_dtype(model)
    stats = freeze_non_beacon_parameters(model)
    return model, tokenizer, stats




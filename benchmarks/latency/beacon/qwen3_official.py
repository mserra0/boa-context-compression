# Adapted from Activation Beacon (https://github.com/FlagOpen/FlagEmbedding, research/Long_LLM/activation_beacon),
# Copyright (c) 2022 staoxiao, MIT License. Modified for the Boa baseline reproduction; see THIRD_PARTY_NOTICES.md.
"""Qwen3 backend for the official Activation Beacon mechanism.

Vendored from an earlier revision of the Activation Beacon Qwen3 implementation (kept
unmodified so the benchmark measures the method as trained). The full training code lives on the
`baseline/activation-beacon` branch, file `src/baselines/activation_beacon/qwen3_official.py`.
That branch's version adds
training-only features this inference-only benchmark does not use: gradient checkpointing
(active only when `self.training`) and a `fixed` beacon_ratio_mix (only reachable with a
multi-value `beacon_ratio`; a single ratio returns before it). It imports its helpers from
`.training` instead of `.support`, and its decoder-layer loop unpacks/repacks the per-layer
cache tuple (equivalent when not checkpointing). Nothing else differs, so the eval-mode
forward path is the same code.

This file intentionally does **not** reuse ``qwen3_simple``.  The simple backend
inserts learned beacon tokens into a normal Qwen3 forward pass; this backend
ports the authors' recurrent Activation Beacon mechanism to Qwen3:

* a Beacon memory controller splits the input into windows;
* Beacon tokens are inserted by the memory controller;
* every layer returns incremental pre-RoPE key/value activations;
* memory keeps sink / beacon / recent raw activations and drops old raw KV;
* the next window attends to compressed Beacon memory with custom masks and
  position ids.

The implementation follows the official Qwen2 remote-code ``Memory`` logic, but
the model-side attention path is adapted to native Qwen3 modules, including
Qwen3's q/k normalization.
"""

from __future__ import annotations

import json
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.qwen3.modeling_qwen3 import repeat_kv, rotate_half
from transformers.utils import ModelOutput

from .support import (
    ensure_beacon_memory_dtype,
    freeze_non_beacon_parameters,
    load_beacon_tokenizer,
)


def _dtype_from_name(name: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }[str(name).lower()]


def _as_ratio_list(value) -> list[int]:
    if value is None:
        return [0]
    if isinstance(value, str):
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            value = [part.strip() for part in value[1:-1].split(",") if part.strip()]
        else:
            value = [value]
    elif isinstance(value, int):
        value = [value]
    return [int(item) for item in value]


def _optional_grad_ctx(with_grad: bool):
    return torch.enable_grad() if with_grad else torch.no_grad()


def _compute_loss(logits: torch.Tensor, labels: torch.Tensor):
    """Official Beacon Memory rotates labels itself, so loss is shift=False."""
    vocab_size = logits.shape[-1]
    flat_loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    )
    token_loss = flat_loss.view_as(labels)
    valid = labels.ne(-100)
    valid_count = valid.sum(dim=-1).clamp_min(1)
    batch_loss = (token_loss * valid).sum(dim=-1) / valid_count
    loss = (token_loss * valid).sum() / valid.sum().clamp_min(1)
    return loss, batch_loss, token_loss


def slice_tensor(x, start=None, end=None, step=None, index=None, dim=2):
    if x is None:
        return None
    if end == 0:
        return None
    if start == x.shape[dim]:
        return None
    if start is not None and start == end:
        return None
    if dim == 2:
        if index is not None:
            return x[:, :, index]
        if start is None and end is not None:
            return x[:, :, :end:step, ...] if step is not None else x[:, :, :end, ...]
        if start is not None and end is None:
            return x[:, :, start::step, ...] if step is not None else x[:, :, start:, ...]
        if start is not None and end is not None:
            return x[:, :, start:end:step, ...] if step is not None else x[:, :, start:end, ...]
    elif dim == 1:
        if index is not None:
            return x[:, index]
        if start is None and end is not None:
            return x[:, :end:step, ...] if step is not None else x[:, :end, ...]
        if start is not None and end is None:
            return x[:, start::step, ...] if step is not None else x[:, start:, ...]
        if start is not None and end is not None:
            return x[:, start:end:step, ...] if step is not None else x[:, start:end, ...]
    raise NotImplementedError(f"Unsupported slice dim={dim}.")


def cat_tensor(list_of_tensors, dim=-1):
    tensors = [tensor for tensor in list_of_tensors if tensor is not None]
    if len(tensors) > 1:
        return torch.cat(tensors, dim=dim)
    if len(tensors) == 1:
        return tensors[0]
    return None


class OfficialBeaconMemory(torch.nn.Module):
    """Port of the authors' Activation Beacon memory controller.

    The implementation mirrors the official remote-code ``Memory`` class for the
    single-window recurrent path used by our experiments.  ``beacon_parallel_window``
    is deliberately rejected here because Qwen3's native layer API needs a custom
    per-layer forward loop; keeping the recurrent path first makes the exact
    compression behavior testable and avoids silently mixing two mechanisms.
    """

    def __init__(self, model_config, k_seq_dim: int = 2, v_seq_dim: int = 2):
        super().__init__()
        self.config = model_config
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.rng = np.random.default_rng(42)
        self.beacon_token = torch.tensor([self.config.vocab_size])
        self._post_validation()
        self.reset()

    def _post_validation(self, verbose: bool = True) -> None:
        if self.config.beacon_window < self.config.beacon_stride:
            raise ValueError(
                "Activation Beacon requires beacon_window >= beacon_stride; "
                f"got {self.config.beacon_window} < {self.config.beacon_stride}."
            )
        if int(getattr(self.config, "beacon_parallel_window", 1)) != 1:
            raise NotImplementedError(
                "qwen3_official currently supports the official recurrent path "
                "with beacon_parallel_window=1."
            )
        for ratio in self.config.beacon_ratio:
            if ratio < 0:
                raise ValueError(f"beacon_ratio values must be >= 0; got {self.config.beacon_ratio}.")
        if self.config.beacon_attn not in {"segmentation", "step-expansion", "full-coverage"}:
            raise ValueError(f"Unsupported beacon_attn={self.config.beacon_attn!r}.")
        if self.config.beacon_ratio_mix not in {"instance-random", "step-random", "sequence"} and "adapt-" not in self.config.beacon_ratio_mix:
            raise ValueError(f"Unsupported beacon_ratio_mix={self.config.beacon_ratio_mix!r}.")
        if self.config.beacon_pos == "interleave" and self.config.beacon_window != self.config.beacon_stride:
            raise ValueError("beacon_pos='interleave' requires beacon_window == beacon_stride.")
        self._cpu = torch.device("cpu")

    def set(self, verbose: bool = True, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self.config, key, value)
        self._post_validation(verbose=verbose)

    def reset(self) -> None:
        self._start_idx = 0
        self._end_idx = 0
        self._all_beacon_sizes = []
        self._batch_loss = None
        self._valid_token_num = None
        self._step_idx = 0
        self._compression_ratio = None
        self._is_full_window = True
        self._raw_size_to_cache = 0
        self._interleave_remainder = 0
        self._interleave_compression_ratio = None
        self._beacon_indices = None
        self.all_input_ids = None
        self.all_attention_mask = None
        self.all_labels = None
        self.beacon_skip_first = None
        self.beacon_skip_last = None
        self.raw_activations = [(None, None) for _ in range(self.config.num_hidden_layers)]
        self.sink_activations = [(None, None) for _ in range(self.config.num_hidden_layers)]
        self.beacon_activations = [(None, None) for _ in range(self.config.num_hidden_layers)]

    @property
    def all_sequence_length(self) -> int:
        return 0 if self.all_input_ids is None else self.all_input_ids.shape[1]

    @property
    def finish(self) -> bool:
        return self._end_idx == self.all_sequence_length

    @property
    def dtype(self) -> torch.dtype:
        return self.config.torch_dtype

    @property
    def min_value(self):
        return torch.finfo(self.dtype).min

    @property
    def max_position_embeddings(self):
        max_position_embeddings = self.config.max_position_embeddings
        if getattr(self.config, "rope_scaling", None) is not None:
            max_position_embeddings = max_position_embeddings * self.config.rope_scaling["factor"]
        return max_position_embeddings

    def get_memory_size(self):
        sink_memory_size = 0
        beacon_memory_size = 0
        raw_memory_size = 0
        if self.sink_activations[0][0] is not None:
            sink_memory_size += self.sink_activations[0][0].shape[self.k_seq_dim]
        if self.beacon_activations[0][0] is not None:
            beacon_memory_size += self.beacon_activations[0][0].shape[self.k_seq_dim]
        if self.raw_activations[0][0] is not None:
            raw_memory_size += self.raw_activations[0][0].shape[self.k_seq_dim]
        return sink_memory_size, beacon_memory_size, raw_memory_size

    def prepare(self, input_ids, attention_mask, labels, skip_first=None, skip_last=None):
        self._device = input_ids.device
        if self.all_input_ids is None:
            self.all_input_ids = input_ids.cpu()
        else:
            self.all_input_ids = torch.cat([self.all_input_ids, input_ids.cpu()], dim=1)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self._cpu)
        if self.all_attention_mask is None:
            self.all_attention_mask = attention_mask.cpu()
        else:
            self.all_attention_mask = torch.cat([self.all_attention_mask, attention_mask.cpu()], dim=1)

        if labels is not None:
            rotated = torch.cat(
                [labels[:, 1:].cpu(), torch.tensor([-100]).expand(labels.shape[0], 1)],
                dim=1,
            )
            if self.all_labels is None:
                self.all_labels = rotated
            else:
                self.all_labels = torch.cat([self.all_labels, rotated], dim=1)
            if self.all_input_ids.shape[1] != self.all_labels.shape[1]:
                raise RuntimeError("Inconsistent input/label lengths in Beacon memory.")

        if skip_first is not None:
            if self.config.beacon_window != self.config.beacon_stride:
                raise ValueError("beacon_skip_first requires beacon_window == beacon_stride.")
            if self.config.beacon_sink_size != 0:
                raise ValueError("beacon_skip_first requires beacon_sink_size=0.")
        if skip_last is not None:
            skip_first = skip_first if skip_first is not None else 0
            if (skip_last - skip_first) % self.config.beacon_window != 0:
                raise ValueError("skip_last - skip_first must be divisible by beacon_window.")
            if self.config.beacon_sink_size != 0:
                raise ValueError("beacon_skip_last requires beacon_sink_size=0.")
        self.beacon_skip_first = skip_first
        self.beacon_skip_last = skip_last

    def set_compression_ratio(self, start_idx, end_idx):
        def filter_ratio(ratios, stride):
            valid = []
            for ratio in ratios:
                if stride < ratio:
                    continue
                if ratio > 0 and stride % ratio != 0:
                    continue
                if ratio == 0 and self.training:
                    previous_has_zero = -1 in self._all_beacon_sizes
                    following_has_nonzero = (start_idx + stride + self.config.beacon_window) <= self.all_sequence_length
                    if previous_has_zero or not following_has_nonzero:
                        continue
                valid.append(ratio)
            if not valid:
                raise ValueError(
                    f"Cannot find valid condensing ratio among {ratios} for stride {stride}."
                )
            return valid

        def get_max_length(ratios):
            max_lengths = []
            for compression_ratio in ratios:
                if compression_ratio > 0:
                    max_lengths.append(
                        (self.max_position_embeddings - self.config.beacon_window)
                        * compression_ratio
                        + self.config.beacon_window
                    )
                else:
                    max_lengths.append(self.max_position_embeddings)
            return max_lengths

        if len(self.config.beacon_ratio) == 1:
            return self.config.beacon_ratio[0]

        ratios = filter_ratio(self.config.beacon_ratio, self.config.beacon_stride)
        ratio_mix = self.config.beacon_ratio_mix
        if ratio_mix == "instance-random":
            if self._compression_ratio is None:
                self._compression_ratio = self.rng.choice(ratios).tolist()
            return self._compression_ratio
        if ratio_mix == "step-random":
            return self.rng.choice(ratios).tolist()
        if ratio_mix == "sequence":
            if self._compression_ratio is None:
                self._compression_ratio = cycle(ratios)
            return next(self._compression_ratio)
        if "adapt" in ratio_mix:
            if self._compression_ratio is None:
                future_length = int(ratio_mix.split("-")[1])
                sequence_length = self.all_input_ids.shape[1] + future_length
                max_lengths = get_max_length(ratios)
                valid = [item for item in enumerate(max_lengths) if item[1] >= sequence_length]
                if valid:
                    ratio = ratios[min(valid, key=lambda item: item[1])[0]]
                else:
                    ratio = max(ratios)
                self._compression_ratio = ratio
            return self._compression_ratio
        raise ValueError(f"Unsupported beacon_ratio_mix={ratio_mix!r}.")

    def step(self):
        return self._step()

    def _step(self, ignore_memory: bool = False):
        start_idx = self._start_idx
        end_idx = start_idx + self.config.beacon_window
        if end_idx > self.all_sequence_length:
            end_idx = self.all_sequence_length
            is_full_window = False
        else:
            is_full_window = True

        if self.training and end_idx == self.all_sequence_length:
            next_start_idx = start_idx
            is_full_window = False
            raw_size_to_cache = -1
            beacon_size = 0
            compression_ratio = -1
        elif self._step_idx == 0 and self.beacon_skip_first is not None:
            end_idx = start_idx + self.beacon_skip_first
            next_start_idx = end_idx
            is_full_window = True
            raw_size_to_cache = -1
            beacon_size = 0
            compression_ratio = -1
        elif self.beacon_skip_last is not None and start_idx >= self.beacon_skip_last:
            end_idx = min(start_idx + self.config.beacon_window, self.all_sequence_length)
            next_start_idx = end_idx
            is_full_window = False
            raw_size_to_cache = -1
            beacon_size = 0
            compression_ratio = -1
        elif self.config.beacon_pos == "append":
            if is_full_window:
                compression_ratio = self.set_compression_ratio(start_idx=start_idx, end_idx=end_idx)
                beacon_size = self.config.beacon_stride // compression_ratio if compression_ratio > 0 else -1
                next_start_idx = start_idx + self.config.beacon_stride
                raw_size_to_cache = end_idx - next_start_idx
            else:
                next_start_idx = start_idx
                raw_size_to_cache = -1
                beacon_size = 0
                compression_ratio = 0
        elif self.config.beacon_pos == "interleave":
            input_size = end_idx - self._end_idx
            if self._is_full_window:
                compression_ratio = self.set_compression_ratio(start_idx=start_idx, end_idx=end_idx)
                self._interleave_compression_ratio = compression_ratio
            else:
                compression_ratio = self._interleave_compression_ratio
            beacon_size = (input_size + self._interleave_remainder) // compression_ratio if compression_ratio > 0 else -1
            if is_full_window:
                next_start_idx = start_idx + self.config.beacon_stride
                raw_size_to_cache = 0
            else:
                next_start_idx = start_idx
                raw_size_to_cache = -1
        else:
            raise ValueError(f"Unsupported beacon_pos={self.config.beacon_pos!r}.")

        input_ids = self.all_input_ids[:, self._end_idx:end_idx].to(self._device)
        attention_mask = self.all_attention_mask[:, self._end_idx:end_idx].to(self._device)
        labels = self.all_labels[:, self._end_idx:end_idx].to(self._device) if self.all_labels is not None else None
        batch_size = input_ids.shape[0]

        if self.config.beacon_pos == "append":
            if is_full_window and beacon_size > 0:
                input_ids = torch.cat(
                    [input_ids, self.beacon_token.expand(batch_size, beacon_size).to(input_ids.device, dtype=input_ids.dtype)],
                    dim=1,
                )
                attention_mask = torch.cat([attention_mask, attention_mask.new_ones(batch_size, beacon_size)], dim=1)
                if labels is not None:
                    labels = torch.cat([labels, labels.new_full((batch_size, beacon_size), -100)], dim=1)
        elif self.config.beacon_pos == "interleave":
            input_len = input_ids.shape[1]
            if beacon_size > 0:
                input_ids_with_beacons = input_ids.new_full(
                    (batch_size, input_len + beacon_size), self.beacon_token.item()
                )
                raw_token_indices = torch.arange(input_ids_with_beacons.shape[1], device=input_ids.device)
                interleave_start_idx = compression_ratio - self._interleave_remainder
                raw_token_indices = raw_token_indices[
                    raw_token_indices % (compression_ratio + 1) != interleave_start_idx
                ].unsqueeze(0).expand_as(input_ids)
                input_ids = input_ids_with_beacons.scatter(dim=1, index=raw_token_indices, src=input_ids)

                mask_with_beacons = attention_mask.new_ones(batch_size, attention_mask.shape[1] + beacon_size)
                attention_mask = mask_with_beacons.scatter(dim=1, index=raw_token_indices, src=attention_mask)
                if labels is not None:
                    labels_with_beacons = labels.new_full((batch_size, labels.shape[1] + beacon_size), -100)
                    labels = labels_with_beacons.scatter(dim=1, index=raw_token_indices, src=labels)
            if compression_ratio > 0:
                self._interleave_remainder = (input_len + self._interleave_remainder) % compression_ratio

        if self.training and self._step_idx == 0 and not (
            self.config.beacon_pos == "interleave" and self.config.beacon_attn == "full-coverage"
        ):
            labels[:] = -100

        beacon_indices = (input_ids[0] == self.beacon_token.item()).long()
        if self._is_full_window:
            self._beacon_indices = torch.tensor([], dtype=torch.long, device=input_ids.device)
        beacon_indices = torch.cat([self._beacon_indices, beacon_indices])
        self._beacon_indices = beacon_indices
        if is_full_window and beacon_size == -1:
            beacon_indices[: self.config.beacon_stride] = -1

        past_key_values = []
        for layer_idx in range(self.config.num_hidden_layers):
            if ignore_memory:
                key, value = None, None
            else:
                sink_key, sink_value = self.sink_activations[layer_idx]
                beacon_key, beacon_value = self.beacon_activations[layer_idx]
                raw_key, raw_value = self.raw_activations[layer_idx]
                key = cat_tensor([sink_key, beacon_key, raw_key], dim=self.k_seq_dim)
                value = cat_tensor([sink_value, beacon_value, raw_value], dim=self.v_seq_dim)
            past_key_values.append((key, value, beacon_size, beacon_indices))

        first_key = past_key_values[0][0]
        mem_size = first_key.shape[self.k_seq_dim] if first_key is not None else 0
        if mem_size > 0:
            attention_mask = torch.cat([attention_mask.new_ones(batch_size, mem_size), attention_mask], dim=1)

        input_length = input_ids.shape[1]
        position_ids = torch.arange(attention_mask.shape[-1], dtype=torch.long, device=self._device).repeat(batch_size, 1)

        if self.config._attn_implementation == "flash_attention_2":
            if self.config.beacon_attn != "full-coverage":
                raise ValueError("flash_attention_2 requires beacon_attn='full-coverage'.")
            if 0 not in attention_mask:
                attention_mask = None
        elif self.config._attn_implementation == "sdpa" and self.config.beacon_pos == "append" and beacon_size <= 0 and (input_length == 1 or mem_size == 0):
            attention_mask = None
        else:
            attention_mask, position_ids = self._make_4d_attention_mask_and_position_ids(
                attention_mask,
                position_ids,
                mem_size,
                beacon_size,
                compression_ratio,
            )

        self._is_full_window = is_full_window
        self._raw_size_to_cache = raw_size_to_cache
        self._all_beacon_sizes.append(beacon_size)
        self._start_idx = next_start_idx
        self._end_idx = end_idx
        self._step_idx += 1
        return input_ids, attention_mask, position_ids, past_key_values, labels

    def _make_4d_attention_mask_and_position_ids(self, attention_mask, position_ids, mem_size, beacon_size, compression_ratio):
        tgt_size = attention_mask.size(-1) - mem_size
        dtype = self.dtype
        min_value = self.min_value
        device = self._device
        batch_size, src_size = attention_mask.size()

        causal_mask = torch.full((tgt_size, tgt_size), min_value, device=device, dtype=dtype)
        mask_cond = torch.arange(causal_mask.size(-1), device=device)
        causal_mask.masked_fill_(mask_cond < (mask_cond + 1).view(causal_mask.size(-1), -1), 0)
        causal_mask = torch.cat([torch.zeros(tgt_size, mem_size, dtype=dtype, device=device), causal_mask], dim=-1)
        causal_mask = causal_mask[None, None, ...].expand(batch_size, 1, tgt_size, src_size)
        expand_mask = attention_mask[:, None, None, :].expand(batch_size, 1, tgt_size, src_size)
        invert_mask = 1.0 - expand_mask
        invert_mask.masked_fill_(invert_mask.bool(), min_value)
        attention_mask = causal_mask.masked_fill(invert_mask.bool(), min_value)

        if self.config.beacon_attn == "step-expansion":
            if self.config.beacon_pos == "append" and beacon_size > 0:
                window_size = self.config.beacon_window
                window_size_with_beacon = window_size + beacon_size
                beacon_start_idx = -beacon_size
                reference_attention_mask = attention_mask[..., -beacon_size - 1, -window_size_with_beacon:-beacon_size]
                beacon_arange = torch.arange(1, beacon_size + 1, device=device) * compression_ratio
                ordinal_arange = torch.arange(window_size, device=device)
                valid_pos = ordinal_arange.expand(beacon_size, window_size) < beacon_arange.unsqueeze(-1)
                ordinal_attention_mask = torch.where(valid_pos, 0, min_value)
                ordinal_attention_mask = ordinal_attention_mask[None, None, ...] + reference_attention_mask.unsqueeze(-2)
                if self.config.beacon_attend_prev:
                    beacon_attention_mask = attention_mask.new_full((beacon_size, beacon_size), min_value).triu(1)
                    ordinal_position_ids = position_ids[:, -window_size_with_beacon:-beacon_size]
                    beacon_position_ids = ordinal_position_ids[:, compression_ratio - 1 :: compression_ratio] + torch.arange(1, beacon_size + 1, device=device)[None]
                else:
                    beacon_attention_mask = attention_mask.new_full((beacon_size, beacon_size), min_value).fill_diagonal_(0)
                    ordinal_position_ids = position_ids[:, -window_size_with_beacon:-beacon_size]
                    beacon_position_ids = ordinal_position_ids[:, compression_ratio - 1 :: compression_ratio] + 1
                position_ids[:, beacon_start_idx:] = beacon_position_ids
                attention_mask[..., beacon_start_idx:, -window_size_with_beacon:-beacon_size] = ordinal_attention_mask
                attention_mask[..., beacon_start_idx:, beacon_start_idx:] = beacon_attention_mask
            elif self.config.beacon_pos == "interleave" and (self._beacon_indices == 1).any():
                if self.config.beacon_attend_prev:
                    raise ValueError("step-expansion + interleave requires beacon_attend_prev=False.")
                beacon_indices = self._beacon_indices.bool()
                cur_position_ids = position_ids[:, -len(beacon_indices):]
                base_position = cur_position_ids[:, 0] - 1
                position_template = cur_position_ids.new_ones(cur_position_ids.shape)
                position_template[:, compression_ratio + 1 :: compression_ratio + 1] = 0
                cur_position_ids = base_position + position_template.cumsum(-1)
                position_ids[:, -len(beacon_indices):] = cur_position_ids
                cur_input_length = len(beacon_indices)
                cur_attention_mask = attention_mask[..., -cur_input_length:, -cur_input_length:]
                cur_attention_mask[..., beacon_indices] = min_value
                input_ids_attention_mask = cur_attention_mask[..., -tgt_size:, -tgt_size:]
                input_ids_attention_mask[..., range(tgt_size), range(tgt_size)] = 0
        elif self.config.beacon_attn == "segmentation":
            if self.config.beacon_pos == "append" and beacon_size > 0:
                window_size = self.config.beacon_window
                window_size_with_beacon = window_size + beacon_size
                beacon_start_idx = -beacon_size
                reference_attention_mask = attention_mask[..., -beacon_size - 1, -window_size_with_beacon:-beacon_size]
                indices = torch.arange(compression_ratio * beacon_size, device=device).view(beacon_size, -1)
                ordinal_attention_mask = attention_mask.new_full((beacon_size, window_size), min_value)
                ordinal_attention_mask.scatter_(dim=-1, index=indices, value=0)
                ordinal_attention_mask = ordinal_attention_mask[None, None, ...] + reference_attention_mask.unsqueeze(-2)
                if self.config.beacon_attend_prev:
                    beacon_attention_mask = attention_mask.new_full((beacon_size, beacon_size), min_value).triu(1)
                    beacon_position_ids = position_ids.new_full((beacon_size,), fill_value=compression_ratio + mem_size) + torch.arange(beacon_size, device=device)
                else:
                    beacon_attention_mask = attention_mask.new_full((beacon_size, beacon_size), min_value).fill_diagonal_(0)
                    beacon_position_ids = position_ids.new_full((beacon_size,), fill_value=compression_ratio + mem_size)
                position_ids[:, beacon_start_idx:] = beacon_position_ids
                attention_mask[..., beacon_start_idx:, -window_size_with_beacon:-beacon_size] = ordinal_attention_mask
                attention_mask[..., beacon_start_idx:, beacon_start_idx:] = beacon_attention_mask
                attention_mask[..., beacon_start_idx:, -beacon_size:beacon_start_idx] = min_value
            elif self.config.beacon_pos == "interleave":
                raise NotImplementedError("Official segmentation + interleave is not implemented upstream.")
        elif self.config.beacon_attn == "full-coverage":
            pass
        return attention_mask, position_ids

    def update_memory(self, past_key_values) -> None:
        for layer_idx, (key, value, beacon_size, beacon_indices) in enumerate(past_key_values):
            previous_raw_key, previous_raw_value = self.raw_activations[layer_idx]
            if self.beacon_skip_first is not None and self.sink_activations[layer_idx][0] is None:
                self.sink_activations[layer_idx] = [key, value]
                continue

            if self.beacon_activations[layer_idx][0] is None and self.config.beacon_sink_size > 0:
                self.sink_activations[layer_idx] = [
                    slice_tensor(key, end=self.config.beacon_sink_size, dim=self.k_seq_dim),
                    slice_tensor(value, end=self.config.beacon_sink_size, dim=self.v_seq_dim),
                ]

            if not self._is_full_window:
                raw_key = cat_tensor([previous_raw_key, key], dim=self.k_seq_dim)
                raw_value = cat_tensor([previous_raw_value, value], dim=self.v_seq_dim)
                self.raw_activations[layer_idx] = (raw_key, raw_value)
            else:
                previous_beacon_key, previous_beacon_value = self.beacon_activations[layer_idx]
                beacon_key, beacon_value, raw_key, raw_value = self._extract_beacon_and_raw_memory(
                    key,
                    value,
                    previous_beacon_key,
                    previous_beacon_value,
                    previous_raw_key,
                    previous_raw_value,
                    beacon_indices,
                )
                self.beacon_activations[layer_idx] = (beacon_key, beacon_value)
                self.raw_activations[layer_idx] = (raw_key, raw_value)

    def update_loss(self, batch_loss, valid_token_num) -> None:
        if self._batch_loss is None:
            self._batch_loss = batch_loss * valid_token_num
            self._valid_token_num = valid_token_num
        else:
            self._batch_loss = self._batch_loss + batch_loss * valid_token_num
            self._valid_token_num = self._valid_token_num + valid_token_num

    def output(self, model_outputs):
        if self._batch_loss is not None:
            loss = self._batch_loss.sum() / self._valid_token_num.sum().clamp_min(1)
            batch_loss = self._batch_loss / self._valid_token_num.clamp_min(1)
            batch_loss = batch_loss.masked_fill(self._valid_token_num == 0, 0.0)
            model_outputs["loss"] = loss
            model_outputs["batch_loss"] = batch_loss

        beacon_size = self._all_beacon_sizes[-1]
        if beacon_size > 0:
            logits = model_outputs["logits"]
            beacon_indices = self._beacon_indices[-logits.shape[1] :]
            model_outputs["logits"] = logits[:, beacon_indices == 0]
        return model_outputs

    def _extract_beacon_and_raw_memory(self, key, value, previous_beacon_key, previous_beacon_value, previous_raw_key, previous_raw_value, beacon_indices):
        key = cat_tensor([previous_raw_key, key], dim=self.k_seq_dim)
        value = cat_tensor([previous_raw_value, value], dim=self.v_seq_dim)
        beacon_selector = torch.logical_or(beacon_indices == 1, beacon_indices == -1)
        beacon_key = slice_tensor(key, index=beacon_selector, dim=self.k_seq_dim)
        beacon_key = cat_tensor([previous_beacon_key, beacon_key], dim=self.k_seq_dim)
        beacon_value = slice_tensor(value, index=beacon_selector, dim=self.v_seq_dim)
        beacon_value = cat_tensor([previous_beacon_value, beacon_value], dim=self.v_seq_dim)
        if self._raw_size_to_cache > 0:
            raw_key = slice_tensor(key, index=beacon_indices == 0, dim=self.k_seq_dim)
            raw_key = slice_tensor(raw_key, start=-self._raw_size_to_cache, dim=self.k_seq_dim)
            raw_value = slice_tensor(value, index=beacon_indices == 0, dim=self.v_seq_dim)
            raw_value = slice_tensor(raw_value, start=-self._raw_size_to_cache, dim=self.v_seq_dim)
        else:
            raw_key = None
            raw_value = None
        return beacon_key, beacon_value, raw_key, raw_value


class OfficialQwen3ActivationBeacon(nn.Module):
    """Qwen3 model wrapper using the official recurrent Beacon mechanism."""

    def __init__(self, base_model, settings: dict, tokenizer=None):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.settings = dict(settings)
        self.tokenizer = tokenizer
        self._enable_beacon = True
        self._install_config()
        self._install_beacon_modules(tokenizer=tokenizer)
        self.memory = OfficialBeaconMemory(self.config, k_seq_dim=2, v_seq_dim=2)

    def _install_config(self) -> None:
        settings = self.settings
        self.config.beacon_window = int(settings["beacon_window"])
        self.config.beacon_stride = int(settings.get("beacon_stride", self.config.beacon_window))
        self.config.beacon_attn = settings.get("beacon_attn", "full-coverage")
        self.config.beacon_ratio = _as_ratio_list(settings.get("beacon_ratio", [2, 4, 8]))
        self.config.beacon_ratio_mix = settings.get("beacon_ratio_mix", "step-random")
        self.config.beacon_param = list(settings.get("beacon_param", ["q", "k", "v"]))
        self.config.beacon_embed_init = settings.get("beacon_embed_init", "eos")
        self.config.beacon_sink_size = int(settings.get("beacon_sink_size", 0))
        self.config.beacon_attend_prev = bool(settings.get("beacon_attend_prev", True))
        self.config.beacon_pos = settings.get("beacon_pos", "interleave")
        self.config.beacon_parallel_window = int(settings.get("beacon_parallel_window", 1))
        self.config._attn_implementation = settings.get("attn_implementation", getattr(self.config, "_attn_implementation", "sdpa"))
        self.config.torch_dtype = _dtype_from_name(settings.get("dtype", "bf16"))
        self.config.use_cache = True

    def _install_beacon_modules(self, tokenizer=None) -> None:
        model = self.base_model.model
        base_embedding = model.embed_tokens.weight
        created_beacon_embedding = False
        if not hasattr(model, "beacon_embed_tokens"):
            model.beacon_embed_tokens = nn.Embedding(1, self.config.hidden_size)
            created_beacon_embedding = True
        model.beacon_embed_tokens = model.beacon_embed_tokens.to(
            device=base_embedding.device,
            dtype=base_embedding.dtype,
        )
        self._init_beacon_embedding(tokenizer, force=created_beacon_embedding)
        for layer in model.layers:
            attention = layer.self_attn
            if "q" in self.config.beacon_param and not hasattr(attention, "beacon_q_proj"):
                attention.beacon_q_proj = self._clone_linear(attention.q_proj)
            if "k" in self.config.beacon_param and not hasattr(attention, "beacon_k_proj"):
                attention.beacon_k_proj = self._clone_linear(attention.k_proj)
            if "v" in self.config.beacon_param and not hasattr(attention, "beacon_v_proj"):
                attention.beacon_v_proj = self._clone_linear(attention.v_proj)
            if "o" in self.config.beacon_param and not hasattr(attention, "beacon_o_proj"):
                attention.beacon_o_proj = self._clone_linear(attention.o_proj)

    @staticmethod
    def _clone_linear(source: nn.Linear) -> nn.Linear:
        clone = nn.Linear(source.in_features, source.out_features, bias=source.bias is not None)
        with torch.no_grad():
            clone.weight.copy_(source.weight.detach())
            if source.bias is not None and clone.bias is not None:
                clone.bias.copy_(source.bias.detach())
        return clone.to(device=source.weight.device, dtype=source.weight.dtype)

    def _init_beacon_embedding(self, tokenizer=None, *, force: bool = False) -> None:
        embedding = self.base_model.model.beacon_embed_tokens
        if not force and embedding.weight.detach().abs().sum() > 0:
            return
        token_id = None
        if self.config.beacon_embed_init == "bos":
            token_id = getattr(self.config, "bos_token_id", None)
        elif self.config.beacon_embed_init == "eos":
            token_id = getattr(self.config, "eos_token_id", None)
            if isinstance(token_id, list):
                token_id = token_id[0]
        if token_id is None and tokenizer is not None:
            token_id = tokenizer.eos_token_id
        if token_id is None:
            nn.init.normal_(embedding.weight, mean=0.0, std=float(getattr(self.config, "initializer_range", 0.02)))
            return
        with torch.no_grad():
            source = self.base_model.get_input_embeddings().weight[int(token_id)]
            embedding.weight.copy_(source.detach().to(embedding.weight.device, embedding.weight.dtype).view_as(embedding.weight))

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    def parameters(self, recurse: bool = True):  # pragma: no cover - nn.Module delegation readability
        return super().parameters(recurse=recurse)

    def _embed_input_ids(self, input_ids: torch.Tensor, beacon_size: int, beacon_indices: torch.Tensor | None):
        if beacon_size > 0:
            cur_beacon_indices = beacon_indices[-input_ids.shape[1] :]
            ordinal_mask = cur_beacon_indices == 0
            ordinal_input_ids = input_ids[:, ordinal_mask]
            beacon_input_ids = input_ids[:, ~ordinal_mask]
            ordinal_embeds = self.base_model.model.embed_tokens(ordinal_input_ids)
            beacon_embeds = self.base_model.model.beacon_embed_tokens(
                beacon_input_ids - self.config.vocab_size
            ).to(dtype=ordinal_embeds.dtype)
            inputs_embeds = ordinal_embeds.new_zeros(
                *input_ids.shape, ordinal_embeds.shape[-1]
            )
            inputs_embeds[:, ordinal_mask] = ordinal_embeds
            inputs_embeds[:, ~ordinal_mask] = beacon_embeds
            return inputs_embeds
        return self.base_model.model.embed_tokens(input_ids)

    def _project_qkv_with_beacon(self, attention, hidden_states, beacon_size, beacon_indices):
        if beacon_size <= 0:
            return attention.q_proj(hidden_states), attention.k_proj(hidden_states), attention.v_proj(hidden_states)
        cur_beacon_indices = beacon_indices[-hidden_states.shape[1] :]
        ordinal = (cur_beacon_indices == 0).view(1, -1, 1)

        if "q" in self.config.beacon_param:
            query_states = torch.where(ordinal, attention.q_proj(hidden_states), attention.beacon_q_proj(hidden_states))
        else:
            query_states = attention.q_proj(hidden_states)
        if "k" in self.config.beacon_param:
            key_states = torch.where(ordinal, attention.k_proj(hidden_states), attention.beacon_k_proj(hidden_states))
        else:
            key_states = attention.k_proj(hidden_states)
        if "v" in self.config.beacon_param:
            value_states = torch.where(ordinal, attention.v_proj(hidden_states), attention.beacon_v_proj(hidden_states))
        else:
            value_states = attention.v_proj(hidden_states)
        return query_states, key_states, value_states

    def _project_o_with_beacon(self, attention, attn_output, beacon_size, beacon_indices):
        if beacon_size > 0 and "o" in self.config.beacon_param:
            cur_beacon_indices = beacon_indices[-attn_output.shape[1] :]
            ordinal = (cur_beacon_indices == 0).view(1, -1, 1)
            return torch.where(ordinal, attention.o_proj(attn_output), attention.beacon_o_proj(attn_output))
        return attention.o_proj(attn_output)

    def _apply_qwen3_rope(self, attention, query_states, key_states, hidden_states, position_ids):
        query_position_ids = position_ids[:, -query_states.shape[2] :]
        q_cos, q_sin = self.base_model.model.rotary_emb(hidden_states, query_position_ids)
        k_cos, k_sin = self.base_model.model.rotary_emb(hidden_states, position_ids)
        query_states = (query_states * q_cos.unsqueeze(1)) + (rotate_half(query_states) * q_sin.unsqueeze(1))
        key_states = (key_states * k_cos.unsqueeze(1)) + (rotate_half(key_states) * k_sin.unsqueeze(1))
        return query_states, key_states

    def _attention_forward(self, attention, hidden_states, attention_mask, position_ids, past_key_value, output_attentions=False):
        batch_size, query_length, _ = hidden_states.shape
        past_key, past_value, beacon_size, beacon_indices = past_key_value
        query_states, key_states, value_states = self._project_qkv_with_beacon(
            attention, hidden_states, beacon_size, beacon_indices
        )
        hidden_shape = (batch_size, query_length, -1, attention.head_dim)
        query_states = attention.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = attention.k_norm(key_states.view(batch_size, query_length, -1, attention.head_dim)).transpose(1, 2)
        value_states = value_states.view(batch_size, query_length, -1, attention.head_dim).transpose(1, 2)
        present_key_value = (key_states, value_states, beacon_size, beacon_indices)

        if past_key is not None:
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)

        query_states, key_states = self._apply_qwen3_rope(
            attention, query_states, key_states, hidden_states, position_ids
        )
        key_states = repeat_kv(key_states, attention.num_key_value_groups)
        value_states = repeat_kv(value_states, attention.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * attention.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=attention.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch_size, query_length, -1)
        attn_output = self._project_o_with_beacon(attention, attn_output, beacon_size, beacon_indices)
        return attn_output, (attn_weights if output_attentions else None), present_key_value

    def _decoder_layer_forward(self, layer, hidden_states, attention_mask, position_ids, past_key_value, output_attentions=False):
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)
        attn_output, attn_weights, present_key_value = self._attention_forward(
            layer.self_attn,
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions=output_attentions,
        )
        hidden_states = residual + attn_output
        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, attn_weights, present_key_value

    def _native_forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=True,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        **kwargs,
    ):
        del inputs_embeds, kwargs
        if past_key_values is None:
            past_key_values = [(None, None, 0, None) for _ in range(self.config.num_hidden_layers)]
        if input_ids is None:
            raise ValueError("qwen3_official requires input_ids; inputs_embeds are not supported.")
        _, _, beacon_size, beacon_indices = past_key_values[0]
        hidden_states = self._embed_input_ids(input_ids, beacon_size, beacon_indices)
        if position_ids is None:
            if attention_mask is None:
                position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand(input_ids.shape[0], -1)
            else:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids = position_ids.masked_fill(attention_mask == 0, 1)
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = []
        for layer_idx, decoder_layer in enumerate(self.base_model.model.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            hidden_states, attn_weights, present = self._decoder_layer_forward(
                decoder_layer,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_values[layer_idx],
                output_attentions=output_attentions,
            )
            next_decoder_cache.append(present)
            if output_attentions:
                all_self_attns += (attn_weights,)
        hidden_states = self.base_model.model.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        logits = self.base_model.lm_head(hidden_states).float()
        loss = batch_loss = token_loss = None
        if labels is not None:
            loss, batch_loss, token_loss = _compute_loss(logits, labels)
        return ModelOutput(
            loss=loss,
            batch_loss=batch_loss,
            token_loss=token_loss,
            logits=logits,
            past_key_values=tuple(next_decoder_cache) if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _beacon_forward(self, input_ids=None, attention_mask=None, labels=None, beacon_skip_first=None, beacon_skip_last=None, **kwargs):
        self.memory.prepare(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            skip_first=beacon_skip_first,
            skip_last=beacon_skip_last,
        )
        outputs = None
        while not self.memory.finish:
            input_ids, attention_mask, position_ids, past_key_values, labels = self.memory.step()
            outputs = self._native_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                labels=labels,
                **kwargs,
            )
            self.memory.update_memory(outputs.past_key_values)
            if labels is not None:
                self.memory.update_loss(outputs.batch_loss, (labels != -100).sum(-1))
        return self.memory.output(outputs)

    def forward(self, **kwargs):
        with _optional_grad_ctx(with_grad=self.training):
            if getattr(self, "_enable_beacon", True) is False:
                return self._native_forward(**kwargs)
            return self._beacon_forward(**kwargs)

    @torch.no_grad()
    def generate(self, input_ids=None, attention_mask=None, max_new_tokens=80, do_sample=False, pad_token_id=None, eos_token_id=None, **kwargs):
        if do_sample:
            raise ValueError("qwen3_official.generate currently supports greedy decoding only.")
        if getattr(self, "_enable_beacon", True) is False:
            return self.base_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                **kwargs,
            )
        eos_ids = eos_token_id
        if eos_ids is None:
            eos_ids = []
        elif isinstance(eos_ids, int):
            eos_ids = [eos_ids]
        else:
            eos_ids = list(eos_ids)
        generated = input_ids
        next_input_ids = input_ids
        next_attention_mask = attention_mask
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        self.memory.reset()
        for _ in range(int(max_new_tokens)):
            outputs = self.forward(input_ids=next_input_ids, attention_mask=next_attention_mask)
            next_token = outputs.logits[:, -1].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if eos_ids:
                finished |= torch.isin(next_token.squeeze(-1), torch.tensor(eos_ids, device=next_token.device))
                if bool(finished.all()):
                    break
            next_input_ids = next_token
            next_attention_mask = torch.ones_like(next_token)
        return generated

    def save_pretrained(self, save_directory, state_dict: dict[str, torch.Tensor] | None = None, safe_serialization: bool = True, **kwargs) -> None:
        destination = Path(save_directory)
        destination.mkdir(parents=True, exist_ok=True)
        full_state = state_dict if state_dict is not None else self.state_dict()
        base_state = {
            key.removeprefix("base_model."): value.detach().cpu()
            for key, value in full_state.items()
            if key.startswith("base_model.") and "beacon_" not in key
        }
        adapter_state = {
            key: value.detach().cpu()
            for key, value in full_state.items()
            if "beacon_" in key
        }
        self.base_model.save_pretrained(
            destination,
            state_dict=base_state,
            safe_serialization=safe_serialization,
            **kwargs,
        )
        torch.save(adapter_state, destination / "activation_beacon_qwen3_official.pt")
        (destination / "activation_beacon_backend.json").write_text(
            json.dumps(
                {
                    "backend": "qwen3_official",
                    "beacon_window": self.config.beacon_window,
                    "beacon_stride": self.config.beacon_stride,
                    "beacon_attn": self.config.beacon_attn,
                    "beacon_ratio": self.config.beacon_ratio,
                    "beacon_ratio_mix": self.config.beacon_ratio_mix,
                    "beacon_param": self.config.beacon_param,
                    "beacon_embed_init": self.config.beacon_embed_init,
                    "beacon_sink_size": self.config.beacon_sink_size,
                    "beacon_attend_prev": self.config.beacon_attend_prev,
                    "beacon_pos": self.config.beacon_pos,
                    "beacon_parallel_window": self.config.beacon_parallel_window,
                },
                indent=2,
            )
        )


def _is_qwen3_official_checkpoint(path: str | Path) -> bool:
    return (Path(path) / "activation_beacon_qwen3_official.pt").is_file()


def _read_activation_beacon_backend(path: str | Path) -> str | None:
    metadata_path = Path(path) / "activation_beacon_backend.json"
    if not metadata_path.is_file():
        return None
    try:
        return json.loads(metadata_path.read_text()).get("backend")
    except json.JSONDecodeError:
        return None


def _load_qwen3_official_adapter(model: OfficialQwen3ActivationBeacon, checkpoint_path: str | Path) -> None:
    adapter_path = Path(checkpoint_path) / "activation_beacon_qwen3_official.pt"
    if not adapter_path.is_file():
        return
    state = torch.load(adapter_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected_beacon = [key for key in unexpected if "beacon_" in key]
    if unexpected_beacon:
        raise RuntimeError(f"Unexpected qwen3_official Beacon keys: {unexpected_beacon[:10]}")
    missing_beacon = [key for key in missing if "beacon_" in key]
    if missing_beacon:
        raise RuntimeError(f"Missing qwen3_official Beacon keys: {missing_beacon[:10]}")


def _settings_with_defaults(settings: dict, metadata: dict | None = None) -> dict:
    metadata = metadata or {}
    merged = dict(settings)
    for key in (
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
        if merged.get(key) is None and key in metadata:
            merged[key] = metadata[key]
    def set_default(key, value):
        if merged.get(key) is None:
            merged[key] = value

    set_default("beacon_window", 128)
    set_default("beacon_stride", merged["beacon_window"])
    set_default("beacon_attn", "full-coverage")
    set_default("beacon_ratio", [2, 4, 8])
    set_default("beacon_ratio_mix", "step-random")
    set_default("beacon_param", ["q", "k", "v"])
    set_default("beacon_embed_init", "eos")
    set_default("beacon_sink_size", 0)
    set_default("beacon_attend_prev", True)
    set_default("beacon_pos", "interleave")
    set_default("beacon_parallel_window", 1)
    return merged


def load_trainable_qwen3_official_beacon_model(config: dict, tokenizer=None, backend_metadata: dict | None = None):
    settings = _settings_with_defaults(config["model"], backend_metadata)
    if tokenizer is None:
        tokenizer = load_beacon_tokenizer({"model": settings})

    checkpoint_path = Path(settings["base_model_name_or_path"])
    is_adapter_checkpoint = _is_qwen3_official_checkpoint(checkpoint_path)
    source_backend = _read_activation_beacon_backend(checkpoint_path)
    if source_backend and source_backend != "qwen3_official" and not is_adapter_checkpoint:
        raise ValueError(
            "qwen3_official cannot load an Activation Beacon checkpoint created by "
            f"backend={source_backend!r}. Those Beacon parameters and, for older "
            "qwen3_simple checkpoints, wrapped base projection keys are not compatible "
            "with the official recurrent memory backend. Use the plain Qwen3 checkpoint "
            "as model.base_model_name_or_path for from-scratch official-AB finetuning, "
            "or first run qwen3_official pretraining and point to that qwen3_official "
            "checkpoint. This guard prevents Q/K/V weights from being randomly initialized."
        )
    base_model_name_or_path = (
        settings.get("beacon_code_model")
        if is_adapter_checkpoint and settings.get("beacon_code_model")
        else settings["base_model_name_or_path"]
    )
    base_config = AutoConfig.from_pretrained(
        base_model_name_or_path,
        trust_remote_code=True,
        local_files_only=bool(settings.get("local_files_only", False)),
    )
    if getattr(base_config, "model_type", None) != "qwen3":
        raise ValueError(
            "model.beacon_backend='qwen3_official' requires a Qwen3 checkpoint; "
            f"got model_type={getattr(base_config, 'model_type', None)!r}."
        )
    dtype = _dtype_from_name(settings.get("dtype", "bf16"))
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        attn_implementation=settings.get("attn_implementation", "sdpa"),
        local_files_only=bool(settings.get("local_files_only", False)),
    )
    model = OfficialQwen3ActivationBeacon(base_model, settings, tokenizer=tokenizer)
    if is_adapter_checkpoint:
        _load_qwen3_official_adapter(model, checkpoint_path)
        print(f"Loaded Qwen3 official Activation Beacon adapter from {checkpoint_path}.")
    for parameter in model.base_model.parameters():
        parameter.requires_grad = False
    stats = freeze_non_beacon_parameters(model)
    ensure_beacon_memory_dtype(model)
    return model, tokenizer, stats


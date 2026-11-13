# Adapted from Activation Beacon (https://github.com/FlagOpen/FlagEmbedding, research/Long_LLM/activation_beacon),
# Copyright (c) 2022 staoxiao, MIT License. Modified for the Boa baseline reproduction; see THIRD_PARTY_NOTICES.md.
"""Qwen3-native Activation Beacon backend."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM

from .training import (
    ensure_beacon_memory_dtype,
    freeze_non_beacon_parameters,
    load_beacon_tokenizer,
)


class _NoOpBeaconMemory:
    """Compatibility shim for the pretraining loop used by the Qwen3 path."""

    def __init__(self, config):
        self.config = config

    def reset(self) -> None:
        return None

    def get_memory_size(self):
        return 0, 0, 0


class BeaconLinear(nn.Module):
    """Frozen base linear plus trainable Beacon-only delta projection.

    Qwen3 keeps its native attention implementation. We replace q/k/v linears
    with this wrapper so non-beacon tokens see the frozen base projection, while
    inserted beacon tokens get an additional trainable projection. This mirrors
    the official Activation Beacon idea of training beacon-specific attention
    parameters without modifying the rest of the backbone.
    """

    def __init__(self, base_linear: nn.Linear):
        super().__init__()
        self.base_linear = base_linear
        self.beacon_projection = nn.Linear(
            base_linear.in_features,
            base_linear.out_features,
            bias=base_linear.bias is not None,
        )
        nn.init.zeros_(self.beacon_projection.weight)
        if self.beacon_projection.bias is not None:
            nn.init.zeros_(self.beacon_projection.bias)
        self._beacon_mask: torch.Tensor | None = None

    @property
    def weight(self):
        return self.base_linear.weight

    @property
    def bias(self):
        return self.base_linear.bias

    def set_beacon_mask(self, mask: torch.Tensor | None) -> None:
        self._beacon_mask = mask

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = self.base_linear(hidden_states)
        if self._beacon_mask is None:
            return output
        mask = self._beacon_mask
        if mask.shape[:2] != hidden_states.shape[:2]:
            # During generation, the prefill call sees the full expanded prompt
            # and uses the Beacon mask. Subsequent cached decoding steps see one
            # token at a time; those generated tokens are not Beacon tokens.
            if hidden_states.shape[1] == 1:
                return output
            raise RuntimeError(
                "Beacon mask shape does not match hidden states: "
                f"mask={tuple(mask.shape)}, hidden={tuple(hidden_states.shape)}"
            )
        projection_dtype = self.beacon_projection.weight.dtype
        delta = self.beacon_projection(hidden_states.to(dtype=projection_dtype))
        delta = delta.to(dtype=output.dtype)
        return output + delta * mask.to(dtype=output.dtype, device=output.device).unsqueeze(-1)


def _normalize_beacon_ratios(value, *, beacon_window: int) -> list[int]:
    if value is None:
        return [int(beacon_window)]
    if isinstance(value, str):
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            value = [part.strip() for part in value[1:-1].split(",") if part.strip()]
        else:
            value = [value]
    elif isinstance(value, int):
        value = [value]
    ratios = [int(item) for item in value]
    if not ratios:
        raise ValueError("beacon_ratio must contain at least one ratio.")
    for ratio in ratios:
        if ratio <= 0:
            raise ValueError(f"beacon_ratio values must be positive; got {ratio}.")
        if ratio > beacon_window:
            raise ValueError(
                f"beacon_ratio={ratio} cannot exceed beacon_window={beacon_window}."
            )
        if beacon_window % ratio != 0:
            raise ValueError(
                f"beacon_window={beacon_window} must be divisible by beacon_ratio={ratio}."
            )
    return ratios


class SimpleQwen3ActivationBeacon(nn.Module):
    """Qwen3-native Activation Beacon wrapper.

    This backend keeps Qwen3 loaded by native Transformers classes, freezes the
    backbone, inserts learned beacon tokens into the causal stream, and trains
    per-layer Beacon-only q/k/v attention projections selected by
    ``model.beacon_param``. It is separate from the official Qwen2 remote-code
    backend because Qwen3 attention contains q/k normalization modules that the
    Qwen2 wrapper does not model.
    """

    def __init__(self, base_model, config: dict):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.memory = _NoOpBeaconMemory(self.config)
        self.beacon_window = int(config["beacon_window"])
        self.beacon_stride = int(config.get("beacon_stride", self.beacon_window))
        if self.beacon_stride != self.beacon_window:
            raise ValueError(
                "qwen3_simple currently requires beacon_stride == beacon_window."
            )
        self.beacon_ratios = _normalize_beacon_ratios(
            config.get("beacon_ratio"), beacon_window=self.beacon_window
        )
        self.beacon_ratio_mix = str(config.get("beacon_ratio_mix", "fixed"))
        self.last_beacon_ratio = self.beacon_ratios[0]
        hidden_size = int(base_model.config.hidden_size)
        self.beacon_embedding = nn.Parameter(torch.empty(1, hidden_size))
        self.beacon_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.normal_(
            self.beacon_embedding,
            mean=0.0,
            std=float(getattr(base_model.config, "initializer_range", 0.02)),
        )
        nn.init.zeros_(self.beacon_projection.weight)
        self.beacon_attention_projections = list(config.get("beacon_param", ["q", "k", "v"]))
        self._beacon_linears: list[BeaconLinear] = []
        self._wrap_attention_projections()

    def _wrap_attention_projections(self) -> None:
        projection_names = {
            "q": "q_proj",
            "k": "k_proj",
            "v": "v_proj",
            "o": "o_proj",
        }
        layers = getattr(getattr(self.base_model, "model", None), "layers", None)
        if layers is None:
            raise ValueError("Expected Qwen3 model to expose base_model.model.layers.")
        for layer in layers:
            attention = layer.self_attn
            for param in self.beacon_attention_projections:
                name = projection_names.get(param)
                if name is None:
                    raise ValueError(f"Unsupported Qwen3 Beacon projection: {param!r}")
                original = getattr(attention, name)
                if isinstance(original, BeaconLinear):
                    wrapped = original
                else:
                    wrapped = BeaconLinear(original)
                    setattr(attention, name, wrapped)
                self._beacon_linears.append(wrapped)

    def _set_beacon_mask(self, mask: torch.Tensor | None) -> None:
        for linear in self._beacon_linears:
            linear.set_beacon_mask(mask)

    def initialize_beacon_from_token(self, token_id: int) -> None:
        with torch.no_grad():
            source = self.base_model.get_input_embeddings().weight[token_id]
            self.beacon_embedding.copy_(source.detach().to(self.beacon_embedding.dtype))

    def _beacon_token(self, dtype: torch.dtype) -> torch.Tensor:
        beacon = self.beacon_embedding.to(dtype)
        projection_dtype = self.beacon_projection.weight.dtype
        delta = self.beacon_projection(beacon.to(dtype=projection_dtype))
        return beacon + delta.to(dtype=dtype)

    def _select_beacon_ratio(self) -> int:
        if len(self.beacon_ratios) == 1:
            ratio = self.beacon_ratios[0]
        elif self.training and self.beacon_ratio_mix == "step-random":
            index = torch.randint(len(self.beacon_ratios), (), device=self.beacon_embedding.device)
            ratio = self.beacon_ratios[int(index.item())]
        else:
            # Evaluation should pass one explicit ratio. If a list slips through,
            # choose the least compressed option to keep behavior conservative.
            ratio = min(self.beacon_ratios)
        self.last_beacon_ratio = int(ratio)
        return int(ratio)

    def _num_beacons_for_chunk(self, chunk_length: int, ratio: int) -> int:
        if chunk_length < self.beacon_window:
            return 0
        return max(1, self.beacon_window // int(ratio))

    def _expand_with_beacons(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        labels: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
        embeddings = self.base_model.get_input_embeddings()(input_ids)
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)
        expanded_embeddings = []
        expanded_masks = []
        expanded_beacon_masks = []
        expanded_labels = [] if labels is not None else None

        beacon_ratio = self._select_beacon_ratio()

        for row in range(input_ids.shape[0]):
            active = attention_mask[row].bool()
            sequence = embeddings[row, active]
            row_labels = labels[row, active] if labels is not None else None
            pieces = []
            mask_pieces = []
            beacon_mask_pieces = []
            label_pieces = [] if labels is not None else None
            for start in range(0, sequence.shape[0], self.beacon_window):
                chunk = sequence[start : start + self.beacon_window]
                pieces.append(chunk)
                mask_pieces.append(
                    torch.ones(chunk.shape[0], dtype=torch.long, device=input_ids.device)
                )
                beacon_mask_pieces.append(
                    torch.zeros(chunk.shape[0], dtype=torch.bool, device=input_ids.device)
                )
                if labels is not None:
                    label_pieces.append(row_labels[start : start + self.beacon_window])
                num_beacons = self._num_beacons_for_chunk(chunk.shape[0], beacon_ratio)
                if num_beacons:
                    pieces.append(self._beacon_token(sequence.dtype).expand(num_beacons, -1))
                    mask_pieces.append(
                        torch.ones(num_beacons, dtype=torch.long, device=input_ids.device)
                    )
                    beacon_mask_pieces.append(
                        torch.ones(num_beacons, dtype=torch.bool, device=input_ids.device)
                    )
                    if labels is not None:
                        label_pieces.append(
                            torch.full(
                                (num_beacons,),
                                -100,
                                dtype=row_labels.dtype,
                                device=input_ids.device,
                            )
                        )
            expanded_embeddings.append(torch.cat(pieces, dim=0))
            expanded_masks.append(torch.cat(mask_pieces, dim=0))
            expanded_beacon_masks.append(torch.cat(beacon_mask_pieces, dim=0))
            if labels is not None:
                expanded_labels.append(torch.cat(label_pieces, dim=0))

        max_length = max(item.shape[0] for item in expanded_embeddings)
        padded_embeddings = []
        padded_masks = []
        padded_beacon_masks = []
        padded_labels = [] if labels is not None else None
        for index, item in enumerate(expanded_embeddings):
            padding = max_length - item.shape[0]
            if padding:
                item = torch.cat(
                    [
                        item,
                        torch.zeros(
                            padding,
                            item.shape[-1],
                            dtype=item.dtype,
                            device=item.device,
                        ),
                    ],
                    dim=0,
                )
            padded_embeddings.append(item)
            mask = expanded_masks[index]
            beacon_mask = expanded_beacon_masks[index]
            if padding:
                mask = torch.cat(
                    [
                        mask,
                        torch.zeros(padding, dtype=torch.long, device=mask.device),
                    ],
                    dim=0,
                )
                beacon_mask = torch.cat(
                    [
                        beacon_mask,
                        torch.zeros(padding, dtype=torch.bool, device=beacon_mask.device),
                    ],
                    dim=0,
                )
            padded_masks.append(mask)
            padded_beacon_masks.append(beacon_mask)
            if labels is not None:
                label = expanded_labels[index]
                if padding:
                    label = torch.cat(
                        [
                            label,
                            torch.full(
                                (padding,),
                                -100,
                                dtype=label.dtype,
                                device=label.device,
                            ),
                        ],
                        dim=0,
                    )
                padded_labels.append(label)
        return (
            torch.stack(padded_embeddings, dim=0),
            torch.stack(padded_masks, dim=0),
            torch.stack(padded_labels, dim=0) if labels is not None else None,
            torch.stack(padded_beacon_masks, dim=0),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ):
        inputs_embeds, expanded_mask, expanded_labels, beacon_mask = self._expand_with_beacons(
            input_ids, attention_mask, labels
        )
        self._set_beacon_mask(beacon_mask)
        try:
            return self.base_model(
                inputs_embeds=inputs_embeds,
                attention_mask=expanded_mask,
                labels=expanded_labels,
                **kwargs,
            )
        finally:
            self._set_beacon_mask(None)

    def save_pretrained(
        self,
        save_directory,
        state_dict: dict[str, torch.Tensor] | None = None,
        safe_serialization: bool = True,
        **kwargs,
    ) -> None:
        destination = Path(save_directory)
        destination.mkdir(parents=True, exist_ok=True)
        base_state = None
        beacon_state = None
        if state_dict is not None:
            base_state = {
                key.removeprefix("base_model."): value
                for key, value in state_dict.items()
                if key.startswith("base_model.") and ".beacon_" not in key
            }
            base_state = _unwrap_beacon_linear_base_state(base_state)
            beacon_state = {
                key: value.detach().cpu()
                for key, value in state_dict.items()
                if "beacon_" in key
            }
        self.base_model.save_pretrained(
            destination,
            state_dict=base_state,
            safe_serialization=safe_serialization,
            **kwargs,
        )
        if beacon_state is None:
            beacon_state = {
                key: value.detach().cpu()
                for key, value in self.state_dict().items()
                if "beacon_" in key
            }
        torch.save(beacon_state, destination / "activation_beacon_qwen3_simple.pt")
        (destination / "activation_beacon_backend.json").write_text(
            json.dumps(
                {
                    "backend": "qwen3_simple",
                    "beacon_window": self.beacon_window,
                    "beacon_stride": self.beacon_stride,
                    "beacon_ratio": self.beacon_ratios,
                    "beacon_ratio_mix": self.beacon_ratio_mix,
                    "beacon_param": self.beacon_attention_projections,
                },
                indent=2,
            )
        )


def _unwrap_beacon_linear_base_state(
    state_dict: dict[str, torch.Tensor] | None,
) -> dict[str, torch.Tensor] | None:
    """Convert wrapped BeaconLinear base keys back to native Qwen3 keys.

    ``SimpleQwen3ActivationBeacon`` replaces q/k/v projections with
    ``BeaconLinear`` wrappers. The wrapped module's frozen base weight is named
    ``q_proj.base_linear.weight`` inside ``state_dict``. Native Qwen3 checkpoints
    expect ``q_proj.weight``. Without this conversion, a later
    ``Qwen3ForCausalLM.from_pretrained`` silently initializes Q/K/V from scratch.
    """
    if state_dict is None:
        return None
    converted = {}
    for key, value in state_dict.items():
        if ".beacon_projection." in key:
            continue
        converted[key.replace(".base_linear.", ".")] = value
    return converted


def _is_qwen3_simple_checkpoint(path: str | Path) -> bool:
    return (Path(path) / "activation_beacon_qwen3_simple.pt").is_file()


def _restore_wrapped_base_linears(base_model, checkpoint_path: str | Path) -> int:
    """Restore native Qwen3 q/k/v weights from old wrapped-key checkpoints."""
    checkpoint_path = Path(checkpoint_path)
    safetensor_files = sorted(checkpoint_path.glob("*.safetensors"))
    if not safetensor_files:
        return 0
    parameters = dict(base_model.named_parameters())
    restored = 0
    with torch.no_grad():
        for file_path in safetensor_files:
            with safe_open(file_path, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    if ".base_linear." not in key:
                        continue
                    target = key.replace(".base_linear.", ".")
                    parameter = parameters.get(target)
                    if parameter is None:
                        continue
                    parameter.copy_(
                        handle.get_tensor(key).to(parameter.device, parameter.dtype)
                    )
                    restored += 1
    return restored


def _load_qwen3_simple_adapter(
    model: SimpleQwen3ActivationBeacon, checkpoint_path: str | Path
) -> None:
    adapter_path = Path(checkpoint_path) / "activation_beacon_qwen3_simple.pt"
    if not adapter_path.is_file():
        return
    state = torch.load(adapter_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected = [key for key in unexpected if "base_linear" not in key]
    if unexpected:
        raise RuntimeError(
            "Unexpected keys while loading Qwen3 Activation Beacon adapter: "
            f"{unexpected[:10]}"
        )
    missing_beacon = [key for key in missing if "beacon_" in key]
    if missing_beacon:
        raise RuntimeError(
            "Missing Beacon keys while loading Qwen3 Activation Beacon adapter: "
            f"{missing_beacon[:10]}"
        )

def load_trainable_qwen3_simple_beacon_model(config: dict, tokenizer=None):
    settings = config["model"]
    if tokenizer is None:
        tokenizer = load_beacon_tokenizer(config)

    checkpoint_path = Path(settings["base_model_name_or_path"])
    is_adapter_checkpoint = _is_qwen3_simple_checkpoint(checkpoint_path)
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
            "model.beacon_backend='qwen3_simple' requires a Qwen3 checkpoint; "
            f"got model_type={getattr(base_config, 'model_type', None)!r}."
        )
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[settings.get("dtype", "bf16")]
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        attn_implementation=settings.get("attn_implementation", "sdpa"),
        local_files_only=bool(settings.get("local_files_only", False)),
    )
    if is_adapter_checkpoint and base_model_name_or_path == settings["base_model_name_or_path"]:
        restored = _restore_wrapped_base_linears(base_model, checkpoint_path)
        if restored:
            print(f"Restored {restored} wrapped Qwen3 base projection tensors.")
    model = SimpleQwen3ActivationBeacon(base_model, settings)
    if is_adapter_checkpoint:
        _load_qwen3_simple_adapter(model, checkpoint_path)
        print(f"Loaded Qwen3 Activation Beacon adapter from {checkpoint_path}.")
    elif settings.get("beacon_embed_init") == "eos" and tokenizer.eos_token_id is not None:
        model.initialize_beacon_from_token(int(tokenizer.eos_token_id))
    for parameter in model.base_model.parameters():
        parameter.requires_grad = False
    stats = freeze_non_beacon_parameters(model)
    ensure_beacon_memory_dtype(model)
    return model, tokenizer, stats

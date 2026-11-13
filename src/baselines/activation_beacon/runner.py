# Adapted from Activation Beacon (https://github.com/FlagOpen/FlagEmbedding, research/Long_LLM/activation_beacon),
# Copyright (c) 2022 staoxiao, MIT License. Modified for the Boa baseline reproduction; see THIRD_PARTY_NOTICES.md.
"""MRQA benchmark adapter for Activation Beacon checkpoints."""

from __future__ import annotations

import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import MethodType

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .prompting import build_qa_inputs


def _torch_dtype(name: str) -> torch.dtype:
    aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return aliases[name.lower()]
    except KeyError as error:
        raise ValueError(f"Unsupported dtype {name!r}") from error


def _prepare_4d_causal_mask(
    attention_mask: torch.Tensor,
    query_length: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Convert a 2-D padding mask to the additive mask expected by SDPA/eager."""
    batch_size, key_length = attention_mask.shape
    past_length = key_length - query_length
    if past_length < 0:
        raise ValueError(
            f"Attention-mask length {key_length} is shorter than query length "
            f"{query_length}."
        )

    device = attention_mask.device
    query_positions = (
        torch.arange(query_length, device=device) + past_length
    ).view(query_length, 1)
    key_positions = torch.arange(key_length, device=device).view(1, key_length)
    causal = key_positions <= query_positions
    valid_keys = attention_mask.to(torch.bool)[:, None, None, :]
    allowed = causal[None, None, :, :] & valid_keys

    additive_mask = torch.zeros(
        (batch_size, 1, query_length, key_length),
        dtype=dtype,
        device=device,
    )
    return additive_mask.masked_fill(~allowed, torch.finfo(dtype).min)


def _prepare_full_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    """Build positions for every cached key because the checkpoint caches pre-RoPE K."""
    position_ids = attention_mask.long().cumsum(-1) - 1
    return position_ids.masked_fill(attention_mask == 0, 1)


def _merge_native_kv_cache(previous_cache, current_cache, query_length: int):
    """Make the checkpoint's incremental cache cumulative for HF generation."""
    if previous_cache is None or current_cache is None:
        return current_cache

    merged_cache = []
    for previous_layer, current_layer in zip(previous_cache, current_cache):
        previous_key, previous_value, _, _ = previous_layer
        current_key, current_value, beacon_size, beacon_indices = current_layer
        if previous_key is None:
            merged_cache.append(current_layer)
            continue

        previous_length = previous_key.shape[2]
        current_length = current_key.shape[2]
        expected_length = previous_length + query_length
        if current_length == expected_length:
            merged_cache.append(current_layer)
        elif current_length == query_length:
            merged_cache.append(
                (
                    torch.cat((previous_key, current_key), dim=2),
                    torch.cat((previous_value, current_value), dim=2),
                    beacon_size,
                    beacon_indices,
                )
            )
        else:
            raise ValueError(
                "Unexpected native KV-cache length: "
                f"previous={previous_length}, current={current_length}, "
                f"query={query_length}."
            )
    return tuple(merged_cache)




def _load_backend_metadata(model_path: str | Path) -> dict:
    metadata_path = Path(model_path) / "activation_beacon_backend.json"
    if not metadata_path.is_file():
        return {}
    try:
        return json.loads(metadata_path.read_text())
    except json.JSONDecodeError:
        return {}

def _install_native_attention_mask_adapter(model) -> None:
    """Adapt the checkpoint's non-beacon path to HF generation conventions.

    Activation Beacon's memory path already builds 4-D masks and intentionally
    consumes incremental KV tensors. When beacon processing is disabled, however,
    Hugging Face generation expects a 4-D SDPA/eager mask and a cumulative cache.
    The published checkpoint provides neither, so both are adapted here without
    changing the cached remote code or the beacon path.
    """
    if getattr(model, "_native_attention_mask_adapter_installed", False):
        return

    original_native_forward = model._native_forward

    def native_forward_with_4d_mask(self, *args, **kwargs):
        attention_mask = kwargs.get("attention_mask")
        native_generation = attention_mask is not None and attention_mask.ndim == 2
        previous_cache = kwargs.get("past_key_values")
        query_length = None
        if native_generation:
            input_ids = kwargs.get("input_ids")
            inputs_embeds = kwargs.get("inputs_embeds")
            if input_ids is not None:
                query_length = input_ids.shape[1]
            elif inputs_embeds is not None:
                query_length = inputs_embeds.shape[1]
            elif args:
                query_length = args[0].shape[1]
            else:
                raise ValueError(
                    "Cannot infer query length while adapting the attention mask."
                )
            # This checkpoint stores keys before RoPE and reapplies RoPE to the
            # complete key tensor on every call. HF generation normally supplies
            # only the newest position when a cache exists, so reconstruct all
            # positions before forwarding the cumulative native cache.
            kwargs["position_ids"] = _prepare_full_position_ids(attention_mask)
            kwargs["attention_mask"] = _prepare_4d_causal_mask(
                attention_mask,
                query_length=query_length,
                dtype=self.model.embed_tokens.weight.dtype,
            )

        outputs = original_native_forward(*args, **kwargs)
        if (
            native_generation
            and previous_cache is not None
            and hasattr(outputs, "past_key_values")
        ):
            outputs.past_key_values = _merge_native_kv_cache(
                previous_cache,
                outputs.past_key_values,
                query_length=query_length,
            )
        return outputs

    model._native_forward = MethodType(native_forward_with_4d_mask, model)
    model._native_attention_mask_adapter_installed = True


class ActivationBeaconRunner:
    """Expose Activation Beacon through the same interface as ``ModelRunner``.

    ``run_teacher`` disables beacon processing and runs the checkpoint natively.
    ``run_student`` enables recurrent beacon compression. Samples are processed one
    at a time because the upstream memory controller owns per-sequence state.
    """

    def __init__(self, config: dict, device: str = "cuda"):
        self.settings = dict(config)
        self.device = torch.device(device)
        self.model_name_or_path = self.settings["model_name_or_path"]
        self.teacher_model_name_or_path = self.settings.get("teacher_model_name_or_path")
        self.teacher_model = None
        self.config = {
            "llm_model_name": self.model_name_or_path,
            "teacher_model_name": self.teacher_model_name_or_path or self.model_name_or_path,
        }
        self.llm_name_safe = Path(self.model_name_or_path).name
        self.run_name = self.settings.get(
            "run_name", f"activation_beacon_{self.llm_name_safe}"
        )
        self.max_input_tokens = int(self.settings.get("max_input_tokens", 20_000))
        self.max_new_tokens = int(self.settings.get("max_new_tokens", 80))
        self.newline_as_eos = bool(self.settings.get("newline_as_eos", False))
        self.local_files_only = bool(self.settings.get("local_files_only", False))
        self.last_run_stats: list[dict] = []
        self.backend = "qwen2_remote"
        self.model, self.tokenizer = self._load_resources()

    def _load_resources(self):
        model_path = self.model_name_or_path
        dtype = _torch_dtype(self.settings.get("dtype", "bf16"))

        if self.settings.get("attn_implementation", "sdpa") in {"sdpa", "eager"}:
            compat_root = Path(__file__).with_name("compat")
            if str(compat_root) not in sys.path:
                sys.path.insert(0, str(compat_root))

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side="left",
            trust_remote_code=True,
            local_files_only=self.local_files_only,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=self.local_files_only,
        )
        architecture = (getattr(config, "architectures", None) or [None])[0]
        backend_metadata = _load_backend_metadata(model_path)
        requested_backend = self.settings.get("beacon_backend") or backend_metadata.get("backend")
        if requested_backend == "qwen3_official":
            return self._load_qwen3_official_resources(tokenizer, backend_metadata)
        if architecture == "Qwen3ForCausalLM" or backend_metadata.get("backend") == "qwen3_simple":
            return self._load_qwen3_resources(tokenizer, backend_metadata)
        if architecture != "Qwen2ForCausalLM":
            raise ValueError(
                "The Activation Beacon benchmark expects Qwen2ForCausalLM, "
                "a qwen3_simple checkpoint, or a qwen3_official checkpoint; "
                f"checkpoint reports {architecture!r}."
            )
        if not getattr(config, "beacon_param", None) and not self.settings.get(
            "allow_untrained_beacon", False
        ):
            raise ValueError(
                "This checkpoint does not advertise trained beacon parameters. "
                "Use the authors' published Activation Beacon checkpoint."
            )

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
            if self.settings.get(key) is not None:
                setattr(config, key, self.settings[key])

        # The published checkpoint omits pad_token_id in some revisions. Reconcile
        # special IDs from its tokenizer before handing control to its remote code.
        # Inference is single-sample and unpadded. Keep the model padding index unset:
        # the authors' one-row beacon embedding cannot use Qwen's vocabulary pad id.
        config.pad_token_id = None
        if getattr(config, "eos_token_id", None) is None:
            config.eos_token_id = tokenizer.eos_token_id
        if getattr(config, "bos_token_id", None) is None:
            config.bos_token_id = tokenizer.bos_token_id

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            trust_remote_code=True,
            torch_dtype=dtype,
            attn_implementation=self.settings.get("attn_implementation", "sdpa"),
            device_map={"": str(self.device)},
            local_files_only=self.local_files_only,
        )
        if not hasattr(model, "memory"):
            raise TypeError(
                "The loaded checkpoint did not expose Activation Beacon memory. "
                "Check trust_remote_code and the checkpoint path."
            )
        if config._attn_implementation in {"sdpa", "eager"}:
            _install_native_attention_mask_adapter(model)
        model.eval().requires_grad_(False)
        return model, tokenizer

    def _load_qwen3_resources(self, tokenizer, backend_metadata: dict):
        from .qwen3_simple import load_trainable_qwen3_simple_beacon_model

        self.backend = "qwen3_simple"
        settings = dict(self.settings)
        settings["base_model_name_or_path"] = self.model_name_or_path
        # qwen3_simple checkpoints store these values in activation_beacon_backend.json.
        for config_key, metadata_key in (
            ("beacon_window", "beacon_window"),
            ("beacon_stride", "beacon_stride"),
            ("beacon_ratio", "beacon_ratio"),
            ("beacon_ratio_mix", "beacon_ratio_mix"),
            ("beacon_param", "beacon_param"),
        ):
            if settings.get(config_key) is None and metadata_key in backend_metadata:
                settings[config_key] = backend_metadata[metadata_key]
        if settings.get("beacon_window") is None:
            settings["beacon_window"] = 512
        if settings.get("beacon_stride") is None:
            settings["beacon_stride"] = settings["beacon_window"]
        if settings.get("beacon_ratio") is None:
            settings["beacon_ratio"] = settings["beacon_window"]
        if settings.get("beacon_ratio_mix") is None:
            settings["beacon_ratio_mix"] = "fixed"
        if settings.get("beacon_param") is None:
            settings["beacon_param"] = ["q", "k", "v"]
        model, tokenizer, _ = load_trainable_qwen3_simple_beacon_model(
            {"model": settings},
            tokenizer=tokenizer,
        )
        model.to(self.device)
        model.eval().requires_grad_(False)
        if self.teacher_model_name_or_path:
            self.teacher_model = self._load_qwen3_teacher_model()
        return model, tokenizer

    def _load_qwen3_official_resources(self, tokenizer, backend_metadata: dict):
        from .qwen3_official import load_trainable_qwen3_official_beacon_model

        self.backend = "qwen3_official"
        settings = dict(self.settings)
        settings["base_model_name_or_path"] = self.model_name_or_path
        settings["beacon_backend"] = "qwen3_official"
        model, tokenizer, _ = load_trainable_qwen3_official_beacon_model(
            {"model": settings},
            tokenizer=tokenizer,
            backend_metadata=backend_metadata,
        )
        model.to(self.device)
        model.eval().requires_grad_(False)
        if self.teacher_model_name_or_path:
            self.teacher_model = self._load_qwen3_teacher_model()
        return model, tokenizer

    def _load_qwen3_teacher_model(self):
        teacher_path = self.teacher_model_name_or_path
        dtype = _torch_dtype(self.settings.get("teacher_dtype", self.settings.get("dtype", "bf16")))
        teacher_config = AutoConfig.from_pretrained(
            teacher_path,
            trust_remote_code=True,
            local_files_only=self.local_files_only,
        )
        if getattr(teacher_config, "model_type", None) != "qwen3":
            raise ValueError(
                "activation_beacon.teacher_model_name_or_path must point to a Qwen3 checkpoint; "
                f"got model_type={getattr(teacher_config, 'model_type', None)!r}."
            )
        teacher = AutoModelForCausalLM.from_pretrained(
            teacher_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            attn_implementation=self.settings.get("attn_implementation", "sdpa"),
            local_files_only=self.local_files_only,
        )
        teacher.to(self.device)
        teacher.eval().requires_grad_(False)
        return teacher

    @contextmanager
    def _beacon_mode(self, enabled: bool):
        previous = getattr(self.model, "_enable_beacon", True)
        self.model._enable_beacon = enabled
        self.model.memory.reset()
        try:
            yield
        finally:
            self.model._enable_beacon = previous
            self.model.memory.reset()


    def _build_eos_token_ids(self, model=None):
        eos_token_id = None
        model = model or self.model
        if getattr(self, "newline_as_eos", False):
            generation_config = getattr(model, "generation_config", None)
            if generation_config is None and hasattr(model, "base_model"):
                generation_config = getattr(model.base_model, "generation_config", None)
            base_eos = getattr(generation_config, "eos_token_id", None)
            if base_eos is None:
                base_eos = self.tokenizer.eos_token_id
            eos_token_id = base_eos if isinstance(base_eos, list) else [base_eos]
            newline_ids = self.tokenizer.encode("\n", add_special_tokens=False)
            if newline_ids:
                eos_token_id = [token for token in eos_token_id if token is not None]
                eos_token_id.append(newline_ids[-1])
        return eos_token_id

    def _generate_qwen3_one(self, sample: dict, *, use_context: bool, beacon: bool) -> str:
        context = sample.get("context", "") if use_context else ""
        inputs = build_qa_inputs(
            self.tokenizer,
            sample["question"],
            context,
            self.max_input_tokens,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        generation_model = self.model.base_model
        if not beacon and self.teacher_model is not None:
            generation_model = self.teacher_model
        eos_token_id = self._build_eos_token_ids(generation_model)
        if beacon:
            inputs_embeds, expanded_mask, _, beacon_mask = self.model._expand_with_beacons(
                inputs["input_ids"],
                inputs["attention_mask"],
                labels=None,
            )
            prompt_length = expanded_mask.shape[1]
            dummy_token = self.tokenizer.eos_token_id
            if dummy_token is None:
                dummy_token = self.tokenizer.pad_token_id
            if dummy_token is None:
                dummy_token = 0
            expanded_input_ids = torch.full(
                expanded_mask.shape,
                int(dummy_token),
                dtype=inputs["input_ids"].dtype,
                device=self.device,
            )
            self.model._set_beacon_mask(beacon_mask)
            try:
                outputs = generation_model.generate(
                    input_ids=expanded_input_ids,
                    inputs_embeds=inputs_embeds,
                    attention_mask=expanded_mask,
                    do_sample=False,
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=eos_token_id,
                )
            finally:
                self.model._set_beacon_mask(None)
            generated = outputs[0, prompt_length:] if outputs.shape[1] > prompt_length else outputs[0]
            memory_sizes = self.model.memory.get_memory_size()
            input_length = inputs["input_ids"].shape[1]
        else:
            input_length = inputs["input_ids"].shape[1]
            outputs = generation_model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=eos_token_id,
            )
            generated = outputs[0, input_length:]
            memory_sizes = (0, 0, 0)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.last_run_stats.append(
            {
                "input_tokens": input_length,
                "output_tokens": int(generated.shape[0]),
                "latency_seconds": time.perf_counter() - started,
                "beacon_enabled": beacon,
                "sink_tokens": int(memory_sizes[0]),
                "beacon_tokens": int(memory_sizes[1]),
                "raw_tokens": int(memory_sizes[2]),
                "beacon_ratio": int(getattr(self.model, "last_beacon_ratio", 0)) if beacon else None,
            }
        )
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def _generate_qwen3_official_one(self, sample: dict, *, use_context: bool, beacon: bool) -> str:
        context = sample.get("context", "") if use_context else ""
        inputs = build_qa_inputs(
            self.tokenizer,
            sample["question"],
            context,
            self.max_input_tokens,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        input_length = inputs["input_ids"].shape[1]
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        eos_token_id = self._build_eos_token_ids(self.model if beacon or self.teacher_model is None else self.teacher_model)
        if beacon:
            with self._beacon_mode(True):
                outputs = self.model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=eos_token_id,
                )
                memory_sizes = self.model.memory.get_memory_size()
        elif self.teacher_model is not None:
            outputs = self.teacher_model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=eos_token_id,
            )
            memory_sizes = (0, 0, 0)
        else:
            with self._beacon_mode(False):
                outputs = self.model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=eos_token_id,
                )
            memory_sizes = (0, 0, 0)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        generated = outputs[0, input_length:]
        self.last_run_stats.append(
            {
                "input_tokens": input_length,
                "output_tokens": int(generated.shape[0]),
                "latency_seconds": time.perf_counter() - started,
                "beacon_enabled": beacon,
                "sink_tokens": int(memory_sizes[0]),
                "beacon_tokens": int(memory_sizes[1]),
                "raw_tokens": int(memory_sizes[2]),
            }
        )
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    @torch.no_grad()
    def _generate_one(self, sample: dict, *, use_context: bool, beacon: bool) -> str:
        backend = getattr(self, "backend", "qwen2_remote")
        if backend == "qwen3_simple":
            return self._generate_qwen3_one(sample, use_context=use_context, beacon=beacon)
        if backend == "qwen3_official":
            return self._generate_qwen3_official_one(sample, use_context=use_context, beacon=beacon)

        context = sample.get("context", "") if use_context else ""
        inputs = build_qa_inputs(
            self.tokenizer,
            sample["question"],
            context,
            self.max_input_tokens,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        input_length = inputs["input_ids"].shape[1]

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        eos_token_id = self._build_eos_token_ids()

        with self._beacon_mode(beacon):
            outputs = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=eos_token_id,
            )
            memory_sizes = self.model.memory.get_memory_size() if beacon else (0, 0, 0)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        generated = outputs[0, input_length:]
        self.last_run_stats.append(
            {
                "input_tokens": input_length,
                "output_tokens": int(generated.shape[0]),
                "latency_seconds": time.perf_counter() - started,
                "beacon_enabled": beacon,
                "sink_tokens": int(memory_sizes[0]),
                "beacon_tokens": int(memory_sizes[1]),
                "raw_tokens": int(memory_sizes[2]),
            }
        )
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def _run(self, samples: list, *, use_context: bool, beacon: bool,
             progress=None, task_id=None) -> list[str]:
        self.last_run_stats = []
        answers = []
        for sample in samples:
            answers.append(
                self._generate_one(sample, use_context=use_context, beacon=beacon)
            )
            if progress is not None and task_id is not None:
                progress.update(task_id, advance=1)
        return answers

    def run_teacher(self, samples: list, use_context: bool = True,
                    batch_size: int = 1, progress=None, task_id=None) -> list[str]:
        del batch_size
        return self._run(
            samples,
            use_context=use_context,
            beacon=False,
            progress=progress,
            task_id=task_id,
        )

    def run_student(self, samples: list, batch_size: int = 1,
                    progress=None, task_id=None) -> list[str]:
        del batch_size
        return self._run(
            samples,
            use_context=True,
            beacon=True,
            progress=progress,
            task_id=task_id,
        )

# Adapted from Activation Beacon (https://github.com/FlagOpen/FlagEmbedding, research/Long_LLM/activation_beacon),
# Copyright (c) 2022 staoxiao, MIT License. Modified for the Boa baseline reproduction; see THIRD_PARTY_NOTICES.md.
"""Training utilities for adapting Activation Beacon to supported Qwen backbones."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import torch
from datasets import Dataset as HFDataset
from datasets import concatenate_datasets, load_from_disk
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer

from ...dataset import (
    EncodedQwenMRQAExample,
    MRQA_FILENAME,
    encode_qwen_mrqa_example,
    mrqa_generator,
)


def encode_mrqa_example(
    tokenizer,
    sample: dict,
    *,
    max_length: int,
    beacon_window: int,
    min_length: int | None = None,
    context_ids: list[int] | None = None,
) -> EncodedQwenMRQAExample | None:
    """Encode MRQA with the repository-wide Qwen prompt and answer labels."""
    encoded = encode_qwen_mrqa_example(
        tokenizer,
        sample,
        max_length=max_length,
        include_answer=True,
        include_context=True,
        context_ids=context_ids,
    )
    required_length = max(beacon_window + 1, min_length or 0)
    if encoded is None or len(encoded.input_ids) < required_length:
        return None
    return encoded


def _resolve_mrqa_root(data_dir: str, split: str) -> Path:
    root = Path(data_dir)
    split_root = root / split
    if all((split_root / filename).is_file() for filename in MRQA_FILENAME.values()):
        return split_root
    if not all((root / filename).is_file() for filename in MRQA_FILENAME.values()):
        missing = [name for name in MRQA_FILENAME.values() if not (root / name).is_file()]
        raise FileNotFoundError(
            f"MRQA files were not found under {root} or {split_root}: {missing}"
        )
    return root


def load_mrqa_records(data_dir: str, split: str, max_samples: int | None) -> list[dict]:
    """Legacy list loader kept for callers outside the training pipeline."""
    root = _resolve_mrqa_root(data_dir, split)
    records: list[dict] = []
    for filename in MRQA_FILENAME.values():
        for sample in mrqa_generator(str(root / filename)):
            records.append(sample)
            if max_samples is not None and len(records) >= max_samples:
                return records
    return records


def _load_mrqa_arrow_dataset(
    data_dir: str,
    split: str,
    max_samples: int | None,
    cache_dir: str | None,
) -> HFDataset:
    if max_samples is not None:
        # Avoid materializing every MRQA split for a capped smoke/overfit run.
        records = load_mrqa_records(data_dir, split, int(max_samples))
        return HFDataset.from_list(records)

    root = _resolve_mrqa_root(data_dir, split)
    parts = [
        HFDataset.from_generator(
            mrqa_generator,
            gen_kwargs={"file_path": str(root / filename)},
            cache_dir=cache_dir,
        )
        for filename in MRQA_FILENAME.values()
    ]
    return concatenate_datasets(parts)


def _encoded_cache_path(tokenizer, data_config: dict, beacon_window: int) -> Path | None:
    cache_root = data_config.get("encoded_cache_dir")
    if not cache_root:
        return None
    root = _resolve_mrqa_root(
        data_config["data_dir"], data_config.get("split", "train")
    )
    sources = []
    for filename in MRQA_FILENAME.values():
        source = root / filename
        stat = source.stat()
        sources.append((str(source.resolve()), stat.st_size, stat.st_mtime_ns))
    fingerprint = {
        "version": 3,
        "tokenizer": getattr(tokenizer, "name_or_path", type(tokenizer).__name__),
        "sources": sources,
        "max_samples": data_config.get("max_samples"),
        "max_length": int(data_config["max_length"]),
        "min_length": data_config.get("min_length"),
        "beacon_window": int(beacon_window),
    }
    digest = sha256(
        json.dumps(fingerprint, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:20]
    return Path(cache_root).expanduser() / f"mrqa-{digest}"


def _encode_mrqa_batch(
    batch: dict[str, list],
    *,
    tokenizer,
    max_length: int,
    beacon_window: int,
    min_length: int | None,
) -> dict[str, list]:
    outputs = {"input_ids": [], "labels": []}
    # Questions from one MRQA passage are adjacent. Reuse the context tokens for
    # all of them inside this Arrow map batch.
    context_cache: dict[str, list[int]] = {}
    for context, question, answer in zip(
        batch["context"], batch["question"], batch["answer"]
    ):
        context_ids = context_cache.get(context)
        if context_ids is None:
            context_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
            context_cache[context] = context_ids
        encoded = encode_mrqa_example(
            tokenizer,
            {"context": context, "question": question, "answer": answer},
            max_length=max_length,
            beacon_window=beacon_window,
            min_length=min_length,
            context_ids=context_ids,
        )
        if encoded is not None:
            outputs["input_ids"].append(encoded.input_ids)
            outputs["labels"].append(encoded.labels)
    return outputs


def build_encoded_datasets(tokenizer, data_config: dict, beacon_window: int):
    cache_path = _encoded_cache_path(tokenizer, data_config, beacon_window)
    overwrite_cache = bool(data_config.get("overwrite_encoded_cache", False))
    metadata_path = cache_path / "metadata.json" if cache_path else None

    if cache_path and cache_path.is_dir() and not overwrite_cache:
        encoded_dataset = load_from_disk(str(cache_path))
        metadata = (
            json.loads(metadata_path.read_text())
            if metadata_path and metadata_path.is_file()
            else {"records": len(encoded_dataset), "retained": len(encoded_dataset)}
        )
        print(f"Loaded encoded MRQA cache: {cache_path}")
    else:
        raw_dataset = _load_mrqa_arrow_dataset(
            data_config["data_dir"],
            data_config.get("split", "train"),
            data_config.get("max_samples"),
            data_config.get("datasets_cache_dir"),
        )
        records = len(raw_dataset)
        num_proc = max(1, int(data_config.get("preprocessing_num_proc", 1)))
        encoded_dataset = raw_dataset.map(
            _encode_mrqa_batch,
            batched=True,
            batch_size=int(data_config.get("preprocessing_batch_size", 64)),
            num_proc=num_proc,
            remove_columns=raw_dataset.column_names,
            fn_kwargs={
                "tokenizer": tokenizer,
                "max_length": int(data_config["max_length"]),
                "beacon_window": int(beacon_window),
                "min_length": data_config.get("min_length"),
            },
            desc="Encoding MRQA for Activation Beacon",
            load_from_cache_file=not overwrite_cache,
        )
        metadata = {"records": records, "retained": len(encoded_dataset)}
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            encoded_dataset.save_to_disk(str(cache_path))
            metadata_path.write_text(json.dumps(metadata, indent=2))
            print(f"Saved encoded MRQA cache: {cache_path}")

    seed = int(data_config.get("seed", 42))
    encoded_dataset = encoded_dataset.shuffle(seed=seed)
    examples = len(encoded_dataset)
    validation_fraction = float(data_config.get("validation_fraction", 0.05))
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1).")
    if examples < 2 and validation_fraction:
        raise ValueError("At least two beacon-active examples are required for validation.")

    validation_size = int(round(examples * validation_fraction))
    if validation_fraction and validation_size == 0:
        validation_size = 1
    validation = encoded_dataset.select(range(validation_size))
    training = encoded_dataset.select(range(validation_size, examples))
    if not len(training):
        raise ValueError("No MRQA examples remain for Activation Beacon training.")
    return training, (validation if len(validation) else None), metadata

def collate_beacon_batch(batch: list[dict], pad_token_id: int) -> dict[str, torch.Tensor]:
    max_length = max(len(item["input_ids"]) for item in batch)
    input_ids, attention_mask, labels = [], [], []
    for item in batch:
        padding = max_length - len(item["input_ids"])
        input_ids.append([pad_token_id] * padding + item["input_ids"])
        attention_mask.append([0] * padding + [1] * len(item["input_ids"]))
        labels.append([-100] * padding + item["labels"])
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def create_beacon_dataloader(
    dataset,
    *,
    batch_size: int,
    collate_fn,
    num_workers: int = 0,
    seed: int = 42,
    shuffle: bool = True,
    accelerator=None,
) -> tuple[DataLoader, DistributedSampler | None]:
    """Build a rank-sharded MRQA dataloader for torchrun/Accelerate DDP.

    Pretraining self-shards its iterable dataset and leaves the loader outside
    ``accelerator.prepare`` to avoid accidental double-sharding. Fine-tuning uses
    a finite Arrow dataset, so it gets the equivalent behavior through an
    explicit ``DistributedSampler``: each rank sees a deterministic disjoint
    slice, and the sampler pads as needed so every rank has the same number of
    batches.
    """
    world_size = int(getattr(accelerator, "num_processes", 1))
    rank = int(getattr(accelerator, "process_index", 0))
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=int(seed),
            drop_last=False,
        )
    num_workers = int(num_workers)
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        collate_fn=collate_fn,
    )
    return loader, sampler


def validate_training_config(config: dict) -> None:
    model = config["model"]
    data = config["data"]
    training = config["training"]
    if int(data["max_length"]) <= int(model["beacon_window"]):
        raise ValueError("data.max_length must be larger than model.beacon_window.")
    if int(data.get("min_length", 0)) > int(data["max_length"]):
        raise ValueError("data.min_length cannot exceed data.max_length.")
    if int(model["beacon_stride"]) > int(model["beacon_window"]):
        raise ValueError("beacon_stride cannot exceed beacon_window.")
    params = set(model.get("beacon_param", []))
    if not params or not params <= {"q", "k", "v", "o"}:
        raise ValueError("beacon_param must contain one or more of q, k, v, o.")
    if int(training.get("batch_size", 1)) < 1:
        raise ValueError("training.batch_size must be positive.")
    if int(data.get("preprocessing_num_proc", 1)) < 1:
        raise ValueError("data.preprocessing_num_proc must be positive.")
    if int(data.get("preprocessing_batch_size", 64)) < 1:
        raise ValueError("data.preprocessing_batch_size must be positive.")


def ensure_beacon_memory_dtype(model) -> torch.dtype:
    """Keep remote Beacon Memory compatible with torch.finfo after wrappers/config IO."""
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
    """Load only the tokenizer so data can be prepared before 7B model weights."""
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


def load_trainable_beacon_model(config: dict, tokenizer=None):
    settings = config["model"]
    backend = settings.get("beacon_backend", "qwen2_remote")
    if backend == "qwen2_remote":
        from .qwen2_official import load_trainable_qwen2_remote_beacon_model

        return load_trainable_qwen2_remote_beacon_model(config, tokenizer=tokenizer)
    if backend == "qwen3_simple":
        from .qwen3_simple import load_trainable_qwen3_simple_beacon_model

        return load_trainable_qwen3_simple_beacon_model(config, tokenizer=tokenizer)
    if backend == "qwen3_official":
        from .qwen3_official import load_trainable_qwen3_official_beacon_model

        return load_trainable_qwen3_official_beacon_model(config, tokenizer=tokenizer)
    raise ValueError(f"Unsupported Activation Beacon backend: {backend!r}")


def save_training_metadata(output_dir: Path, config: dict, stats: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "activation_beacon_training.json").write_text(
        json.dumps({"config": config, "parameters": stats}, indent=2)
    )


# Backward-compatible re-exports for tests and external callers.
from .qwen2_official import build_beacon_config  # noqa: E402


def __getattr__(name: str):
    if name == "SimpleQwen3ActivationBeacon":
        from .qwen3_simple import SimpleQwen3ActivationBeacon

        return SimpleQwen3ActivationBeacon
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

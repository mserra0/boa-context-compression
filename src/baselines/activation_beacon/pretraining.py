# Adapted from Activation Beacon (https://github.com/FlagOpen/FlagEmbedding, research/Long_LLM/activation_beacon),
# Copyright (c) 2022 staoxiao, MIT License. Modified for the Boa baseline reproduction; see THIRD_PARTY_NOTICES.md.
"""Streaming FineWeb-Edu utilities for Activation Beacon pretraining."""

from __future__ import annotations

import glob
import random
from pathlib import Path

import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset, get_worker_info


def discover_parquet_files(data_dir: str, pattern: str = "**/*.parquet") -> list[str]:
    files = sorted(glob.glob(str(Path(data_dir) / pattern), recursive=True))
    if not files:
        raise FileNotFoundError(f"No Parquet files matched {pattern!r} under {data_dir}.")
    return files


class FineWebTokenDataset(IterableDataset):
    """Stream documents, append EOS, filter by length, and avoid document packing."""

    def __init__(
        self,
        files: list[str],
        tokenizer,
        *,
        text_column: str = "text",
        min_length: int,
        max_length: int,
        seed: int = 42,
        tokenizer_batch_size: int = 64,
        rank: int = 0,
        world_size: int = 1,
    ):
        super().__init__()
        self.files = list(files)
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.min_length = int(min_length)
        self.max_length = int(max_length)
        self.seed = int(seed)
        self.tokenizer_batch_size = int(tokenizer_batch_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self._state = self.initial_state()

    @staticmethod
    def initial_state() -> dict[str, int]:
        return {
            "version": 1,
            "epoch": 0,
            "row_group_index": 0,
            "text_start": 0,
            "encoded_index": 0,
        }

    def state_dict(self) -> dict[str, int]:
        return dict(self._state)

    def load_state_dict(self, state: dict | None) -> None:
        restored = self.initial_state()
        if state:
            restored.update(state)
        if int(restored.get("version", 1)) != 1:
            raise ValueError(f"Unsupported FineWebTokenDataset state: {state!r}")
        self._state = {
            "version": 1,
            "epoch": int(restored["epoch"]),
            "row_group_index": int(restored["row_group_index"]),
            "text_start": int(restored["text_start"]),
            "encoded_index": int(restored["encoded_index"]),
        }

    def _row_groups(self, epoch: int = 0) -> list[tuple[str, int]]:
        groups = []
        for path in self.files:
            parquet = pq.ParquetFile(path)
            groups.extend((path, index) for index in range(parquet.num_row_groups))
        # Deterministic shuffle is identical on every rank (shared seed) so the
        # rank slice below partitions the same ordering without overlap. The epoch
        # offset reshuffles on each infinite-iterator pass while keeping every rank
        # in agreement, so the partition stays disjoint across ranks per epoch.
        random.Random(self.seed + epoch).shuffle(groups)
        # Shard across DDP ranks first (dispatch_batches=False: each rank streams
        # its own disjoint shard), then across DataLoader workers within a rank.
        if self.world_size > 1:
            groups = groups[self.rank :: self.world_size]
        worker = get_worker_info()
        if worker is not None:
            groups = groups[worker.id :: worker.num_workers]
        return groups

    def __iter__(self):
        eos = self.tokenizer.eos_token_id
        if eos is None:
            raise ValueError("FineWeb pretraining requires tokenizer.eos_token_id.")

        # Infinite stream: the training loop stops every rank together via the
        # all-reduced global token budget. Never raise StopIteration — a single
        # rank draining its shard first would exit while peers block on the next
        # collective, deadlocking DDP until the NCCL watchdog kills the job.
        state = self.state_dict()
        epoch = int(state["epoch"])
        while True:
            groups = self._row_groups(epoch)
            row_group_index = int(state["row_group_index"])
            while row_group_index < len(groups):
                path, row_group = groups[row_group_index]
                table = pq.ParquetFile(path).read_row_group(
                    row_group, columns=[self.text_column]
                )
                texts = [text for text in table[self.text_column].to_pylist() if text]
                text_start = int(state["text_start"])
                while text_start < len(texts):
                    encoded = self.tokenizer(
                        texts[text_start : text_start + self.tokenizer_batch_size],
                        add_special_tokens=False,
                        truncation=True,
                        max_length=self.max_length - 1,
                    )["input_ids"]
                    encoded_index = int(state["encoded_index"])
                    while encoded_index < len(encoded):
                        input_ids = encoded[encoded_index]
                        state = self._next_state(
                            epoch,
                            row_group_index,
                            text_start,
                            encoded_index,
                            encoded_count=len(encoded),
                            text_count=len(texts),
                            group_count=len(groups),
                        )
                        self._state = state
                        encoded_index += 1
                        input_ids = list(input_ids) + [eos]
                        if len(input_ids) < self.min_length:
                            continue
                        yield {"input_ids": input_ids}
                    text_start += self.tokenizer_batch_size
                    state = {
                        "version": 1,
                        "epoch": epoch,
                        "row_group_index": row_group_index,
                        "text_start": text_start,
                        "encoded_index": 0,
                    }
                    self._state = state
                row_group_index += 1
                state = {
                    "version": 1,
                    "epoch": epoch,
                    "row_group_index": row_group_index,
                    "text_start": 0,
                    "encoded_index": 0,
                }
                self._state = state
            epoch += 1
            state = {
                "version": 1,
                "epoch": epoch,
                "row_group_index": 0,
                "text_start": 0,
                "encoded_index": 0,
            }
            self._state = state

    def _next_state(
        self,
        epoch: int,
        row_group_index: int,
        text_start: int,
        encoded_index: int,
        *,
        encoded_count: int,
        text_count: int,
        group_count: int,
    ) -> dict[str, int]:
        next_encoded = encoded_index + 1
        if next_encoded < encoded_count:
            return {
                "version": 1,
                "epoch": epoch,
                "row_group_index": row_group_index,
                "text_start": text_start,
                "encoded_index": next_encoded,
            }
        next_text_start = text_start + self.tokenizer_batch_size
        if next_text_start < text_count:
            return {
                "version": 1,
                "epoch": epoch,
                "row_group_index": row_group_index,
                "text_start": next_text_start,
                "encoded_index": 0,
            }
        next_row_group_index = row_group_index + 1
        if next_row_group_index < group_count:
            return {
                "version": 1,
                "epoch": epoch,
                "row_group_index": next_row_group_index,
                "text_start": 0,
                "encoded_index": 0,
            }
        return {
            "version": 1,
            "epoch": epoch + 1,
            "row_group_index": 0,
            "text_start": 0,
            "encoded_index": 0,
        }


def collate_causal_lm(batch: list[dict], pad_token_id: int) -> dict[str, torch.Tensor]:
    max_length = max(len(item["input_ids"]) for item in batch)
    input_ids, attention_mask, labels = [], [], []
    for item in batch:
        ids = item["input_ids"]
        padding = max_length - len(ids)
        input_ids.append(ids + [pad_token_id] * padding)
        attention_mask.append([1] * len(ids) + [0] * padding)
        # The official Beacon Memory rotates labels left by one position before
        # splitting windows, then computes loss with shift=False. Supply an
        # unshifted copy here and mask only padding.
        labels.append(ids + [-100] * padding)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def cap_batch_to_token_budget(
    batch: dict[str, torch.Tensor], remaining_tokens: int
) -> tuple[dict[str, torch.Tensor], int]:
    """Mask excess targets so the global supervised-token count stops exactly."""
    labels = batch["labels"]
    # Beacon Memory rotates labels as labels[:, 1:] + [-100]. Therefore only
    # positions 1..N are effective next-token targets; position zero is discarded.
    effective = labels[:, 1:].ne(-100)
    available = int(effective.sum().item())
    keep = min(int(remaining_tokens), available)
    if keep < available:
        active_indices = effective.flatten().nonzero(as_tuple=False).flatten()
        # Convert flattened indices from labels[:, 1:] back to full label indices.
        row_width = labels.shape[1]
        effective_width = row_width - 1
        excess = active_indices[keep:]
        rows = torch.div(excess, effective_width, rounding_mode="floor")
        columns = excess.remainder(effective_width) + 1
        labels[rows, columns] = -100
        batch["labels"] = labels
    return batch, keep


def token_lr_scale(tokens_processed: int, token_budget: int, warmup_tokens: int) -> float:
    if warmup_tokens > 0 and tokens_processed < warmup_tokens:
        return max(tokens_processed / warmup_tokens, 1e-8)
    decay_tokens = max(1, token_budget - warmup_tokens)
    return max(0.0, (token_budget - tokens_processed) / decay_tokens)


def validate_pretraining_config(config: dict) -> None:
    model = config["model"]
    data = config["data"]
    training = config["training"]
    if int(data["max_length"]) <= int(model["beacon_window"]):
        raise ValueError("data.max_length must be larger than model.beacon_window.")
    if int(data["min_length"]) <= int(model["beacon_window"]):
        raise ValueError("data.min_length must activate at least one Beacon window.")
    if int(data["min_length"]) > int(data["max_length"]):
        raise ValueError("data.min_length cannot exceed data.max_length.")
    if int(training["token_budget"]) <= 0:
        raise ValueError("training.token_budget must be positive.")
    if int(training.get("warmup_tokens", 0)) >= int(training["token_budget"]):
        raise ValueError("training.warmup_tokens must be smaller than token_budget.")
    if int(training.get("batch_size", 1)) < 1:
        raise ValueError("training.batch_size must be positive.")
    if int(data.get("tokenizer_batch_size", 64)) < 1:
        raise ValueError("data.tokenizer_batch_size must be positive.")
    if (
        training.get("save_restart_state", False)
        or training.get("resume_from_checkpoint")
    ) and int(data.get("num_workers", 0)) != 0:
        raise ValueError(
            "Exact pretraining restart requires data.num_workers=0. "
            "DataLoader worker prefetching hides the consumed dataset cursor."
        )


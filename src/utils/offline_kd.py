"""Offline knowledge-distillation helpers: read a precomputed top-K teacher dump and
turn its sparse rows into dense KL targets, so training never runs the teacher live."""
import os
import json
import glob
from collections import OrderedDict

import torch


class OfflineTeacherIndex:
    """Maps a dataset example_idx to its dumped teacher row across sharded .pt files."""

    # cache_size is the number of shards kept in memory. dump_logits writes example_idx
    # round-robin across ranks, so a shuffled sampler requests indices scattered across
    # shards and most get() calls miss the cache and reload a whole shard for one row.
    # Larger dumps may need a larger cache_size (up to the number of shards, RAM permitting).
    def __init__(self, dump_dir: str, split: str = "train", cache_size: int = 2):
        self.dump_dir = dump_dir
        with open(os.path.join(dump_dir, f"manifest_{split}.json")) as f:
            self.manifest = json.load(f)
        self.total_examples = int(self.manifest["total_examples"])
        self.top_k = int(self.manifest["top_k"])
        self.max_new_tokens = int(self.manifest["max_new_tokens"])
        shard_glob = self.manifest.get("shard_glob", f"{split}_shard_r*.pt")
        self.shard_paths = sorted(glob.glob(os.path.join(dump_dir, shard_glob)))
        # example_idx -> (shard_path, row). One pass; each shard released after reading idx.
        self._locator: dict[int, tuple[str, int]] = {}
        for sp in self.shard_paths:
            idx_vec = torch.load(sp, map_location="cpu")["example_idx"]
            for row, ex in enumerate(idx_vec.tolist()):
                self._locator[int(ex)] = (sp, row)
        self._cache: "OrderedDict[str, dict]" = OrderedDict()
        self._cache_size = cache_size

    def _load_shard(self, path: str) -> dict:
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]
        shard = torch.load(path, map_location="cpu")
        self._cache[path] = shard
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return shard

    def get(self, example_idx: int) -> dict:
        sp, row = self._locator[int(example_idx)]
        shard = self._load_shard(sp)
        return {
            "generated_ids": shard["generated_ids"][row],
            "topk_values": shard["topk_values"][row],
            "topk_indices": shard["topk_indices"][row],
            "length": int(shard["length"][row]),
        }

    def __contains__(self, example_idx: int) -> bool:
        return int(example_idx) in self._locator

    def __len__(self) -> int:
        return len(self._locator)


def reconstruct_teacher_logits(topk_values: torch.Tensor, topk_indices: torch.Tensor,
                               vocab_size: int, dtype: torch.dtype = torch.float32,
                               device=None) -> torch.Tensor:
    """Scatter sparse top-k teacher logits into a dense [B,G,V] tensor, -inf off-support.

    Mirrors the online teacher: softmax over this dense tensor equals softmax over just the
    top-k logits, renormalized (off-support mass is exactly zero). dtype float32 matches the
    online generate() scores that kl_loss already consumes.
    """
    B, G, _ = topk_values.shape
    if device is None:
        device = topk_values.device
    dense = torch.full((B, G, vocab_size), float("-inf"), dtype=dtype, device=device)
    dense.scatter_(-1, topk_indices.to(device), topk_values.to(dtype=dtype, device=device))
    return dense

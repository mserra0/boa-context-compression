import math
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from colorama import Fore, Style
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from torch.utils.data import DataLoader, Sampler


def _is_main_process() -> bool:
    """True on global rank 0 (torchrun sets RANK), else single-process default."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return os.environ.get("RANK", "0") == "0"


def progress_iter(iterable, description, total=None, enabled=None):
    """Wrap an iterable with a rich progress bar on rank 0 only.

    Off-rank (and when disabled) returns the iterable untouched so non-main
    processes stay silent and add no overhead.
    """
    if enabled is None:
        enabled = _is_main_process()
    if not enabled:
        return iterable
    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        transient=True,
    )

    def _gen():
        with progress:
            yield from progress.track(iterable, total=total, description=description)

    return _gen()

# Set when collate_fn_mamba2 has already warned this process about an
# over-budget pack, so the warning prints at most once per worker.
_BUDGET_WARNED = False


def _build_llm_batch(
    llm_inputs_list: list,
    llm_tokenizer,
    max_seq_len: int,
) -> dict:
    """Left-pad LLM input sequences to the longest in the batch (up to max_seq_len).

    Returns:
        dict with 'input_ids' [B, L] and 'attention_mask' [B, L].
    """
    clipped = [ids[:max_seq_len] for ids in llm_inputs_list]
    batch_max = max(len(ids) for ids in clipped) if clipped else 0

    input_ids, attention_masks = [], []
    for ids in clipped:
        pad_len = batch_max - len(ids)
        input_ids.append([llm_tokenizer.pad_token_id] * pad_len + ids)
        attention_masks.append([0] * pad_len + [1] * len(ids))

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
    }


def collate_fn(batch, llm_tokenizer, compressor_tokenizer, max_length=256):
    """Left-padded collate for Mamba. Both LLM and compressor inputs are left-padded."""
    llm_inputs_list = [item["llm_inputs"] for item in batch]
    compressor_inputs_list = [item["compressor_inputs"] for item in batch]

    llm_batch = _build_llm_batch(llm_inputs_list, llm_tokenizer, max_length)

    comp_ids_list = [item['input_ids'][:max_length] for item in compressor_inputs_list]
    batch_comp_max = max(len(ids) for ids in comp_ids_list) if comp_ids_list else 0

    comp_input_ids, comp_attention_masks = [], []
    for ids in comp_ids_list:
        pad_len = batch_comp_max - len(ids)
        comp_input_ids.append([compressor_tokenizer.pad_token_id] * pad_len + ids)
        comp_attention_masks.append([0] * pad_len + [1] * len(ids))

    out = {
        "llm_inputs": llm_batch,
        "compressor_inputs": {
            "input_ids": torch.tensor(comp_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(comp_attention_masks, dtype=torch.long),
        },
    }

    if batch and "question_comp_len" in batch[0]:
        # B-ctx: attach the absolute (padded-tensor) index where the context
        # begins per row, plus the literal (un-tensorized) question token ids.
        comp_real_len = torch.tensor([len(ids) for ids in comp_ids_list])
        first_real = batch_comp_max - comp_real_len
        qlen = torch.tensor([item["question_comp_len"] for item in batch])
        context_start = (first_real + torch.minimum(qlen, comp_real_len)).long()
        out["compressor_inputs"]["context_start"] = context_start
        out["compressor_inputs"]["question_ids"] = [item["question_ids"] for item in batch]

    return out


def collate_fn_mamba2(batch, llm_tokenizer, compressor_tokenizer, max_length=256, tokens_per_batch=None):
    """Packed (padding-free) collate for Mamba2.

    The compressor receives a single flat tensor of concatenated token IDs [1, T]
    plus a seq_idx tensor [1, T] that tells Mamba2 where each sequence starts/ends
    so the SSM state is reset at boundaries. No padding tokens reach the compressor.

    Returns:
        llm_inputs:
            input_ids      [B, L]   left-padded
            attention_mask [B, L]
        compressor_inputs:
            input_ids      [1, T]   packed
            seq_idx        [1, T]   int32
            lengths        [B]      int64
    """
    llm_inputs_list = [item["llm_inputs"] for item in batch]
    compressor_inputs_list = [item["compressor_inputs"] for item in batch]

    llm_batch = _build_llm_batch(llm_inputs_list, llm_tokenizer, max_length)

    packed_ids, seq_idx_list, lengths = [], [], []
    for i, item in enumerate(compressor_inputs_list):
        ids = item['input_ids'][:max_length]
        packed_ids.extend(ids)
        seq_idx_list.extend([i] * len(ids))
        lengths.append(len(ids))

    if tokens_per_batch is not None:
        T = len(packed_ids)
        if T > tokens_per_batch:
            global _BUDGET_WARNED
            if not _BUDGET_WARNED:
                print(
                    f"{Fore.YELLOW}Warning: packed length {T} exceeds tokens_per_batch "
                    f"{tokens_per_batch}; length source disagrees with collate truncation"
                    f"{Style.RESET_ALL}"
                )
                _BUDGET_WARNED = True

    out = {
        "llm_inputs": llm_batch,
        "compressor_inputs": {
            "input_ids": torch.tensor(packed_ids, dtype=torch.long).unsqueeze(0),
            "seq_idx": torch.tensor(seq_idx_list, dtype=torch.int32).unsqueeze(0),
            "lengths": torch.tensor(lengths, dtype=torch.long),
        },
    }

    if batch and "question_comp_len" in batch[0]:
        # B-ctx: per-segment context offset (clamped to that segment's packed
        # length), plus the literal (un-tensorized) question token ids.
        context_starts = [
            min(item["question_comp_len"], L) for item, L in zip(batch, lengths)
        ]
        out["compressor_inputs"]["context_start"] = torch.tensor(context_starts, dtype=torch.long)
        out["compressor_inputs"]["question_ids"] = [item["question_ids"] for item in batch]

    return out

def _pretrain_xy_pairs(batch, llm_tokenizer, compressor_tokenizer,
                       max_ctx, max_pred, split_lo, split_hi, rng, teacher=False):
    """Per doc: sample a random CHAR boundary, tokenize prefix X (compressor tok) and
    suffix Y (LLM tok). Char split keeps the two tokenizers independent (no
    decode/re-encode roundtrip). Degenerate docs (empty Y) dropped. Shared by the
    Mamba1/Mamba2 pretrain collates so the split can never drift between them.

    Returns (x_ids_list, y_ids_list, tx_ids_list). tx_ids_list is the LLM-tokenized
    prefix X (same char boundary c), needed only by the KD collates so the teacher can
    condition on the PLAINTEXT prefix; it is [] when teacher=False (no wasted tokenizing
    on the CE path).
    """
    x_ids_list, y_ids_list, tx_ids_list = [], [], []
    for item in batch:
        text = item["text"]
        n = len(text)
        lo = max(1, int(n * split_lo))
        hi = max(lo + 1, int(n * split_hi))
        c = rng.randint(lo, min(hi, n - 1)) if n > 1 else 0

        # Keep the LAST max_ctx tokens of the prefix so X ends exactly where Y begins:
        # X and Y stay adjacent (true continuation). Front-capping ([:max_ctx]) would
        # drop the middle [max_ctx, c) for long prefixes -> compressor sees the head while
        # Y is a later passage with no lead-in.
        prefix_text = text[:c]
        x_ids = compressor_tokenizer(prefix_text, add_special_tokens=False)["input_ids"][-max_ctx:]
        y_ids = llm_tokenizer(text[c:], add_special_tokens=False)["input_ids"][:max_pred]
        if not x_ids:
            prefix_text = text
            x_ids = compressor_tokenizer(prefix_text, add_special_tokens=False)["input_ids"][-max_ctx:]
        if not y_ids:
            continue
        x_ids_list.append(x_ids)
        y_ids_list.append(y_ids)
        if teacher:
            # Teacher prefix mirrors the compressor prefix (same char region, same last-
            # max_ctx tail cap) but in the LLM vocab, so the teacher sees the plaintext X
            # the student had to compress.
            tx_ids = llm_tokenizer(prefix_text, add_special_tokens=False)["input_ids"][-max_ctx:]
            tx_ids_list.append(tx_ids)
    return x_ids_list, y_ids_list, tx_ids_list


def _build_pretrain_targets(y_ids_list, pad_id):
    """Right-pad the suffixes Y into target_ids [B, G] + target_mask [B, G]."""
    B = len(y_ids_list)
    G = max(len(y) for y in y_ids_list)
    target_ids = torch.full((B, G), pad_id, dtype=torch.long)
    target_mask = torch.zeros((B, G), dtype=torch.long)
    for b, y in enumerate(y_ids_list):
        target_ids[b, :len(y)] = torch.tensor(y, dtype=torch.long)
        target_mask[b, :len(y)] = 1
    return target_ids, target_mask


def collate_fn_pretrain(batch, llm_tokenizer, compressor_tokenizer, max_length=256, *,
                        max_ctx: int = 1024, max_pred: int = 256,
                        split_lo: float = 0.2, split_hi: float = 0.8, rng=None):
    """Continuation-LM collate (Mamba2): compress the prefix X (packed [1, T], no pad
    enters the SSM), predict the suffix Y (right-padded targets). max_ctx / max_pred cap
    the X / Y token counts; PretrainDataset enforces the lower bound.
    """
    rng = rng or random
    x_ids_list, y_ids_list, _ = _pretrain_xy_pairs(
        batch, llm_tokenizer, compressor_tokenizer, max_ctx, max_pred, split_lo, split_hi, rng)
    if not y_ids_list:
        raise ValueError("collate_fn_pretrain: entire batch degenerate (no valid X/Y pairs)")

    packed_ids, seq_idx_list, lengths = [], [], []
    for x_ids in x_ids_list:
        packed_ids.extend(x_ids)
        seq_idx_list.extend([len(lengths)] * len(x_ids))
        lengths.append(len(x_ids))

    target_ids, target_mask = _build_pretrain_targets(y_ids_list, llm_tokenizer.pad_token_id)
    return {
        "compressor_inputs": {
            "input_ids": torch.tensor(packed_ids, dtype=torch.long).unsqueeze(0),
            "seq_idx": torch.tensor(seq_idx_list, dtype=torch.int32).unsqueeze(0),
            "lengths": torch.tensor(lengths, dtype=torch.long),
        },
        "target_inputs": {
            "target_ids": target_ids,
            "target_mask": target_mask,
        },
    }


def collate_fn_pretrain_mamba1(batch, llm_tokenizer, compressor_tokenizer, max_length=256, *,
                               max_ctx: int = 1024, max_pred: int = 256,
                               split_lo: float = 0.2, split_hi: float = 0.8, rng=None):
    """Continuation-LM collate (Mamba1): same random X/Y char split, but the prefix X is
    LEFT-padded into [B, I] + attention_mask (MambaCompressor's stride anchoring assumes
    left-padding) instead of Mamba2's packed [1, T]. Y targets are right-padded, identical
    to the Mamba2 path.
    """
    rng = rng or random
    x_ids_list, y_ids_list, _ = _pretrain_xy_pairs(
        batch, llm_tokenizer, compressor_tokenizer, max_ctx, max_pred, split_lo, split_hi, rng)
    if not y_ids_list:
        raise ValueError("collate_fn_pretrain_mamba1: entire batch degenerate (no valid X/Y pairs)")

    B = len(x_ids_list)
    I = max(len(x) for x in x_ids_list)
    comp_pad = compressor_tokenizer.pad_token_id
    comp_input_ids = torch.full((B, I), comp_pad, dtype=torch.long)
    comp_attention = torch.zeros((B, I), dtype=torch.long)
    for b, x in enumerate(x_ids_list):
        comp_input_ids[b, I - len(x):] = torch.tensor(x, dtype=torch.long)  # left-pad
        comp_attention[b, I - len(x):] = 1

    target_ids, target_mask = _build_pretrain_targets(y_ids_list, llm_tokenizer.pad_token_id)
    return {
        "compressor_inputs": {
            "input_ids": comp_input_ids,
            "attention_mask": comp_attention,
        },
        "target_inputs": {
            "target_ids": target_ids,
            "target_mask": target_mask,
        },
    }


def collate_fn_pretrain_kd(batch, llm_tokenizer, compressor_tokenizer, max_length=256, *,
                           max_ctx: int = 1024, max_pred: int = 256,
                           split_lo: float = 0.2, split_hi: float = 0.8, rng=None):
    """KD continuation collate (Mamba2): identical compressor/target contract to
    collate_fn_pretrain, plus teacher_inputs = the LLM-tokenized plaintext prefix X,
    LEFT-padded [B, Lx], so KDPretrainer's teacher can teacher-force the same Y.
    """
    rng = rng or random
    x_ids_list, y_ids_list, tx_ids_list = _pretrain_xy_pairs(
        batch, llm_tokenizer, compressor_tokenizer, max_ctx, max_pred,
        split_lo, split_hi, rng, teacher=True)
    if not y_ids_list:
        raise ValueError("collate_fn_pretrain_kd: entire batch degenerate (no valid X/Y pairs)")

    packed_ids, seq_idx_list, lengths = [], [], []
    for x_ids in x_ids_list:
        packed_ids.extend(x_ids)
        seq_idx_list.extend([len(lengths)] * len(x_ids))
        lengths.append(len(x_ids))

    target_ids, target_mask = _build_pretrain_targets(y_ids_list, llm_tokenizer.pad_token_id)
    teacher_inputs = _build_llm_batch(tx_ids_list, llm_tokenizer, max_ctx)
    return {
        "compressor_inputs": {
            "input_ids": torch.tensor(packed_ids, dtype=torch.long).unsqueeze(0),
            "seq_idx": torch.tensor(seq_idx_list, dtype=torch.int32).unsqueeze(0),
            "lengths": torch.tensor(lengths, dtype=torch.long),
        },
        "target_inputs": {
            "target_ids": target_ids,
            "target_mask": target_mask,
        },
        "teacher_inputs": teacher_inputs,
    }


def collate_fn_pretrain_kd_mamba1(batch, llm_tokenizer, compressor_tokenizer, max_length=256, *,
                                  max_ctx: int = 1024, max_pred: int = 256,
                                  split_lo: float = 0.2, split_hi: float = 0.8, rng=None):
    """KD continuation collate (Mamba1): left-padded compressor prefix [B, I] as in
    collate_fn_pretrain_mamba1, plus the LEFT-padded LLM-tokenized teacher prefix.
    """
    rng = rng or random
    x_ids_list, y_ids_list, tx_ids_list = _pretrain_xy_pairs(
        batch, llm_tokenizer, compressor_tokenizer, max_ctx, max_pred,
        split_lo, split_hi, rng, teacher=True)
    if not y_ids_list:
        raise ValueError("collate_fn_pretrain_kd_mamba1: entire batch degenerate (no valid X/Y pairs)")

    B = len(x_ids_list)
    I = max(len(x) for x in x_ids_list)
    comp_pad = compressor_tokenizer.pad_token_id
    comp_input_ids = torch.full((B, I), comp_pad, dtype=torch.long)
    comp_attention = torch.zeros((B, I), dtype=torch.long)
    for b, x in enumerate(x_ids_list):
        comp_input_ids[b, I - len(x):] = torch.tensor(x, dtype=torch.long)  # left-pad
        comp_attention[b, I - len(x):] = 1

    target_ids, target_mask = _build_pretrain_targets(y_ids_list, llm_tokenizer.pad_token_id)
    teacher_inputs = _build_llm_batch(tx_ids_list, llm_tokenizer, max_ctx)
    return {
        "compressor_inputs": {
            "input_ids": comp_input_ids,
            "attention_mask": comp_attention,
        },
        "target_inputs": {
            "target_ids": target_ids,
            "target_mask": target_mask,
        },
        "teacher_inputs": teacher_inputs,
    }


def compute_token_lengths(dataset, tokenizer=None, length_cap=None,
                          text_keys=("question", "context")):
    """Per-sample token length for the token-budget sampler.

    Uses an exact 'token_count' column if present (e.g. FineWeb), else tokenizes
    the joined text fields once. Values are clamped to length_cap so a single
    oversized doc still fits in its own pack.
    """
    cols = getattr(dataset, "column_names", []) or []
    if "token_count" in cols:
        lengths = list(dataset["token_count"])
    elif tokenizer is not None:
        lengths = []
        for i in range(len(dataset)):
            row = dataset[i]
            text = " ".join(str(row[k]) for k in text_keys if k in row and row[k])
            lengths.append(len(tokenizer(text, add_special_tokens=False)["input_ids"]))
    else:
        raise ValueError("compute_token_lengths needs a 'token_count' column or a tokenizer")

    if length_cap is not None:
        lengths = [min(l, length_cap) for l in lengths]
    return lengths


class DistributedLengthGroupedSampler(Sampler):
    def __init__(self, dataset, batch_size, num_replicas=None, rank=None, seed=0, shuffle=True):
        if num_replicas is None:
            if not dist.is_available() or not dist.is_initialized():
                num_replicas = 1
            else:
                num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available() or not dist.is_initialized():
                rank = 0
            else:
                rank = dist.get_rank()

        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.seed = seed
        self.shuffle = shuffle

        if hasattr(dataset, "token_lengths"):
            # PretrainDataset (and any dataset that precomputes per-sample lengths
            # aligned to its own indices) supplies token_lengths directly. The KD
            # CompressionDataset has no such attribute -> falls to the text-length path.
            self.lengths = list(dataset.token_lengths)
        else:
            self.lengths = []
            for i in range(len(dataset)):
                item = dataset.dataset[i]
                self.lengths.append(len(item.get("context", "")) + len(item.get("question", "")))

        self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        indices = np.argsort(self.lengths).tolist()

        if self.shuffle:
            # Group into megabatches
            megabatch_size = self.batch_size * self.num_replicas * 10
            megabatches = [indices[i:i + megabatch_size] for i in range(0, len(indices), megabatch_size)]

            # Shuffle megabatches
            megabatch_indices = torch.randperm(len(megabatches), generator=g).tolist()
            indices = []
            for i in megabatch_indices:
                mb = megabatches[i]
                if len(mb) > 0:
                    mb_perm = torch.randperm(len(mb), generator=g).tolist()
                    indices.extend([mb[j] for j in mb_perm])

        padding_size = self.total_size - len(indices)
        if padding_size > 0:
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]

        assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class DistributedTokenBudgetSampler(Sampler):
    """Batch sampler that groups indices into packs summing to <= tokens_per_batch.

    Yields lists of indices (use as DataLoader(batch_sampler=...)). Designed for
    Mamba2 packing: each pack becomes one [1, T] packed sequence. DDP-safe — every
    rank yields the same number of disjoint packs.
    """

    def __init__(self, lengths, tokens_per_batch, num_replicas=None, rank=None,
                 seed=0, shuffle=True, length_cap=None, max_seqs_per_batch=None):
        if num_replicas is None:
            num_replicas = dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1
        if rank is None:
            rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0

        self.lengths = list(lengths)
        self.tokens_per_batch = tokens_per_batch
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.shuffle = shuffle
        self.length_cap = length_cap
        self.max_seqs_per_batch = max_seqs_per_batch
        self.epoch = 0
        self._batches = self._build()

    def _proxy(self, i):
        l = self.lengths[i]
        return min(l, self.length_cap) if self.length_cap is not None else l

    def _build(self):
        t0 = time.perf_counter()
        n = len(self.lengths)
        idx = list(range(n))
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            perm = torch.randperm(n, generator=g).tolist()
            idx = [idx[i] for i in perm]

        batches, cur, cur_tok = [], [], 0
        for i in progress_iter(idx, "Packing token-budget batches", total=n):
            w = self._proxy(i)
            over_tok = bool(cur) and cur_tok + w > self.tokens_per_batch
            over_cnt = self.max_seqs_per_batch is not None and len(cur) >= self.max_seqs_per_batch
            if cur and (over_tok or over_cnt):
                batches.append(cur)
                cur, cur_tok = [], 0
            cur.append(i)
            cur_tok += w
        if cur:
            batches.append(cur)

        if _is_main_process():
            print(f"{Fore.GREEN}[TokenBudgetSampler] packed {n:,} docs -> "
                  f"{len(batches):,} batches in {time.perf_counter() - t0:.1f}s{Style.RESET_ALL}",
                  flush=True)

        # DDP: equal, disjoint pack count per rank. Drops <= num_replicas-1 packs/epoch.
        n = len(batches) // self.num_replicas
        return batches[self.rank::self.num_replicas][:n] if self.num_replicas > 1 else batches

    def __iter__(self):
        return iter(self._batches)

    def __len__(self):
        return len(self._batches)

    def set_epoch(self, epoch):
        self.epoch = epoch
        self._batches = self._build()


def build_token_budget_loader(dataset, llm_tokenizer, compressor_tokenizer, *,
                              tokens_per_batch, max_seq_len, max_seqs_per_batch=None,
                              seed=0, shuffle=True, num_workers=0, lengths=None,
                              pin_memory=False, collate_fn=None):
    """DataLoader using DistributedTokenBudgetSampler. Defaults to the packed mamba2
    KD collate; pass collate_fn to override (e.g. the pretrain collate)."""
    if lengths is None:
        lengths = compute_token_lengths(
            dataset.dataset, compressor_tokenizer, length_cap=max_seq_len,
        )
    sampler = DistributedTokenBudgetSampler(
        lengths, tokens_per_batch=tokens_per_batch, seed=seed, shuffle=shuffle,
        length_cap=max_seq_len, max_seqs_per_batch=max_seqs_per_batch,
    )
    if collate_fn is None:
        collate_fn = lambda batch: collate_fn_mamba2(
            batch, llm_tokenizer, compressor_tokenizer,
            max_length=max_seq_len, tokens_per_batch=tokens_per_batch,
        )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

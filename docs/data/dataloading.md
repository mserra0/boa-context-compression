# Data Loading and Collation Pipeline

## Overview

The dataloading pipeline supports two main distillation collation strategies (Mamba1 left-padded batching vs. Mamba2 packed sequence batching) as well as continuation-LM pretraining collation.

| Mode | Backend Collate | Compressor Inputs Geometry | LLM Inputs Geometry |
|---|---|---|---|
| Mamba1 KD | `collate_fn` | Left-padded `[B, I]` + `attention_mask` `[B, I]` | Left-padded `[B, L]` + `attention_mask` `[B, L]` |
| Mamba2 KD | `collate_fn_mamba2` | Sequence-packed `[1, T]`, `seq_idx` `[1, T]`, `lengths` `[B]` | Left-padded `[B, L]` + `attention_mask` `[B, L]` |
| Mamba1 Pretrain | `collate_fn_pretrain_mamba1` | Left-padded `[B, I]` + `attention_mask` `[B, I]` | Right-padded `target_ids` `[B, G]` + `target_mask` `[B, G]` |
| Mamba2 Pretrain | `collate_fn_pretrain` | Sequence-packed `[1, T]`, `seq_idx` `[1, T]`, `lengths` `[B]` | Right-padded `target_ids` `[B, G]` + `target_mask` `[B, G]` |

In all KD modes, `_build_llm_batch` constructs left-padded LLM inputs because the frozen teacher model processes batched tensor sequences.

---

## Mamba1 Collation (`collate_fn`)

Left-pads both LLM inputs and compressor inputs up to the maximum sequence length within the batch (`max_length` cap).

```python
# src/utils/dataloading.py
out = {
    "llm_inputs": {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),         # [B, L]
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long), # [B, L]
    },
    "compressor_inputs": {
        "input_ids": torch.tensor(comp_input_ids, dtype=torch.long),      # [B, I]
        "attention_mask": torch.tensor(comp_attention_masks, dtype=torch.long), # [B, I]
    },
}
```

### B-ctx Context Offset (Mamba1)

When training with `student_template == "wrapped"` and `compress_question = False` (B-ctx mode), the literal question is NOT compressed. The collate calculates the exact padded tensor index where context starts:

```python
comp_real_len = torch.tensor([len(ids) for ids in comp_ids_list])
first_real = batch_comp_max - comp_real_len  # left-padding shift
qlen = torch.tensor([item["question_comp_len"] for item in batch])
context_start = (first_real + torch.minimum(qlen, comp_real_len)).long()

out["compressor_inputs"]["context_start"] = context_start  # [B]
out["compressor_inputs"]["question_ids"] = [item["question_ids"] for item in batch]
```

---

## Mamba2 Packed Collation (`collate_fn_mamba2`)

Mamba2 uses sequence packing without padding tokens. Compressor token IDs are flattened into a single `[1, T]` tensor ($T = \sum_{b=0}^{B-1} L_b$).

```python
# src/utils/dataloading.py
packed_ids, seq_idx_list, lengths = [], [], []
for i, item in enumerate(compressor_inputs_list):
    ids = item['input_ids'][:max_length]
    packed_ids.extend(ids)
    seq_idx_list.extend([i] * len(ids))  # sequence index matching batch position i
    lengths.append(len(ids))

out = {
    "llm_inputs": llm_batch,  # [B, L] left-padded
    "compressor_inputs": {
        "input_ids": torch.tensor(packed_ids, dtype=torch.long).unsqueeze(0),    # [1, T]
        "seq_idx": torch.tensor(seq_idx_list, dtype=torch.int32).unsqueeze(0),   # [1, T]
        "lengths": torch.tensor(lengths, dtype=torch.long),                      # [B]
    },
}
```

### B-ctx Context Offset (Mamba2)

For packed sequences, `context_start` gives the local context offset within each packed segment:

```python
context_starts = [min(item["question_comp_len"], L) for item, L in zip(batch, lengths)]
out["compressor_inputs"]["context_start"] = torch.tensor(context_starts, dtype=torch.long)
out["compressor_inputs"]["question_ids"] = [item["question_ids"] for item in batch]
```

---

## Pretraining Collation (`_pretrain_xy_pairs`, `collate_fn_pretrain`, `collate_fn_pretrain_mamba1`)

For LM continuation pretraining (e.g. FineWeb-EDU), documents are dynamically split into prefix $X$ (compressed by Mamba) and continuation suffix $Y$ (predicted by LLM).

### Random Character Boundary Split (`_pretrain_xy_pairs`)

To avoid tokenizer roundtrip artifacts, a random character split point $c$ is sampled between `split_lo` (default 0.2) and `split_hi` (default 0.8) of the document length:

```python
# src/utils/dataloading.py
c = rng.randint(lo, min(hi, n - 1))
x_ids = compressor_tokenizer(text[:c], add_special_tokens=False)["input_ids"][-max_ctx:]
y_ids = llm_tokenizer(text[c:], add_special_tokens=False)["input_ids"][:max_pred]
```

- **Adjacent Continuation:** Taking the **last** `max_ctx` tokens of `text[:c]` ensures $X$ ends exactly where $Y$ begins.
- **Suffix Targets (`_build_pretrain_targets`):** Suffixes $Y$ are right-padded into `target_ids [B, G]` and `target_mask [B, G]`.

### Collate Implementations

- **`collate_fn_pretrain` (Mamba2):** Packs prefixes into `[1, T]` with `seq_idx` `[1, T]` and `lengths [B]`. Returns `compressor_inputs` and `target_inputs`.
- **`collate_fn_pretrain_mamba1` (Mamba1):** Left-pads prefixes into `comp_input_ids [B, I]` and `attention_mask [B, I]`. Returns `compressor_inputs` and `target_inputs`.

---

## Samplers and Token-Budget Packing

### 1. `DistributedTokenBudgetSampler`

`DistributedTokenBudgetSampler` groups document indices into packs such that total pack tokens sum to $\le \text{tokens\_per\_batch}$ (and optional count $\le \text{max\_seqs\_per\_batch}$).

- Yields lists of index batches (used as `batch_sampler` in `DataLoader`).
- Each pack becomes a single packed `[1, T]` sequence in Mamba2.
- **DDP Safety:** Truncates to an equal number of disjoint packs across ranks (`len(batches) // num_replicas`), preventing rank desynchronization.

### 2. `build_token_budget_loader`

Factory function that constructs a `DataLoader` using `DistributedTokenBudgetSampler` and `collate_fn_mamba2` (or a custom `collate_fn`):

```python
# src/utils/dataloading.py
loader = build_token_budget_loader(
    dataset, llm_tokenizer, compressor_tokenizer,
    tokens_per_batch=8192, max_seq_len=512,
    num_workers=4, pin_memory=True,
)
```

### 3. `compute_token_lengths`

Precomputes per-sample token lengths used by budget samplers. Uses pre-existing `token_count` column if present (FineWeb), or tokenizes text fields once, clamping values to `length_cap`.

### 4. `DistributedLengthGroupedSampler`

Used for padded batching (Mamba1). Groups samples of similar lengths into megabatches (`batch_size * num_replicas * 10`), shuffles megabatches and intra-megabatch items, and distributes indices across DDP ranks to minimize zero-padding overhead.

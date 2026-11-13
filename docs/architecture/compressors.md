# Compressors: Mamba1 vs Mamba2

Two compressor backends share the same `BaseCompressor` interface but differ in how they handle batching and padding.

---

## Shared Concepts

### Compression Ratio and Stride

```python
stride = int(1 / compression_ratio)
```

A `compression_ratio` of 0.1 → `stride = 10`. For an input of length L, the compressor produces `C = ceil(L / stride)` soft prompts.

### Stride Anchoring

Both backends anchor the stride grid to the **last real token** of each sequence. This guarantees that position `C-1` (the last slot) always corresponds to the final valid context position — the hidden state that predicts the first generated token.

Without anchoring, using a fixed start position `hidden[0::stride]` would place the anchor at a padding position for samples shorter than the batch maximum, corrupting the soft prompts.

### Left-Padded Output

Both backends produce `[B, C_max, H_llm]` with **left-padding** (zeros at the front for shorter sequences). This means `soft_prompts[:, -1, :]` is always the last valid soft prompt for every sample in the batch — aligning with the teacher's left-padded `llm_inputs`.

---

## `MambaCompressor` — Mamba v1

**Input format:** left-padded batches `[B, I]`

**Backbone:** `MambaForCausalLM` from HuggingFace `transformers`

**Stride logic** (`src/compressors.py`, `MambaCompressor.forward`):

```python
# For each sample i with real length seq_len[i]:
C_i = max(1, ceil(seq_len[i] / stride))
offsets = (C_i - 1 - slot_idx) * stride         # distance from last token
gather_idx = (seq_len[i] - 1) - offsets          # absolute positions
```

The gather itself is performed by `strided_gather` (`src/utils/compression.py`), or by `mamba1_context_gather` when `context_start` is passed (B-ctx layout). Slots that map to positions before the first real token (or before `context_start`) are marked `0` in `compressed_mask` and their soft prompts are zeroed.

**Attention geometry:** fully causal over `[soft_prompts | gen_tokens]`

**Padding strategy:** padded inputs — the SSM processes padding tokens as if they were real (mitigated by correct stride anchoring and masking).

---

## `Mamba2Compressor` — Mamba v2 with Sequence Packing

**Input format:** packed flat tensor `[1, T]` where `T = sum(lengths)`

**Backbone:** `mamba_ssm.MambaLMHeadModel`

**Attention geometry:** fully causal over [prefix | gen] (see [Attention Geometry](attention.md))

### Sequence Packing

Instead of padding sequences to the batch maximum, `collate_fn_mamba2` concatenates all sequences:

```
sample_0 [L_0 tokens] | sample_1 [L_1 tokens] | ... | sample_B [L_B tokens]
seq_idx:  0 0 0 ... 0 | 1 1 1 ... 1           | ... | B B B ... B
```

The `seq_idx [1, T]` tensor tells Mamba2 where each sequence starts, allowing the SSM state to be **reset at sequence boundaries**. This eliminates cross-sequence contamination and produces results numerically equivalent to processing each sample independently (relative diff < 5×10⁻³).

**Benefits:**
- No padding tokens enter the SSM at any point
- GPU utilization improves for variable-length batches (no wasted compute on padding)
- Memory footprint is strictly `sum(lengths)` tokens, not `batch_size × max_length`

### Stride Logic

```python
# For each sequence of length L:
start_idx = (L - 1) % stride          # first selected position (or mamba2_context_start for B-ctx)
selected = hidden[start_idx::stride]   # C = ceil(L / stride) states
```

This is the end-aligned striding that ensures the last selected position is always `hidden[L-1]` (the last real token).

### Projector Bias Zeroing

After projecting with `nn.Linear(bias=True)`, padded positions would have non-zero output (the bias alone). The projector output is therefore masked:

```python
soft_prompts = soft_prompts * compressed_mask.unsqueeze(-1)
```

Padded positions are zeroed after projection so they do not inject noise into the LLM.

---

## `gather_hidden` (KV Path)

Both compressors implement `gather_hidden(compressor_inputs)` (`src/compressors.py` and `221-225`).

This returns the raw pre-projector strided Mamba hidden states `[B, C, H_mamba]` and `compressed_mask [B, C]`. The KV path (`prompt_mode="kv"`) consumes this tensor directly to feed into `KVProjector` instead of calling `forward()`.

---

## Comparison

| Property | MambaCompressor | Mamba2Compressor |
|---|---|---|
| SSM version | Mamba v1 | Mamba v2 |
| Input format | Padded `[B, I]` | Packed `[1, T]` |
| Padding in SSM | Yes (mitigated by masking) | No |
| Attention geometry | Fully causal | Fully causal |
| Batch efficiency | Lower (padding waste) | Higher (packed) |
| Position IDs | Standard cumsum | Cumsum respecting left-pad |
| Inference method | Causal generation | Prefix KV cache + autoregressive |

---

## `compress_single`

Both compressors expose a `compress_single(input_ids)` method for offline / inference use:

```python
soft_prompts, compressed_mask = model.compressor.compress_single(input_ids)
# soft_prompts: [1, C, H_llm]
# compressed_mask: [1, C]
```

This wraps `forward` with a single-sequence batch (no real DDP context needed). Gradients are disabled.

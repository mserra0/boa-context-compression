# Attention Geometry

## The Problem

The frozen Qwen LLM was pretrained with a **fully causal** attention mask: every position attends only to itself and prior positions. It has never seen any other attention geometry. When soft prompts replace the input context, the training-time mask over `[soft_prompts | gen_tokens]` must reproduce that same causal geometry — otherwise the prefix is processed *off-distribution* relative to how Qwen was trained, and the frozen weights misinterpret it.

A bidirectional prefix (every soft prompt attending to every other) would be off-distribution input for the frozen model, because Qwen never attends bidirectionally. The geometry below is therefore **fully causal**.

---

## The Mask: Fully Causal over [prefix | gen]

For both compressors, the training forward uses a **4D attention mask** that is a single causal (lower-triangular) mask over the concatenated `[soft_prompts (C) | gen_tokens (G)]` sequence:

```
              soft_prompts (C)     gen tokens (G)
soft_prompts  [     causal      |     blocked      ]
gen tokens    [   full attend   |     causal        ]
```

- **Prefix × Prefix**: causal — soft prompt `C_i` attends to `C_j` for `j ≤ i`
- **Gen × Prefix**: fully visible — every generated token can see all prefix positions (a gen position `≥ C` sits after the whole prefix under the triangular mask)
- **Gen × Gen**: causal — standard left-to-right mask
- **Prefix × Gen**: blocked — soft prompts do not attend to future generated tokens

This is constructed in `build_causal_mask` (`src/utils/attention.py`), called by `PromptDestiller._create_causal_mask` (`src/model.py`), and cached by `(prefix_len, gen_len)` pair in `self._mask_cache` (FIFO-bounded: the oldest-inserted entry is evicted, capacity `MASK_CACHE_CAPACITY = 64`).

Note: the anchor position `C-1` (the last real soft prompt, which predicts the first generated token) attends to the entire real prefix under either geometry — the last row of a triangular mask is all-`True`. So the predicted-GEN_1 logit is identical to a bidirectional-prefix variant; only the *intermediate* prefix positions change. This is why the train/infer alignment check (`14`) is robust to the geometry switch.

---

## Combining with Padding

The geometry mask is `[1, 1, C+G, C+G]` (broadcastable over the batch). It is combined with the per-sample 2D padding mask in `build_additive_mask` (`src/utils/attention.py`):

```python
final_mask = causal_mask & (full_padding_mask[:, None, None, :] > 0.5)
final_mask.diagonal(dim1=-2, dim2=-1).fill_(True)

attention_mask = torch.zeros_like(final_mask, dtype=dtype)
attention_mask.masked_fill_(~final_mask, torch.finfo(dtype).min)
```

1. **Intersection:** `causal_mask & (full_padding_mask > 0.5)` blocks attention to left-pad slots while preserving causal direction.
2. **NaN Guard:** The diagonal is forced `True` after this conjunction so that even if a row were fully padded, it keeps a non-empty attention target and avoids producing all `-inf` (which turns into `NaN` after softmax).
3. **Additive Bias Conversion:** Valid positions get `0.0`, while masked positions receive `torch.finfo(dtype).min` (e.g. `-3.4028e+38` for FP32).

---

## Position IDs

Position IDs follow the HuggingFace Qwen cumsum recipe:

```python
position_ids = attention_mask.long().cumsum(-1) - 1
position_ids.masked_fill_(attention_mask == 0, 1)
```

For the Mamba2 path with left-padded soft prompts, this ensures:
- Padding positions get position ID `1` (harmless, they are masked out of attention)
- The first real soft prompt is at position `0`
- Generated token `G_i` is at position `C + i`

---

## Inference Alignment

`generate_compressed` reproduces the training geometry in two phases:

### Phase 1 — Prefix forward

```python
# Build causal mask for C prefix positions
causal = torch.tril(torch.ones((C, C), dtype=torch.bool))
prefix_allowed = (compressed_mask[:, None, None, :] > 0.5) & causal[None, None]  # causal ∩ pad cols
additive = torch.where(prefix_allowed, 0.0, float('-inf'))

# Run LLM backbone to seed KV cache
outputs = llm.model(
    inputs_embeds=soft_prompts,
    attention_mask=additive,
    position_ids=prefix_position_ids,
    use_cache=True,
)
past_key_values = outputs.past_key_values
```

### Phase 2 — Autoregressive decode

Each new token attends to all cached prefix KVs and causally to prior generated tokens. This is standard greedy/beam decode using `past_key_values` from phase 1.

The first generated token's logit is verified to match `student_logits[:, 0, :]` from the training forward to relative error < 5×10⁻³.

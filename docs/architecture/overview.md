# Architecture Overview

## High-Level Design

Boa trains a Mamba-based SSM to distill a frozen LLM's long-context understanding into a compact sequence of soft prompts. The core insight is that an SSM can read an arbitrarily long context in linear time and summarize it into a fixed-size continuous representation that a Transformer LLM can condition on.

```
Long Input  ──►  Compressor (SSM)  ──►  Soft Prompts [B, C, H_llm]  ──►  Frozen LLM  ──►  Answer
               linear O(L)              C = ceil(L / stride)              (Qwen2.5)
```

Only the compressor and its linear projector (or KVProjector) are trained. The LLM is frozen — every parameter gets `requires_grad = False` and the module is put in `.eval()` (`src/model.py`).

**Compilation is opt-in, not the default.** `compile_llm` defaults to `False` (`src/model.py`); when it is set, `__init__` compiles the *inner* transformer in place with `self.llm.model.compile(dynamic=True)` (`src/model.py`). Two details matter:

- It compiles `self.llm.model`, not `self.llm`. `torch.compile(self.llm)` is a no-op, because the `ForCausalLM` wrapper delegates to the uncompiled inner module. `lm_head` stays eager.
- `dynamic=True` is required because `(B, C+G)` changes every batch; without it each new shape triggers a recompile.

In practice most configs turn it off: per-batch shape churn made compilation a net loss on the main training path, and the KV path's `DynamicCache` decode shapes graph-break outright (see the comments in `configs/train/overfit_kv_pathB.yaml` and `configs/train/config_01Comp_QC_Mamba2L_bctx.yaml`). Treat `compile_llm: true` as an experiment to benchmark, not a default.

---

## Supported Architectures

- **Frozen LLM Backbone**: Qwen2.5 and Qwen3 decoder-only causal language models (`AutoModelForCausalLM`).
- **Compressor Backends**: Mamba1 (`MambaForCausalLM` from HuggingFace `transformers`) and Mamba2 (`MambaLMHeadModel` from `mamba_ssm`).

---

## Components

### `BaseCompressor` (`src/compressors.py`)

Abstract base class defining the compression interface. Both backends inherit from it. `src/model.py` re-exports it (and both backends) for backward compatibility — the implementations live in `src/compressors.py`.

```python
class BaseCompressor(nn.Module):
    def __init__(self, llm_hidden_dim: int, device: torch.device, stride: int = 2)
    def forward(compressor_inputs: dict) -> (soft_prompts [B, C, H_llm], compressed_mask [B, C])
    def compress_single(input_ids) -> (soft_prompts [1, C, H_llm], compressed_mask [1, C])
```

`stride` is an **explicit integer**: one of every `S` compressor hidden states is kept, so `C = ceil(L / S)`. `stride: 10` means one soft prompt per 10 input tokens.

Configs set `stride` directly. `resolve_stride()` (`src/utils/config.py`) reads it and only falls back to the legacy fractional `compression_ratio` (`stride = round(1 / ratio)`) when `stride` is absent, so checkpoints saved before the rename still load. New configs should never use `compression_ratio`.

---

### `MambaCompressor` — Mamba v1, padded batches

Wraps `MambaForCausalLM` (HuggingFace). Accepts left-padded batches of shape `[B, I]`.

**Forward pass** (`src/compressors.py`):
1. Run the Mamba backbone over `input_ids [B, I]` → `hidden_states [B, I, H_mamba]`
2. Compute real sequence lengths from `attention_mask`
3. **Stride-anchor** to `I-1` — the last column, which the left-padding invariant makes the last real token of *every* sample. Walk backwards in steps of `stride`. Because the columns are shared, `C` is a single batch-wide number: `C = max(1, ceil(max_real_len / stride))`
4. A slot is valid for sample `b` only if its column lands at or after `first_real[b] = I - real_len[b]`; otherwise it is `0` in `compressed_mask` (the gather still reads a clamped column, but the value is masked)
5. Project: `[B, C, H_mamba] → [B, C, H_llm]`

The gather itself is `strided_gather` (`src/utils/compression.py`) or `mamba1_context_gather` when `context_start` is given (B-ctx layout); see [compressors.md](compressors.md#mambacompressor--mamba-v1) for details.

In `PromptDestiller.forward` the Mamba1 pad slots are then overwritten with the LLM's pad-token embedding (`_replace_pad_soft_prompts`, `src/model.py`, called at `src/model.py`) rather than left at the projector's output; they stay masked either way.

---

### `Mamba2Compressor` — Mamba v2, packed sequences

Wraps `mamba_ssm.MambaLMHeadModel`. Accepts a single packed flat tensor — no padding enters the SSM at any point.

**Forward pass** (`src/compressors.py`):
1. Receive `input_ids [1, T]` (all sequences concatenated), `seq_idx [1, T]` (sequence boundary markers), `lengths [B]`
2. Run backbone: `hidden_states [T, H_mamba]` (flat, no batch dimension)
3. Split by sequence length; for each sequence of length L, stride from position `(L-1) % stride` forward: `hidden[start::stride]` (or `mamba2_context_start` when `context_start` is given)
4. Left-pad each strided sequence to `C_max`; assemble `[B, C_max, H_mamba]`
5. Project to `H_llm`; zero padded positions via `soft_prompts * compressed_mask.unsqueeze(-1)`

The `seq_idx` tensor resets the SSM state at sequence boundaries, making packed inference numerically equivalent to running each sample independently (verified: relative diff < 5×10⁻³).

---

### `PromptDestiller` — Main training module

Orchestrates compressor, frozen LLM, student layout, and KD loss.

Full signature (`src/model.py`):

```python
class PromptDestiller(nn.Module):
    def __init__(self, mamba_model_name: str, llm_model_name: str, device: torch.device,
                 stride: int = 2, temperature: float = 2.0,
                 max_new_tokens: int = 50, compressor_type: str = "mamba",
                 loss_type: str = "kl", confidence_mode: str = "max_prob",
                 compile_llm: bool = False, student_template: str = "bare",
                 compress_question: bool = True, prompt_mode: str = "embed",
                 llm_dtype: str = None, amp: bool = False,
                 anchor_init_token_ids: list = None)
```

| Argument | Meaning |
|---|---|
| `stride` | Compression stride `S`; `C = ceil(L / S)` |
| `compressor_type` | `"mamba"` or `"mamba2"` — anything else raises `ValueError` |
| `loss_type` / `confidence_mode` | KD objective: `"kl"` or `"confidence_kl"`; see `src/losses.py` |
| `compile_llm` | Opt-in `torch.compile` of the inner LLM backbone (default off) |
| `student_template` / `compress_question` | Student layout: `bare` (Design A) / `wrapped` with `compress_question=True` (B-qc) / `wrapped` with `compress_question=False` (B-ctx) — see [student-templates.md](student-templates.md) |
| `prompt_mode` | `"embed"` (soft prompts as input embeddings) or `"kv"` (direct KV cache injection) — see [kv-injection.md](kv-injection.md) |
| `llm_dtype` / `amp` | Precision contract (`llm_dtype`: `"bfloat16"`, `"float32"`, etc.; `amp`: boolean) |
| `anchor_init_token_ids` | Seed tokens for the trainable KV anchor (`prompt_mode="kv"`, bare template) |

**Key attributes:**
- `self.llm` — the frozen decoder (Qwen2.5 / Qwen3), `requires_grad=False` and `.eval()`; the inner backbone is compiled only when `compile_llm` is set (`src/model.py`)
- `self.compressor` — `MambaCompressor` or `Mamba2Compressor` (`src/model.py`). Directly accessed via `model.compressor` (no `self.core` wrapper)
- `self.loss_type` / `self.confidence_mode` — dispatch to the free functions `kl_loss()` / `confidence_weighted_kl_loss()` in `src/losses.py` (`src/model.py`)
- `self._eos_token_ids` — sorted union of the eos ids declared by `config` and `generation_config` (`src/model.py`); Qwen declares a *list*, not a scalar
- `self._mask_cache` — dict of causal geometry masks keyed by `(prefix_len, gen_len)`, capped at `MASK_CACHE_CAPACITY = 64` (`src/utils/attention.py`, `src/model.py`)
- `self.kv_projector` / `self.anchor_embed` — `None` unless `prompt_mode="kv"` (`src/model.py`)

---

## Forward Pass (Training)

```
compressor_inputs  ──►  Compressor  ──►  soft_prompts [B, C, H_llm]
                                         compressed_mask [B, C]
                                              │
                                              ▼
llm_inputs  ──►  LLM.generate()  ──►  teacher_logits [B, G, V]
                                       generated_ids  [B, G]
                                              │
                                              ▼
             embed(generated_ids)  ──►  gen_embeds [B, G, H_llm]
                                              │
              cat(soft_prompts, gen_embeds)   │
                    │                         │
                    └────────────────────────►│
                                              ▼
                         combined_embeds [B, C+G, H_llm]
                         attention_mask  [B, 1, C+G, C+G]   (4D)
                         position_ids    [B, C+G]            (cumsum-based)
                                              │
                         LLM.backbone()  ──►  hidden [B, C+G, H_llm]
                         lm_head()       ──►  student_logits [B, G, V]
                                              │
                         KL(student / teacher) * valid_mask  ──►  loss
```

**Loss computation:**

```python
valid = (generated_ids != pad_token_id)   # [B, G]
loss  = (per_token_kl * valid).sum() / valid.sum() * temperature**2
```

Temperature defaults to 2.0. Positions after EOS (where `generated_ids == pad_token_id`) are masked out.

---

## Inference

`PromptDestiller.generate_compressed(soft_prompts, compressed_mask, max_new_tokens)` reproduces the training-time geometry:

1. **Prefix forward** — run LLM backbone over soft prompts with the causal prefix attention mask, seeding the KV cache
2. **Autoregressive decode** — generate tokens one by one, attending causally to prior generated tokens and to all prefix KV entries

This matches the geometry of the training forward pass exactly. See [Attention Geometry](attention.md) for the mask construction.

---

## Notation

| Symbol | Meaning |
|---|---|
| B | Batch size |
| I | Input sequence length (padded) |
| T | Total tokens in packed batch `(= sum of lengths)` |
| C | Compressed sequence length `= ceil(L / stride)` |
| G | Generated sequence length (max_new_tokens) |
| H_mamba | Mamba hidden dimension |
| H_llm | LLM hidden / embedding dimension |
| V | Vocabulary size |
| S | Stride `= int(1 / compression_ratio)` |

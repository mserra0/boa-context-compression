# KV-Injection Compression (`prompt_mode: kv`)

How Boa conditions the frozen Qwen2.5-0.5B LLM by writing
**per-layer key/value tensors** into its attention cache, instead of feeding soft
prompts as input embeddings. Two geometries are supported:
Design A (bare template) and **Path B** — the wrapped chat template
(B-qc) and context-only compression (B-ctx), both with **context-first ordering**
(see §6.1).

---

## 1. Why KV instead of embeddings

In the embedding path the compressor produces `C` soft prompts of shape
`[B, C, H_llm]` (`H_llm = 896`). Those vectors enter the LLM **only at the input
layer**. From there the frozen transformer is free to transform them however its
weights dictate — the compressor has one shot to influence the network, at layer 0.

In the KV path the compressor instead produces the **key and value vectors for
every attention layer directly**. Qwen2.5-0.5B has:

| Quantity | Value |
|---|---|
| Layers `L` | 24 |
| KV heads `kv_heads` (GQA) | 2 |
| Head dim `head_dim` | 64 |
| K and V | 2 |

So per compressed token the KV representation carries
`L · 2 · kv_heads · head_dim = 24 · 2 · 2 · 64 = 6144` dimensions, versus `896`
for an embedding. **~6.9× the conditioning capacity per token**, injected at every
layer rather than just the first. This is the documented reason KV-conditioning
tends to beat embedding-space soft prompts (cf. 500xCompressor).

### Key difference vs 500xCompressor

In 500xCompressor the *encoder is the LLM itself*, so its KV cache is extracted by
slicing the model's own attention cache — the K/V already live in the right space.
**Our compressor is a separate Mamba SSM.** Its hidden states live in Mamba space,
not Qwen attention space. Therefore we cannot slice anything; the K/V must be
**learned via a projection** (`KVProjector`). That projection is the only new
trainable weight of substance in this path.

---

## 2. End-to-end pipeline

```
compressor_inputs
      │
      ▼
MambaCompressor.gather_hidden        # raw strided Mamba hidden, BEFORE the embedding projector
      │   hidden [B, C, H_m]  +  compressed_mask [B, C]
      ▼
KVProjector                          # learned Linear: Mamba space → Qwen K/V space
      │   per_layer_kv [L, 2, B, kv_heads, C, head_dim]
      ▼
build_kv_cache                       # apply RoPE to keys only, pack into DynamicCache
      │   DynamicCache (24 layers populated)
      ▼
frozen Qwen2.5  decode from learned anchor embedding → student_logits
      │
      ▼
KL distillation loss vs teacher_logits
```

Only **`KVProjector`** and **`anchor_embed`** receive gradients. The Mamba backbone
trains too (it is the upstream of `gather_hidden`); the LLM stays fully frozen, and
the compressor's embedding-space `projector` is explicitly frozen so DDP does
not flag it as unused (see §7).

---

## 3. Step 1 — tapping the Mamba hidden (`gather_hidden`)

`src/compressors.py` → `strided_gather` (`src/utils/compression.py`).

The embedding path normally does `mamba_backbone → projector → soft prompts`. The
KV path stops one step earlier: it takes the **raw Mamba hidden states** and applies
the same strided gather the embedding path uses, but skips the projector (the
KVProjector replaces it).

`strided_gather` is **left-anchored** over a left-padded batch:

- The batch is left-padded, so the last real token is always at index `seq_len - 1`.
- It walks backwards from that last token in steps of `stride`, filling slots
  `C-1, C-2, …` So **slot `C-1` is always each sample's last real token** — the
  most semantically loaded position. Earlier slots are strided checkpoints going
  back through the document.
- `compressed_mask[b, c] = 1` where slot `c` lands on a real token, `0` where it
  falls into the left padding (short samples).

Output: `hidden [B, C, H_m]`, `compressed_mask [B, C]`.

---

## 4. Step 2 — projecting to per-layer K/V (`KVProjector`)

`src/kv_projector.py`. This module turns one Mamba hidden vector per compressed slot
into the keys and values for **all 24 LLM layers at once**. It has four pieces, each
there for a training-stability reason (the rationale lives in §9–§10; here is what the
code does).

```python
self.norm   = nn.RMSNorm(mamba_hidden)                 # (C) input-scale normaliser
self.k_proj = nn.Linear(mamba_hidden, L·kv_heads·head_dim)   # (B) keys, all layers
self.v_proj = nn.Linear(mamba_hidden, L·kv_heads·head_dim)   # (B) values, all layers
self.gain   = nn.Parameter(torch.ones(L))              # (A) per-layer learnable scale
# (A) small-init: k_proj/v_proj weights ~ normal(std=0.02), bias = 0
```

Forward:

```python
h = self.norm(hidden)                                  # [B, C, H_m], scale-normalised
k = self.k_proj(h).view(B, C, L, kv_heads, head_dim)
v = self.v_proj(h).view(B, C, L, kv_heads, head_dim)
x = torch.stack([k, v], dim=2)                         # [B, C, 2, L, kvh, hd]
x = x.permute(3, 2, 0, 4, 1, 5).contiguous()           # [L, 2, B, kvh, C, hd]
return x * self.gain.view(L, 1, 1, 1, 1, 1)            # per-layer scale
```

Reading it piece by piece:

1. **RMSNorm first.** The Mamba hidden's magnitude drifts as the backbone trains.
   Normalising before the projection means the injected-K/V scale depends on the
   *direction* of the hidden, not its (wandering) length.
2. **Separate `k_proj` and `v_proj`.** Keys drive attention *weights* (and get
   rotated by RoPE); values carry *content* (never rotated). They have different jobs
   and different statistics, so they get independent matrices and independent
   gradients. Each outputs `L · kv_heads · head_dim` — one full set of K (or V) for
   every layer.
3. **Stacking into the output contract.** `k` and `v` are each
   `[B, C, L, kv_heads, head_dim]`. `torch.stack([k, v], dim=2)` puts the K/V choice
   at axis 2, giving `[B, C, 2, L, kv_heads, head_dim]`; the permute reorders to the
   cache-friendly `[L, 2, B, kv_heads, C, head_dim]`. **The `[L, 2, …]` output layout
   is what `build_kv_cache` and `_kv_forward` consume.**

   > Note: K and V come from separate `k_proj` / `v_proj` matrices rather than one fused
   > `Linear` reshaped with `view(B, C, 2, L, …)`, whose channel order would otherwise be
   > load-bearing (a `view(B, C, L, 2, …)` would silently swap keys and values).
4. **Per-layer gain.** `gain` is one learnable scalar per layer, init `1.0`
   (neutral — start unchanged). Multiplying the `[L, …]` tensor by
   `gain.view(L,1,1,1,1,1)` lets the optimiser dial each layer's injection up or down
   independently (e.g. quiet the deep layers early, ramp them later).

Final layout `[L, 2, B, kv_heads, C, head_dim]`:
- axis 0 = layer, axis 1 = `{0: keys, 1: values}`,
- then the standard attention shape `[B, kv_heads, C, head_dim]`.

These are **pre-RoPE** keys. Values are never rotated.

---

## 5. Step 3 — RoPE + cache packing (`build_kv_cache`)

`src/utils/kv_cache.py`.

Qwen uses Rotary Position Embeddings. When the model fills its own cache during a
normal forward, it applies RoPE to keys **before** storing them, then rotates the
query at decode time by the query's absolute cache position. To make our injected
keys consistent with that convention, we must **pre-rotate the keys at the positions
we want them to occupy**.

```python
cos, sin = rotary_emb(ref, position_ids)        # position_ids = comp_pos [B, C]
for layer_idx in range(L):
    k = per_layer_kv[layer_idx, 0]
    v = per_layer_kv[layer_idx, 1]
    _, k_rot = apply_rotary_pos_emb(k, k, cos, sin)   # rotate keys only
    cache.update(k_rot.contiguous(), v.contiguous(), layer_idx)
```

Notes:
- `cos/sin` computed **once** (positions are identical across layers) and reused for
  all 24 layers — RoPE geometry does not vary by layer.
- `apply_rotary_pos_emb(q, k, …)` rotates both its first two args; we pass `k` twice
  and keep only the rotated key.
- **transformers 5.9.0 API**: `DynamicCache()` + `.update(k, v, layer_idx)` per
  layer. No `from_legacy_cache`. After the loop the cache has all 24 layers
  populated and behaves exactly like a cache the model produced itself.

### Positions (`comp_pos`)

```python
comp_pos = compressed_mask.long().cumsum(-1) - 1
comp_pos = comp_pos.clamp_min(0)
```

This assigns the compressed tokens **left-anchored positions `0 … C_real-1`**. The
`cumsum-1` skips padded slots (their cumsum doesn't advance), and `clamp_min(0)`
keeps the (masked-out anyway) pad positions valid. So real compressed tokens occupy
RoPE positions `0..C_real-1`, exactly as if they were the first `C_real` tokens of a
normal sequence.

---

## 6. Step 4 — decoding from a scaffold

After the cache is seeded, the decoder needs *something* to attend with to produce the
first answer token `T_0` — the answer has no real input token at its start. That role
is filled by a **scaffold**: the literal tokens placed *after* the injected context
cache and *before* the answer. `_build_kv_scaffold(B, dtype, compressor_inputs)`
returns `(embeds [B, A, H], mask [B, A])` and dispatches on the template:

| template | scaffold | trainable? |
|---|---|---|
| `bare` (Path A) | `[anchor]` (`A=3` by default) | yes (`anchor_embed`) |
| `wrapped`, `compress_question=True` (B-qc) | `[tpl_prefix \| tpl_suffix]` | no (frozen template) |
| `wrapped`, `compress_question=False` (B-ctx) | `[tpl_prefix \| question \| sep \| tpl_suffix]` | no (frozen template) |

The mask is all-ones except over the **left-padded literal question** in B-ctx. The
last scaffold token is always the real `tpl_suffix` anchor (`<|im_start|>assistant\n`)
that predicts `T_0`, for every sample — the suffix is never padded.

### Path A: the learned anchor

Path A is the **bare** template: no question literal, no prefix/suffix text. The whole
input (question + context) is compressed into the KV cache, and a learned **anchor
scaffold** `self.anchor_embed [A, H_llm]` — `A` trainable "start querying here"
vectors — stands in for the generation prompt.

**Initialisation (§10.D).** The anchor is seeded from the embeddings
of Qwen2.5's **full assistant-turn scaffold**, not a random vector:

```python
seed = self.llm.get_input_embeddings()(torch.tensor(anchor_init_token_ids))   # [A, H]
self.anchor_embed = nn.Parameter(seed.clone())     # default ids = [151644, 77091, 198]
self.num_anchors  = len(anchor_init_token_ids)     # A = 3 by default
```

The Qwen2.5 and Qwen3 tokenizers have **no BOS token** (`tokenizer.bos_token is None`); its chat turns begin
with `<|im_start|>assistant\n` = ids `[151644, 77091, 198]`. The frozen LLM was
trained to generate the answer **immediately after exactly that 3-token prefix**, so
seeding the anchor with all three gives it the precise scaffold it expects — the
single-token version was a strict subset of this. The ids are configurable via
`anchor_init_token_ids` in the config (a single-token list gives a one-token anchor). The anchors stay fully trainable; this only sets *where they start*.

At decode the input sequence is `[scaffold (A tokens) | teacher answer embeds]`:

```python
scaffold, scaffold_mask = self._build_kv_scaffold(B, ref_dtype, compressor_inputs)
A         = scaffold.shape[1]
dec_input = cat([scaffold, answer_embeds], dim=1)              # [B, A+G, H]
```

### Positions of the decoder stream

The scaffold and answer tokens continue **per sample** from each sample's real
compressed length, **pad-aware** via cumsum (bare/B-qc have no scaffold pad, so this
reduces to a contiguous `arange`):

```python
C_real   = compressed_mask.sum(dim=1)                          # [B]
dec_mask = cat([scaffold_mask, ones(B, G)], dim=1)             # [B, A+G]
dec_pos  = C_real.unsqueeze(1) + (dec_mask.cumsum(-1) - 1).clamp_min(0)
```

So the scaffold occupies positions `C_real .. C_real+A-1`, the first answer token
`C_real+A`, etc. (left-padded question slots in B-ctx consume no position). The query's
RoPE rotation at decode time then lines up correctly against the keys we pre-rotated at
`0..C_real-1`. **This is the whole reason positions had to be explicit and
left-anchored** — query/key relative geometry must be coherent.

### Attention mask

```python
attn_mask = cat([compressed_mask, dec_mask], dim=1)   # [B, C + A+G]
```

The decoder may attend to every **real** compressed slot (pad slots masked) plus the
scaffold (minus padded question slots) and the answer tokens it has seen. The model's
own causal mask handles the left-to-right constraint within the answer; the padding
mask blocks pad-KV leakage.

### Loss alignment

```python
student_logits = self.llm.lm_head(out.last_hidden_state[:, A-1:A-1+G, :])
```

The **last** scaffold position `A-1` predicts `T_0`; position `A-1+G-1` predicts
`T_{G-1}`; the final position is dropped. (With `A=1` this reduces to the plain
`[0:G]` slice.) We slice the hidden states **before** `lm_head` so the dropped
position never pays for a full vocab-sized matmul (a performance optimization — `lm_head` is the
single biggest matmul in the model). Then standard KL distillation against the
teacher logits, same as the embedding path.

### 6.1 Path B — context-first ordering

The embedding path puts the compressed block in the **middle** of the chat template:
`[prefix | question | sep | soft_ctx | suffix]`. KV injection can't reproduce that
directly — the injected cache must sit at positions `0..C-1` (the front). Path B
therefore uses **context-first ordering**:

```
cache:  [ ctx-KV (0..C_real-1) ]
decode: [ tpl_prefix | question | sep | tpl_suffix | answer ]   (B-ctx)
        [ tpl_prefix | tpl_suffix | answer ]                    (B-qc)
```

The system prompt / question come **after** the context cache rather than before it
(the documented trade-off of this design). For **B-ctx**, only the context is
compressed: `MambaCompressor.gather_hidden` takes a `context_start` and gathers only the
context sub-range (`mamba1_context_gather`), exactly mirroring the embedding-path
`forward`; the literal question is embedded into the scaffold by `_build_question_block`
(left-padded to the batch max, hence the only padding in `scaffold_mask`).

---

## 7. Inference (`generate_compressed_kv`)

`src/model.py`. Same cache seeding, then autoregressive greedy decode. The **first
step feeds the whole scaffold** (`_build_kv_scaffold`, same as training) and reads the
last position's logit for `T_0`; every later step feeds one token. A running
`seen_mask` tracks which decode-stream keys are real (so B-ctx's padded question keys
stay masked across all steps). Three things that look optional but are **mandatory**:

1. `cache = out.past_key_values` **inside the loop** — each step must absorb the
   newly produced K/V or later steps attend to a stale cache and emit garbage.
2. First step injects all `A` scaffold positions at once; positions are pad-aware:
   `pos = (C_real + (cat([seen_mask, cur_mask]).cumsum(-1) - 1))[:, -n:]`.
3. Per-step mask `cat([compressed_mask, seen_mask, cur_mask])` — the decode geometry
   must keep matching the training geometry.

Per-row EOS handling emits `pad_id` once a row finishes, and the loop breaks when all
rows are done.

---

## 8. Implementation notes

- **Double-Mamba forward**: the KV branch in `forward()` returns *before*
  `self.compressor(compressor_inputs)` is called, otherwise the backbone would run
  twice (once via `gather_hidden`, once via the embedding path) and the second
  result would be discarded.
- **DDP unused param**: in KV mode the compressor's embedding-space `projector` never
  contributes to the loss → DDP (with `find_unused_parameters=False`) errors on
  step 2. The KV guard therefore calls `self.compressor.projector.requires_grad_(False)`.
- **Launch**: `main.py` calls `init_process_group` unconditionally → launch with
  `torchrun --nproc_per_node=1` (sets `MASTER_ADDR`), not plain `python main.py`.
- **Guards**: KV path rejects `compressor_type` outside `{mamba, mamba2}`, `student_template`
  outside `{bare, wrapped}`, and `compress_question == False` without `wrapped` (B-ctx requires
  the template), all with explicit `ValueError`s.

---

## 9. Expected initial loss relative to the embedding path

The initial loss is expected to be higher than the embedding path's, for two compounding reasons:

1. **All-layer corruption.** An untrained embedding projector corrupts only the
   layer-0 input; the other 23 layers still process whatever they receive with
   calibrated weights. An untrained `KVProjector` injects wrong keys *and* values
   into **all 24 attention layers simultaneously** — every layer's attention is
   wrong from the start, and there is no clean teacher-forced position to anchor on.
   Errors are expected to compound through depth.
2. **Scale mismatch.** Qwen's real `k_proj`/`v_proj` outputs are calibrated to the
   model's attention temperature (the `1/√head_dim` softmax scaling). A freshly
   initialised `KVProjector` emits K/V at an arbitrary scale, so the injected keys
   can produce near-uniform or saturated attention until training rescales them.

Both are expected to be initialisation effects rather than architectural limits; §10 describes how the
projector is initialised to mitigate them.

---

## 10. Projector design notes

The projector (`src/kv_projector.py`, `src/model.py`) uses the following design choices.

### A. K/V initialisation
- **Small-init the projector** — `k_proj`/`v_proj` weights `normal(std=0.02)`,
  bias zero, so injected K/V start near zero and attention starts near-uniform.
- **Per-layer learnable gain** — `self.gain = Parameter(ones(L))`, init neutral,
  broadcast `[L,1,1,1,1,1]` over K/V; lets the optimiser attenuate/ramp each layer.
- **RMS-match the injected K/V** — partially subsumed by §C (RMSNorm on the input
  fixes the *input*-scale term). Matching to Qwen's empirical `k_proj`/`v_proj`
  output statistics is not applied.

### B. Decoupled K and V projections
Separate `k_proj` and `v_proj` `Linear`s. Keys (rotated,
drive attention *weights*) and values (carry *content*) have different statistics, so
they get independent projections and independent gradients.

### C. Normalisation before projection
`nn.RMSNorm(mamba_hidden)` applied to the Mamba hidden before `k_proj`/`v_proj`,
decoupling injected-K/V scale from the backbone's (drifting) activation magnitude.

### D. Anchor from a real token
`anchor_embed` is a `[A, H]` trainable scaffold seeded from the embeddings of
Qwen2.5's **full assistant-turn prompt** `<|im_start|>assistant\n` =
`[151644, 77091, 198]` (so `A = 3`), configurable via `anchor_init_token_ids`.
The Qwen2.5 and Qwen3 tokenizers have **no BOS token** (`bos_token = None`) and was trained to generate the
answer immediately after that exact 3-token prefix — so the scaffold hands the frozen
LLM the precise context it expects, rather than a single token or a random vector. A
single-token list gives a one-token anchor. The decode then slices logits
from the **last** scaffold position (`A-1`); see §6.


---

## File reference

| Concern | File |
|---|---|
| Projector (Mamba → per-layer K/V) | `src/kv_projector.py` |
| RoPE + DynamicCache packing | `src/utils/kv_cache.py` |
| Raw strided hidden tap | `src/compressors.py` (`gather_hidden`), `src/utils/compression.py` (`strided_gather`) |
| KV forward + anchor + loss | `src/model.py` (`_kv_forward`, `__init__` KV guard) |
| KV inference | `src/model.py` (`generate_compressed_kv`) |
| Overfit config | `configs/train/overfit_kv_pathA.yaml` |

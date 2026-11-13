# Student Templates: Bare, Wrapped (B-qc), and Question-Literal (B-ctx)

How the compressed soft prompts are presented to the frozen LLM during the **student** forward
pass. Three layouts exist, selected by two config flags. All three share the same compressor,
the same KD loss, and the same teacher; they differ only in what surrounds the soft prompts and
how the prediction anchor is placed.

| Design | `student_template` | `compress_question` | Student layout |
|---|---|---|---|
| **A** (bare) | `"bare"` | — (ignored) | `[ soft \| gen ]` |
| **B-qc** (wrapped, default) | `"wrapped"` | `true` | `[ tpl_prefix \| soft \| tpl_suffix \| gen ]` |
| **B-ctx** (question-literal) | `"wrapped"` | `false` | `[ tpl_prefix \| question \| sep \| soft_ctx \| tpl_suffix \| gen ]` |

- `tpl_prefix` / `tpl_suffix` — real Qwen chat-template token **embeddings**, precomputed once.
- `soft` / `soft_ctx` — compressed soft prompts `[B, C, H_llm]`, left-padded.
- `question` — the literal question, embedded with the **LLM** tokenizer, left-padded (B-ctx only).
- `sep` — a fixed `"\n"` embedding between question and soft block (B-ctx only).
- `gen` — teacher-forced answer token embeddings (training) or autoregressive (inference).

Config (`configs/train/*.yaml`, `model:` block):

```yaml
student_template: "wrapped"   # "bare" (A) | "wrapped" (B-qc / B-ctx)
compress_question: false      # wrapped only: true=B-qc, false=B-ctx. Default true.
```

Both flags thread through `PromptDestiller.__init__` (`src/model.py`).

---

## Why a Template At All? (Design A → B-qc)

Design A feeds the frozen LLM `[soft | gen]` with nothing else. The LLM was never trained to see
a bare sequence of dense vectors followed by an answer — there is no system prompt, no
`<|im_start|>user` / `<|im_start|>assistant` scaffold. The model generates **out of its native
chat distribution**.

**B-qc** fixes this by wrapping the soft prompts in the real Qwen chat template, so the LLM sees
exactly the format it was instruction-tuned on, with the soft prompts sitting where the document
text normally goes:

```
<|im_start|>system ... <|im_start|>user\n   ← tpl_prefix (P tokens)
[ soft prompts ]                            ← document slot (C tokens)
<|im_end|>\n<|im_start|>assistant\n         ← tpl_suffix (S tokens)
[ answer ]                                  ← gen (G tokens)
```

The split is derived (never hardcoded) by rendering the template around a sentinel and slicing on
it: `split_template_around_doc` (`src/utils/student_layout.py`, re-exported in `src/model.py`). Prefix/suffix are embedded **once** in
`__init__` and stored as buffers `_tpl_prefix_emb [P,H]`, `_tpl_suffix_emb [S,H]`
(`src/model.py`); the forward only broadcasts them across the batch — no per-batch
tokenization.

---

## B-ctx: Question Literal, Context Compressed

B-qc compresses the **whole** compressor input (question + context) into soft prompts. B-ctx asks
a different question: *what if the question is short and cheap, but the context is long?* Then it
is wasteful — and lossy — to compress the question. B-ctx instead:

1. Passes the **question verbatim** to the frozen LLM (literal Qwen embeddings).
2. Compresses **only the context** into soft prompts.
3. Still feeds the question to the **compressor** so the soft prompts are *question-aware*.

So the question is fed **twice**: once to condition the SSM, once literally to the LLM. The
soft prompts become a question-conditioned summary of the context; the LLM reads the exact
question plus that summary.

```
<|im_start|>system ... <|im_start|>user\n   ← tpl_prefix (P)
[ question tokens ]                          ← literal Qwen embeds (Qmax, left-pad)
\n                                           ← sep (sep_len)
[ soft prompts of CONTEXT ]                  ← soft_ctx (Cmax, left-pad)
<|im_end|>\n<|im_start|>assistant\n          ← tpl_suffix (S)
[ answer ]                                   ← gen (G)
```

**Ordering rationale (question BEFORE soft):** the soft block is question-aware (the SSM saw the
question), the literal question is NOT context-aware, and `gen` sees both. Putting the question
first keeps the causal flow consistent with how the SSM conditioned.

The `sep = "\n"` is its own fixed buffer `_tpl_sep_emb` (`src/model.py`), built like
prefix/suffix. `tpl_prefix` / `tpl_suffix` are **unchanged** from B-qc — B-ctx is purely additive.

---

## Batch Geometry: Fixed Anchor via Left-Padding

The prediction anchor `T_0` (first answer token) must be predicted from a **fixed column index**
across the batch so the logit slice is uniform. Both variable-width blocks are left-padded to the
batch maximum:

- soft block → `C_max = max(1, max(compressed_lengths))`
- question block (B-ctx) → `Q_max = max(1, max(question_lengths))` (`_build_question_block`,
  `src/model.py`)

| Design | Anchor offset (index of `assistant\n`'s next prediction) |
|---|---|
| B-qc | `P + C_max + S` |
| B-ctx | `P + Q_max + sep_len + C_max + S` |

`T_0` is predicted from index `anchor-1` (the last suffix token, the `assistant\n` cue) — the same
position the teacher predicts from. The student logit slice is `[:, anchor-1 : -1]`
(`generate_student`, `src/model.py`).

### Masking the pad columns

Left-pad slots inside the question and soft blocks must not leak. The geometry helper handles both
at once. For B-ctx the forward builds a single `prefix_region_mask` spanning
`[tpl_prefix | question | sep]`:

```python
prefix_region_mask = cat([ ones(B,P), q_mask, ones(B,sep_len) ])   # model.py
full_padding_mask, position_ids, anchor = wrapped_geometry(
    prefix_region_mask, compressed_mask, S, gen_mask)               # model.py
```

`wrapped_geometry` (`src/utils/student_layout.py`):

- `full_padding_mask = cat[prefix_region_mask, compressed_mask, suffix_ones, gen_mask]`
- `position_ids = cumsum(mask) - 1`, with pad slots forced to `1` — so **left-pad columns consume
  no position** and never land on the anchor.
- `anchor = prefix_width + C + S`.

B-qc uses the thin wrapper `wrapped_student_geometry` (`src/utils/student_layout.py`), which calls
`wrapped_geometry` with an all-ones prefix of width `P`. Same code path, no question/sep.

The 2D `full_padding_mask` is then intersected with the causal mask in the forward
(`src/model.py`); pad keys are blocked and the geometry stays strictly causal.

---

## Context-Only Stride (B-ctx)

The compressor input is `question + context`, but only the **context** sub-range is strided into
soft prompts. The change is minimal: keep the **end anchor** at the last real token; raise the
stride **lower bound** from `first_real` to `context_start`. Question hidden states are still
computed (they condition the SSM state) but are **never gathered**.

`context_start` is derived by tokenizing `question + sep` alone with the **compressor** tokenizer
(not a string search), and threaded through collate into `compressor_inputs["context_start"]`.
See [dataloading](../data/dataloading.md).

### Mamba1 — `mamba1_context_gather` (`src/utils/compression.py`)

End-aligned gather over the left-padded `[B, seq_len]` batch:

```python
last_real_pos = seq_len - 1
ctx_span      = (last_real_pos - context_start + 1).clamp_min(0)   # per row
C             = ceil(ctx_span.max() / stride)
gather_idx    = last_real_pos - (C-1-slot)*stride                  # [B, C]
valid         = (gather_idx >= context_start) & (gather_idx <= last_real_pos)
```

Slots landing before `context_start` are marked invalid (→ pad in `compressed_mask`).

### Mamba2 — `mamba2_context_start` (`src/utils/compression.py`)

Per packed segment of length `L` with context start `c0`:

```python
start = c0 + ((L - 1 - c0) % stride)     # floor at c0, keep last token L-1
kept  = range(start, L, stride)
# c0 >= L  → start >= L → empty slice (no context)
```

Both branches activate only when `compressor_inputs.get('context_start')` is present
(`src/compressors.py`); otherwise the compressors fall back to the standard
whole-input stride (B-qc / bare / Design A unchanged).

---

## Inference Parity

Training is teacher-forced; inference is autoregressive. Both must reproduce the **same**
geometry or the run is out-of-distribution.

- `generate_compressed` (`src/model.py`) seeds the KV cache with the same prefix:
  - bare → `[soft]`
  - B-qc → `[tpl_prefix | soft | tpl_suffix]`
  - B-ctx → `[tpl_prefix | question | sep | soft | tpl_suffix]`, requiring `question_ids`.
- **Hardened guard:** B-ctx asserts `question_ids is not None` in `generate_compressed`
  (`src/model.py`) — a silent fallback to the B-qc seed would corrupt generation. Loud,
  not silent.
- Mamba1 wrapped eval/inference routes through `generate_compressed` (not a bare-soft-prompt
  path), so train and infer geometry match.

B-ctx train / infer / generate agree numerically (rel ≈ 1.5×10⁻⁶, both backends), and the first-step
student and inference logits agree.

---

## Evaluation: Does It Close the Gap?

Results are reported as **normalised MRQA accuracy**, where `0%` = the frozen teacher with *no*
context (lower bound) and `100%` = the teacher *with* full context (upper bound). A compressor's
score is how much of that gap its soft prompts recover, averaged over SQuAD, NewsQA, TriviaQA, and
HotpotQA.

**The OOD gap (A → B-qc).** Design A's bare layout forces the LLM out of its chat distribution, so
it leaves accuracy on the table even when the soft prompts carry the right information. Wrapping
those *same* soft prompts in the native chat template (B-qc) lets the frozen LLM decode from the
format it was tuned on — recovering quality with **zero extra compression cost** (prefix/suffix are
fixed embeddings, not compressed tokens).

**The compression budget (B-qc → B-ctx).** B-qc spends stride budget compressing the question, which
is short and cheap to pass literally. B-ctx reallocates: question verbatim to the LLM, compression
spent **only on the long context**. Across the MRQA suite B-ctx is the **strongest configuration**,
reaching the highest normalised overall average among all evaluated compressors — the question is no
longer a lossy bottleneck, and the soft prompts stay question-aware via the SSM conditioning path.

The takeaway: **presentation and compression target are levers independent of the compressor.** B-qc
fixes *how* the soft prompts are read; B-ctx fixes *what* gets compressed. They compose, and both are
purely additive on top of the same backbone, KD loss, and teacher.

---

## Cross-References

- Forward pass overview — [overview.md](overview.md)
- Mamba1 vs Mamba2 stride/packing — [compressors.md](compressors.md)
- Attention mask geometry — [attention.md](attention.md)
- Data loading + `context_start` / `question_ids` threading — [dataloading.md](../data/dataloading.md)
- KD loss — [knowledge-distillation.md](../training/knowledge-distillation.md)

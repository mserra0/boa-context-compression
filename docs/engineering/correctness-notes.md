# Correctness Notes

Design decisions that keep training and evaluation correct in Boa. Each one prevents a failure that would not necessarily raise an error — non-convergence, gradient noise, train/inference gaps, or under-reported evaluation metrics.

---

## 1. Left/Right Padding Asymmetry

**Why:** The teacher LLM (Qwen) requires *left*-padded inputs for correct autoregressive generation. A compressor fed *right*-padded inputs would select padding positions for shorter samples in a variable-length batch, so the last selected hidden state would not be the last real token.

**Implementation:** Both the LLM tokenizer and the compressor tokenizer produce left-padded batches in `collate_fn`. The compressor's stride anchoring then correctly finds the last real token regardless of sequence length.

---

## 2. Stride Anchoring to Last Real Token

**Why:** A stride grid like `hidden[:, start::stride, :]` is anchored to absolute position 0. For a sample shorter than the batch maximum, position 0 is a padding position, so the "last" soft prompt would not represent the end of the context.

**Implementation:** The stride is anchored to the *last real token* of each sample:

```python
# MambaCompressor
offsets   = (C - 1 - slot_idx) * stride          # distance from end
gather_idx = (seq_len - 1) - offsets              # absolute position

# Mamba2Compressor (per-sequence, after unpacking)
start_idx = (L - 1) % stride
selected  = hidden[start_idx::stride]
```

This guarantees that position `C-1` in `soft_prompts` always corresponds to `hidden[seq_len-1]` — the hidden state that predicts the first generated token — for every sample in the batch.

---

## 3. Fully-Causal Attention Mask & 2D Expander

**Why:** A bidirectional prefix (every soft prompt attending to every other) would be off-distribution: frozen decoder-only LLMs (Qwen) were pretrained *only* with causal attention, so it distorts self-attention representations. Naive 2D boolean masks also do not interact reliably with SDPA requirements across PyTorch versions.

**Implementation:** A 4D **fully causal** mask `[B, 1, Q, K]` over `[soft_prompts | gen_tokens]`:

```
              soft_prompts (C)     gen tokens (G)
soft_prompts  [     causal      |     blocked      ]
gen tokens    [   full attend   |     causal        ]
```

`_create_causal_mask(C_max, gen_len)` and layout utilities in `src/utils/student_layout.py` construct additive 4D masks where unallowed or padded key positions (`mask[:, None, None, :] <= 0.5`) are filled with `torch.finfo(dtype).min`. See [Attention Geometry](../architecture/attention.md) for full details.

---

## 4. Per-Tokenizer Context Truncation

**Why:** Truncating context by character count before tokenization is not enough. Because the Mamba and Qwen tokenizers have different vocabularies, the same character-count truncation yields sequences of different token lengths. One tokenizer can silently exceed `max_length` after template expansion, causing shape mismatches or truncating the question along with the context.

**Implementation:** `CompressionDataset._truncate_context_for(tokenizer, context, question)` performs truncation independently for each tokenizer. It tokenizes the full templated prompt, computes the excess, and trims characters from the end of the context with a 4-token safety margin.

---

## 5. Supervised EOS Masking (`valid_positions_mask`)

**Why:** Naive pad masking (`generated_ids != pad_token_id`) incorrectly dropped valid end-of-sequence stop tokens when Qwen's `pad_token_id` (`<|endoftext|>`, 151643) matched `eos_token_id`, or included trailing post-EOS tokens when teachers generated past stop sequences. Training the student on post-EOS logits injected arbitrary noise into the KL loss, while dropping the EOS token prevented the student from learning when to stop generation.

**Implementation:** `valid_positions_mask` in `src/losses.py` using inclusive `cumsum`:

```python
is_eos = (generated_ids.unsqueeze(-1) == eos).any(-1)   # [B, G]
eos_seen = is_eos.cumsum(dim=1)                          # [B, G], inclusive
valid = (eos_seen == 0) | (is_eos & (eos_seen == 1))
```

This retains all positions up to and **including** the first EOS token (preserving the crucial "stop" gradient signal) and zeroes out all subsequent post-EOS padding positions.

---

## 6. Projector Bias at Padded Positions (Mamba2)

**Why:** `nn.Linear(bias=True)` produces non-zero output even at zero-input positions — the bias term is always added. After left-padding with zeros and projecting, padded soft prompt positions would hold the projector bias rather than being truly zeroed.

**Implementation:** After projection, padded positions are explicitly zeroed:

```python
soft_prompts = soft_prompts * compressed_mask.unsqueeze(-1)
```

This ensures that padded positions are exactly zero in the soft prompts fed to the LLM, preventing the LLM from attending to spurious non-zero values in the prefix.

---

## 7. `DistributedLengthGroupedSampler`

**Why:** The default `DistributedSampler` distributes samples randomly across ranks. For variable-length datasets like MRQA, random ordering leads to highly variable within-batch padding — a batch containing one 800-token sample and several 50-token samples wastes ~94% of compute on padding.

`DistributedLengthGroupedSampler` groups sequences by length into megabatches before shuffling. This reduces within-batch padding waste while maintaining global shuffle diversity. See [Data Loading](../data/dataloading.md) for the full algorithm.

---

## 8. Mamba2 Sequence Packing without Pad Tokens

**Why:** In Mamba2, sequence processing with standard 2D padded tensors `[B, L]` forces state updates over padding tokens, wasting memory/compute and corrupting SSM hidden states across padded boundaries.

**Implementation:** `collate_fn_mamba2` in `src/dataset.py` flattens all variable-length compressor sequences in a batch into a single contiguous 1D/2D tensor `[1, sum(L_i)]` with sequence segment indicators `seq_idx = [0..0, 1..1, ..., B-1..B-1]`. Zero pad tokens enter the Mamba2 backbone. The backbone uses `seq_idx` to reset the SSM state at sequence boundaries, eliminating padding overhead while maintaining exact numerical equivalence with single-sequence forward passes (rel diff $< 5 \times 10^{-3}$).

---

## 9. KV Cache Injection Precision & RoPE Alignment

**Why:** In KV cache compression mode (`prompt_mode='kv'`), mapping compressor hidden states to LLM Key-Value states requires precise layer-wise projection and rotary position alignment. Naive position indexing or precision mismatches between Mamba (fp32) and LLM (bf16/fp16) cause KV cache logit divergence between parallel training (`_kv_forward`) and stepwise inference decode.

**Implementation:** `build_kv_cache` in `src/utils/kv_cache.py` computes pad-aware position IDs `comp_pos = (compressed_mask.cumsum(-1) - 1).clamp_min(0)` and applies LLM rotary embeddings `rotary_emb(key, position_ids=comp_pos)` before building HF `DynamicCache` instances. Parallel training and stepwise decode match within relative error $< 5 \times 10^{-3}$.

---

## 10. Multi-Gold Evaluation Generator

**Why:** The training dataset generator (`src.dataset.mrqa_generator`) selects only the first detected answer string (`detected_answers[0]['text']`) as a single gold target. Using the training dataset generator for MRQA evaluation leads to severe under-reporting of F1 and EM because official MRQA/SQuAD benchmarks score predictions against **all** valid gold answers per question.

**Implementation:** `src/evaluation/datasets.py` implements `mrqa_eval_generator`, which yields `answers` as the complete list of acceptable gold strings (`qa['answers']`). `SpanMetricScorer` uses `metric_max_over_ground_truths` (`metrics.py`) to compute `max(F1)` and `max(EM)` over all gold answers per sample, strictly matching official MRQA 2019 Shared Task evaluation rules.

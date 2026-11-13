# Knowledge Distillation

## Objective

The compressor is trained to **match the output distribution of the frozen LLM** running on the full uncompressed prompt. We do not train the compressor to produce exact token predictions — we train it to produce soft prompts that, when fed into the frozen LLM, yield a distribution over the next token that is close to the teacher's distribution.

This is **distributional alignment**, not sequence-level supervision. It allows the student to generalize across paraphrases and different phrasings, not just exact surface matches.

---

## Teacher Generation

The teacher is the **same frozen LLM** used for student inference (Qwen2.5). Given the full uncompressed prompt:

```python
generated_ids, teacher_logits = model.generate_teacher(llm_inputs)
# generated_ids:   [B, G]      — tokens the teacher generated
# teacher_logits:  [B, G, V]   — raw logits at each generation step
```

The teacher runs `llm.generate(..., output_scores=True)` to get both the generated tokens and the logits used at each step. These logits are the training targets.

---

## Student Forward

The student takes the compressed soft prompts and the **teacher-generated tokens** (not ground-truth answers) as its input:

```python
combined_embeds = cat([soft_prompts, embed(generated_ids)], dim=1)
# [B, C+G, H_llm]

student_logits = model.generate_student(combined_embeds, attention_mask, compressed_len)
# [B, G, V]
```

The student is trained to predict the same token at each position that the teacher predicted — using the teacher's tokens as context and the teacher's logits as targets.

The indexing `lm_head(hidden[C-1:-1])` shifts by one: the last soft prompt hidden state predicts token `G_0`, the hidden state at position `C + i - 1` predicts token `G_i`. This mirrors the causal language modeling objective.

---

## KL Divergence Loss

```python
kl_loss = nn.KLDivLoss(reduction='batchmean')

student_log_probs  = F.log_softmax(student_logits / T, dim=-1)  # [B, G, V]
teacher_probs      = F.softmax(teacher_logits / T, dim=-1)      # [B, G, V]

per_token_kl = kl_loss(student_log_probs, teacher_probs)        # scalar per position
```

**Temperature scaling:** both distributions are softened by temperature `T` (default 2.0). Higher temperature flattens the distributions, transferring more signal from the teacher's probability mass across non-argmax tokens (dark knowledge).

The loss is scaled by `T²` to compensate for the magnitude reduction caused by temperature softening:

```python
loss = (per_token_kl * valid).sum() / valid.sum() * temperature**2
```

This follows the standard Hinton et al. (2015) KD recipe.

---

## Valid Position Masking

After the teacher's EOS token, `generated_ids` contains padding tokens. The teacher's logits at these positions are arbitrary (the model fills them with pad token outputs). Including them in the loss would inject meaningless gradient signal.

```python
valid = (generated_ids != pad_token_id)   # [B, G], bool
loss  = (per_token_kl * valid).sum() / valid.sum() * temperature**2
```

Only positions up to (and including) EOS are included in the loss. For samples that don't generate EOS within `max_new_tokens`, all positions are valid.

---

## What Is and Is Not Trained

| Component | Trained | Notes |
|---|---|---|
| Mamba backbone | Yes | All parameters receive gradients |
| Linear projector | Yes | `H_mamba → H_llm` |
| Qwen2.5 LLM | No | `requires_grad = False` for all parameters |

Gradient flow: all 220 compressor parameters receive finite non-zero gradients; the LLM has no gradients.

---

## Loss Variants

Two loss functions are implemented in `src/losses.py`, selectable via config.

### `kl` — Token-Uniform KL (default)

```python
loss = kl_loss(student_logits, teacher_logits, generated_ids, temperature, pad_token_id)
```

Every valid (non-pad) position contributes equally to the loss. This is the standard Hinton et al. (2015) KD recipe described above.

---

### `confidence_kl` — Confidence-Weighted KL

```python
loss = confidence_weighted_kl_loss(..., mode="max_prob" | "neg_entropy")
```

Same KL divergence as above, but each position is weighted by how **confident** the teacher is at that step. The intuition: the teacher's distribution is a strong, low-ambiguity signal at factual tokens (dates, names, numbers) and a weak, high-entropy signal at filler tokens ("the", "a", "and"). Weighting by confidence directs gradient toward the positions that actually carry information.

Confidence is computed on the **temperature-softened** teacher distribution — consistent with what the student is trained to match.

#### `mode: "max_prob"`

```python
weights = softmax(teacher_logits / T).max(dim=-1)   # [B, G], range [1/V, 1]
```

Weight at each position = probability mass on the teacher's top token. Simple, numerically stable. A token the teacher predicts with 90% confidence gets 9× the weight of one it predicts with 10%.

#### `mode: "neg_entropy"`

```python
entropy  = -sum(p * log(p))                         # [B, G]
weights  = 1 / (entropy + ε)
```

Weight = inverse Shannon entropy of the teacher's softened distribution. More aggressive than `max_prob`: high-entropy positions (entropy → log V ≈ 13 for V=32k) are strongly suppressed, near-deterministic positions (entropy → 0) dominate. Can amplify noise at positions where the teacher is arbitrarily confident; `max_prob` is the safer default.

#### Normalization

Both modes normalize by the sum of weights (not the count of valid tokens), so the loss scale is comparable to `kl`:

```python
loss = (per_tok_kl * weights).sum() / weights.sum() * T²
```

---

## Configuration

Relevant config keys:

```yaml
temperature: 2.0          # KD temperature
max_new_tokens: 50        # teacher generation length (= number of KL targets)
compression_ratio: 0.1    # stride = 10; C ≈ L/10
loss_type: "kl"           # "kl" | "confidence_kl"
confidence_mode: "max_prob"  # "max_prob" | "neg_entropy" (only used when loss_type = confidence_kl)
```

A higher `max_new_tokens` provides more KL targets per sample but increases memory (teacher must generate and store logits for G tokens, and the student forward processes `[B, C+G, H]` embeddings).

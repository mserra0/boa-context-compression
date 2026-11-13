# Language Model Pretraining Guide

This guide details the continuation Language Model (LM) pretraining pipeline implemented in [`src/pretrain.py`](../../src/pretrain.py) and [`src/pretrain_model.py`](../../src/pretrain_model.py).

---

## 1. Overview & Architecture

### Pretraining Objective
LM pretraining trains the compressor module (Mamba1 or Mamba2) on uncurated corpus text (e.g., FineWeb-EDU) to convert long context prefixes into compressed representations. These compressed representations are injected into a frozen LLM (e.g., Qwen2.5 or Qwen3), enabling the LLM to predict document continuations via standard cross-entropy loss (`lm_ce_loss`).

Unlike task-specific fine-tuning or knowledge distillation, pretraining operates directly on raw document continuations without requiring a teacher forward pass or logit capture.

```
       Uncurated Document Text (FineWeb-EDU)
                   │
           Random Boundary Split
           ┌───────┴───────┐
      Prefix X          Suffix Y
          │                │
     Compressor        Frozen LLM
  (Mamba1 / Mamba2)   (Qwen2.5 / 3)
          │                │
    Soft Prompts / ────────┘
    Soft KV Cache          │
                           ▼
                    Cross-Entropy Loss
                      (lm_ce_loss)
```

### Model Architecture (`LMPretrainer`)
The core pretraining class is [`LMPretrainer`](../../src/pretrain_model.py) in [`src/pretrain_model.py`](../../src/pretrain_model.py). It wraps a frozen Causal LLM decoder and a trainable Mamba compressor.

- **Frozen Decoder Pinning**: The LLM decoder's parameters are frozen (`requires_grad = False`). In [`LMPretrainer.train()`](../../src/pretrain_model.py), `self.llm.eval()` is explicitly called to override default `nn.Module.train()` propagation and prevent dropout from being re-enabled.
- **No Chat Template Wrapping**: Pretraining explicitly avoids chat framing. Because continuation tasks feature raw document text without user/assistant turns, introducing chat templates would create a severe distribution mismatch.

### Soft Prompt Injection Modes
[`LMPretrainer`](../../src/pretrain_model.py) supports two injection modes configured via `prompt_mode`:

#### 1. Soft Embedding Mode (`prompt_mode="embed"`)
- Implemented in [`_embed_forward()`](../../src/pretrain_model.py).
- The compressor projects raw hidden states into soft prompt embeddings $[B, C, H_{\text{llm}}]$.
- Soft prompts are concatenated with target sequence token embeddings $Y$:
  $$\text{combined} = [\text{soft\_prompts} \,;\, \text{embed}(Y)] \quad \in \mathbb{R}^{B \times (C + G) \times H_{\text{llm}}}$$
- Causal attention mask and position IDs handle soft prompt padding and target continuation.
- **Shift Alignment**: The last soft prompt (index $C-1$) predicts continuation token $Y_0$; token $Y_{i-1}$ predicts $Y_i$.

#### 2. Soft Key-Value Cache Mode (`prompt_mode="kv"`)
- Implemented in [`_kv_forward()`](../../src/pretrain_model.py).
- The compressor's raw hidden states are projected layer-by-layer into Key and Value vectors via [`KVProjector`](../../src/kv_projector.py) and seeded into the LLM's past KV cache using [`build_kv_cache`](../../src/utils/kv_cache.py).
- **Learned Continuation Trigger (`kv_trigger`)**: Decoder input consists of a trainable trigger vector $[1, H_{\text{llm}}]$ (warm-started from the LLM BOS embedding) concatenated with target embeddings:
  $$\text{dec\_input} = [\mathbf{v}_{\text{trigger}} \,;\, \text{embed}(Y)] \quad \in \mathbb{R}^{B \times (1 + G) \times H_{\text{llm}}}$$
- The trigger token (index 0) queries the seeded KV cache to predict continuation token $Y_0$.

---

## 2. Data Pipeline & Sequence Packing

### Dataset Preparation (`PretrainDataset`)
Defined in [`src/dataset.py`](../../src/dataset.py), [`PretrainDataset`](../../src/dataset.py) wraps Arrow/Parquet shards of FineWeb-EDU:
- Filters documents below `min_doc_tokens` (default: 320).
- Performs a single columnar read over `token_count` (or character length proxy) for instant filtering and budget computation without per-row Arrow random access overhead.

### Dynamic Boundary Splitting (`_pretrain_xy_pairs`)
Implemented in [`src/utils/dataloading.py`](../../src/utils/dataloading.py), boundary splitting divides each document into prefix $X$ (compressor input) and suffix $Y$ (continuation target):
1. Samples a random character split boundary $c \in [\lfloor N \cdot \text{split\_lo} \rfloor, \lfloor N \cdot \text{split\_hi} \rfloor]$.
2. Character-level splitting decouples compressor and LLM tokenizers, avoiding text decode/re-encode cycles.
3. **Contiguity Preservation**: Takes the **last** `max_ctx` tokens of prefix $X$ (`[-max_ctx:]`). This ensures $X$ ends exactly where $Y$ begins, avoiding lead-in drop gaps.
4. Truncates target suffix $Y$ to `max_pred` tokens (`[:max_pred]`).

### Target Construction (`_build_pretrain_targets`)
Defined in [`src/utils/dataloading.py`](../../src/utils/dataloading.py), [`_build_pretrain_targets()`](../../src/utils/dataloading.py) right-pads target suffixes into fixed target tensors:
- `target_ids`: $[B, G]$ tensor padded with `pad_token_id`.
- `target_mask`: $[B, G]$ binary indicator ($1$ for valid target tokens, $0$ for padding).

### Token Budget Packing & Collate Paths
Sequences are grouped dynamically using [`DistributedTokenBudgetSampler`](../../src/utils/dataloading.py) in [`src/utils/dataloading.py`](../../src/utils/dataloading.py), packing items into batches summing to $\le \text{tokens\_per\_batch}$ (e.g., 4096 or 8192 tokens).

| Backend | Collate Function | Batch Structure | Padding Handling |
| :--- | :--- | :--- | :--- |
| **Mamba2** | [`collate_fn_pretrain`](../../src/utils/dataloading.py) | Packed 1D (`input_ids` $[1, T]$, `seq_idx` $[1, T]$, `lengths` $[B]$) | Padding-free; `seq_idx` resets SSM state at doc boundaries |
| **Mamba1** | [`collate_fn_pretrain_mamba1`](../../src/utils/dataloading.py) | Dense 2D (`input_ids` $[B, I]$, `attention_mask` $[B, I]$) | Left-padded; required for Mamba1 stride anchoring |

---

## 3. Loss Formulations

### Continuation Cross-Entropy Loss (`lm_ce_loss`)
Defined in [`src/losses.py`](../../src/losses.py), `lm_ce_loss` computes the token-mean cross-entropy over target positions $Y$:

$$\mathcal{L}_{\text{CE}} = \frac{\sum_{b=1}^{B} \sum_{g=1}^{G} m_{b,g} \cdot \text{CE}\left(S_{b,g}, \, y_{b,g}\right)}{\sum_{b=1}^{B} \sum_{g=1}^{G} m_{b,g} + \epsilon}$$

Where:
- $S_{b,g} \in \mathbb{R}^V$ is the student logit slice for sequence $b$ at continuation position $g$.
- $y_{b,g}$ is the target token ID from `target_ids`.
- $m_{b,g} \in \{0, 1\}$ is the binary mask from `target_mask`.
- $\epsilon = 10^{-9}$ prevents division by zero if an entire batch is padded.

```python
# src/losses.py
def lm_ce_loss(
    student_logits: torch.Tensor,
    target_ids: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    B, G, V = student_logits.shape
    mask = target_mask.to(student_logits.dtype)
    per_tok = F.cross_entropy(
        student_logits.reshape(-1, V),
        target_ids.reshape(-1),
        reduction="none",
    ).reshape(B, G)
    return (per_tok * mask).sum() / mask.sum().clamp_min(1.0)
```

### Contrast with Knowledge Distillation Losses
While pretraining uses target cross-entropy (`lm_ce_loss`), distillation pipelines ([`docs/training/knowledge-distillation.md`](knowledge-distillation.md)) match the full teacher probability distribution using KL divergence:

- **Distillation KL Loss (`kl_loss`)**: Softmax temperature matching scaled by $T^2$ over teacher-generated tokens.
- **EOS Masking (`valid_positions_mask`)**: Distillation uses EOS-aware masks to include the stopping token while ignoring post-EOS padding tails.

---

## 4. Configuration & Execution

### Configuration Hierarchy
Pretraining configurations reside in [`configs/pretrain/`](../../configs/pretrain):

1. **Base Configuration**: [`configs/pretrain/config_pretrain.yaml`](../../configs/pretrain/config_pretrain.yaml)
   Standard reference setup for local or single-node experiments.
2. **Overfit / Verification Configs**: [`configs/pretrain/overfit/`](../../configs/pretrain/overfit)
   Small sample configs (e.g., `config_pretrain_overfit_kv_mamba2_qwen3.yaml`) used to verify loss convergence on small document subsets.
3. **Cluster-scale configs**: [`configs/pretrain/mn5/`](../../configs/pretrain/mn5)
   Multi-node distributed configs optimized for large GPU clusters (e.g., Qwen3-1.7B and Qwen3-4B backends across 64 GPUs).

#### Example Config (`configs/pretrain/config_pretrain.yaml`)
```yaml
model:
  mamba_model_name: "state-spaces/mamba2-1.3b"
  llm_model_name: "Qwen/Qwen2.5-0.5B-Instruct"
  compressor_type: "mamba2"
  stride: 10
  objective: "lm"

training:
  max_steps: 50000
  warmup_steps: 1000
  batch_size: 8
  learning_rate: 0.0003
  weight_decay: 0.01
  eval_interval: 500
  save_interval: 2000
  log_interval: 20
  gradient_clip_max_norm: 1.0
  gradient_accumulation_steps: 4

dataloading:
  packing: "token_budget"
  tokens_per_batch: 8192
  max_seqs_per_batch: 32

data:
  data_dir: "/path/to/fineweb-edu-10B"
  compressor_tokenizer_name: "EleutherAI/gpt-neox-20b"
  min_doc_tokens: 320
  max_ctx: 1024
  max_pred: 256
  val_samples: 2000
  seed: 42
```

### Execution Driver (`src/pretrain.py`)
Pretraining is driven by [`pretrain()`](../../src/pretrain.py) in [`src/pretrain.py`](../../src/pretrain.py):
- **Step-Based Progress**: Counts global optimizer steps rather than epochs, making a single pass over the corpus.
- **CPU Micro-Batch Token Counting**: Uses [`_batch_token_counts()`](../../src/pretrain.py) to tally context and target tokens on host CPU without introducing GPU synchronization barriers.
- **Evaluation**: Calls [`evaluate_lm()`](../../src/pretrain.py) to calculate cross-entropy loss and perplexity ($\text{PPL} = \exp(\text{CE})$) over validation data.

### Execution Commands

`src/pretrain.py` is a library module, not an entry point. Pretraining is launched through
[`main.py`](../../main.py), which dispatches to
[`_main_pretrain()`](../../main.py) — and from there to
`pretrain()` — whenever the config sets `objective: "lm"`.

#### Single-GPU Execution
```bash
uv run torchrun --standalone --nproc_per_node=1 main.py --config configs/pretrain/config_pretrain.yaml
```

#### Multi-GPU Distributed Data Parallel (DDP)
```bash
torchrun --nproc_per_node=4 main.py \
    --config configs/pretrain/mn5/qwen3_1-7b/pretrain_mamba2_kv_qwen3.yaml
```

# Ablation Benchmark Matrix & Scaling Documentation

This guide documents the ablation configuration matrix, pretrain vs. finetune experimental suite, and model scale sweeps located in [`configs/ablations/`](../../configs/ablations) and [`configs/pretrain/`](../../configs/pretrain).

---

## 1. Overview & Experimental Matrix

The ablation suite systematically evaluates the contribution of each architectural component, training phase, supervision signal and model-scale setting of Boa.

### Controlled Experimental Dimensions

```
                               ┌────────────────────────────────────────┐
                               │           Boa         │
                               └───────────────────┬────────────────────┘
                                                   │
         ┌───────────────────────┬─────────────────┴───────┬───────────────────────┐
         │                       │                         │                       │
┌────────▼─────────┐    ┌────────▼─────────┐      ┌────────▼─────────┐    ┌────────▼─────────┐
│   Compressor     │    │   Prompt Mode    │      │  Training Phase  │    │   Model Scale    │
│   Backbone       │    │                  │      │   & Objective    │    │   & Stride       │
├──────────────────┤    ├──────────────────┤      ├──────────────────┤    ├──────────────────┤
│ • Mamba1 (1.4B)  │    │ • Embed (Input)  │      │ • Pretrain (CE)  │    │ • Decoder: 1.7B/ │
│ • Mamba2 (1.3B)  │    │ • KV Cache (Inj) │      │ • Pretrain (offKD│    │   4B Qwen3       │
│ • Mamba2 (2.7B)  │    │                  │      │ • Finetune (KD)  │    │ • Stride: 4x, 8x,│
│                  │    │                  │      │ • Finetune (SFT) │    │   16x            │
└──────────────────┘    └──────────────────┘      └──────────────────┘    └──────────────────┘
```

1. **Compressor Backbones (`compressor_type`)**:
   - **Mamba1 (`mamba`)**: 1.4B parameter SSM using dense 2D left-padded tensors (`[B, I]`).
   - **Mamba2 (`mamba2`)**: 1.3B or 2.7B parameter SSM using padding-free 1D token-budget sequence packing (`[1, T]` with `seq_idx`).
2. **Prompt Injection Modes (`prompt_mode`)**:
   - **Embedding Injection (`embed`)**: Soft prompts are prepended directly to the LLM's token embedding sequence.
   - **KV Cache Injection (`kv`)**: Soft prompts are projected layer-by-layer into Key-Value states via [`KVProjector`](../../src/kv_projector.py) and injected into the LLM's past KV cache, queried by a learned continuation trigger (`kv_trigger`).
3. **Pretraining Objectives**:
   - **Language Model Continuation CE (`objective: "lm"`)**: Standard cross-entropy loss predicting target document continuations on FineWeb-EDU.
   - **Off-Policy KD Pretrain (`pretrain_loss: "kd"`)**: Distillation loss where a frozen FullFT teacher processes the uncompressed prefix while the student processes the compressed prefix to predict continuation logits.
4. **Fine-Tuning Objectives**:
   - **Knowledge Distillation (`loss_type: "kl"`)**: KL divergence distillation matching full teacher logit distributions over generated responses.
   - **Supervised Fine-Tuning (`objective: "sft"`)**: Cross-entropy training directly on ground-truth answer spans without a teacher forward pass.
5. **Warm-Start Pretraining Contribution**:
   - **With Pretraining**: Warm-started from Stage 1 FineWeb-EDU pretraining checkpoints.
   - **Without Pretraining (`start_from: null`)**: Trained directly on MRQA from randomly initialized compressor and projector weights.
6. **Model Scale & Compression Ratio Sweeps**:
   - **Decoder Scaling**: Frozen FullFT Qwen3-1.7B vs Qwen3-4B decoders.
   - **Encoder Scaling**: Mamba2-1.3B vs Mamba2-2.7B compressors.
   - **Compression Ratios**: 4x (`stride: 4`), 8x (`stride: 8`), and 16x (`stride: 16`).

---

## 2. 8x Compression Ablation Matrix (`configs/ablations/8x/qwen3_1-7b/`)

All 8x compression ratio ablations share a single frozen teacher/decoder: **FullFT Qwen3-1.7B** located at `/path/to/checkpoints/qwen3_1_7b_fullft/final`.

### Stage 1: Pretraining Arm Configurations (`configs/ablations/8x/qwen3_1-7b/pretrain/`)

Pretraining runs on FineWeb-EDU (100BT sample) for 10,000 steps with learning rate $1.0 \times 10^{-4}$.

| Config File | Backbone | Injection | Objective | WandB Name | Key Settings |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [`m1_emb.yaml`](../../configs/ablations/8x/qwen3_1-7b/pretrain/m1_emb.yaml) | Mamba1 1.4B | Embed | LM (CE) | `ABL_8x_pretrain_LM_Mamba1_emb_Qwen3-1.7B` | `compressor_type: mamba`, `prompt_mode: embed`, left-padded `[B, I]` |
| [`m1_kv.yaml`](../../configs/ablations/8x/qwen3_1-7b/pretrain/m1_kv.yaml) | Mamba1 1.4B | KV Cache | LM (CE) | `ABL_8x_pretrain_LM_Mamba1_kv_Qwen3-1.7B` | `compressor_type: mamba`, `prompt_mode: kv`, left-padded `[B, I]` |
| [`m2_emb.yaml`](../../configs/ablations/8x/qwen3_1-7b/pretrain/m2_emb.yaml) | Mamba2 1.3B | Embed | LM (CE) | `ABL_8x_pretrain_LM_Mamba2_emb_Qwen3-1.7B` | `compressor_type: mamba2`, `prompt_mode: embed`, token-budget packed `[1, T]` |
| [`m2_kv_offkd.yaml`](../../configs/ablations/8x/qwen3_1-7b/pretrain/m2_kv_offkd.yaml) | Mamba2 1.3B | KV Cache | Off-Policy KD | `ABL_8x_pretrain_offKD_Mamba2_kv_Qwen3-1.7B` | `compressor_type: mamba2`, `prompt_mode: kv`, `pretrain_loss: kd` |

### Stage 2: Finetuning Arm Configurations (`configs/ablations/8x/qwen3_1-7b/finetune/`)

Fine-tuning runs on MRQA for up to 10,000 epochs (early stopping patience = 4) with learning rate $1.0 \times 10^{-4}$.

| Config File | Backbone | Injection | Objective | Pretrain Warm-Start (`start_from`) | Purpose / Paper Table Mapping |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [`m1_emb_kd.yaml`](../../configs/ablations/8x/qwen3_1-7b/finetune/m1_emb_kd.yaml) | Mamba1 1.4B | Embed | KD (KL) | `pretrain/m1_emb` | Evaluates Mamba1 under soft embedding injection. |
| [`m1_kv_kd.yaml`](../../configs/ablations/8x/qwen3_1-7b/finetune/m1_kv_kd.yaml) | Mamba1 1.4B | KV Cache | KD (KL) | `pretrain/m1_kv` | Evaluates Mamba1 under KV cache injection (Mamba1 vs Mamba2 KV baseline). |
| [`m2_emb_kd.yaml`](../../configs/ablations/8x/qwen3_1-7b/finetune/m2_emb_kd.yaml) | Mamba2 1.3B | Embed | KD (KL) | `pretrain/m2_emb` | **Table 2 Ablation ("w/o KV projection")**: Isolates benefit of KV injection over embedding injection in Mamba2. |
| [`m2_kv_kd_noPretrain.yaml`](../../configs/ablations/8x/qwen3_1-7b/finetune/m2_kv_kd_noPretrain.yaml) | Mamba2 1.3B | KV Cache | KD (KL) | `null` (Random Init) | **Table 2 Ablation ("w/o LM pretraining")**: Measures contribution of FineWeb-EDU Stage 1 pretraining. |
| [`m2_kv_lm.yaml`](../../configs/ablations/8x/qwen3_1-7b/finetune/m2_kv_lm.yaml) | Mamba2 1.3B | KV Cache | SFT (GT-CE) | LM Pretrain Baseline | **Table 1 & Table 2 ("LM pretrain -> LM finetune / w/o KD finetuning")**: Evaluates performance without KD teacher supervision. |
| [`m2_kv_offkd_lm.yaml`](../../configs/ablations/8x/qwen3_1-7b/finetune/m2_kv_offkd_lm.yaml) | Mamba2 1.3B | KV Cache | SFT (GT-CE) | `pretrain/m2_kv_offkd` | **Table 1 ("Off-policy KD pretrain -> LM finetune")**: Tests off-policy KD pretraining combined with SFT finetuning. |

---

## 3. Scale & Compression Ratio Sweeps (`configs/pretrain/mn5/`)

To evaluate scalability across model sizes and compression factors, dedicated pretraining configs are structured under [`configs/pretrain/mn5/`](../../configs/pretrain/mn5).

### 1. Qwen3-1.7B Decoder Suite (`configs/pretrain/mn5/qwen3_1-7b/`)

- **Compression Ratio Sweep (Mamba2 1.3B + Qwen3-1.7B)**:
  - 4x Compression (`stride: 4`): [`pretrain_mamba2_kv_qwen3_4x.yaml`](../../configs/pretrain/mn5/qwen3_1-7b/pretrain_mamba2_kv_qwen3_4x.yaml)
  - 8x Compression (`stride: 8`): [`pretrain_mamba2_kv_qwen3_8x.yaml`](../../configs/pretrain/mn5/qwen3_1-7b/pretrain_mamba2_kv_qwen3_8x.yaml)
  - 16x Compression (`stride: 16`): [`pretrain_mamba2_kv_qwen3_16x.yaml`](../../configs/pretrain/mn5/qwen3_1-7b/pretrain_mamba2_kv_qwen3_16x.yaml)
- **Direct Architecture Comparison**:
  - Mamba1 1.4B KV Pretrain: [`pretrain_mamba1_kv_qwen3.yaml`](../../configs/pretrain/mn5/qwen3_1-7b/pretrain_mamba1_kv_qwen3.yaml)
  - Mamba1 LM CE Baseline: [`pretrain_mamba1_lm.yaml`](../../configs/pretrain/mn5/qwen3_1-7b/pretrain_mamba1_lm.yaml)
  - Mamba2 LM CE Baseline: [`pretrain_mamba2_lm.yaml`](../../configs/pretrain/mn5/qwen3_1-7b/pretrain_mamba2_lm.yaml)

### 2. Qwen3-4B Decoder Suite (`configs/pretrain/mn5/qwen3_4b/`)

Scales the frozen LLM decoder to **Qwen3-4B FullFT** (`/path/to/checkpoints/qwen3_4b_fullft/final`).

- **Decoder Scaling (Mamba2-1.3B Compressor + Qwen3-4B Decoder)**:
  - 4x Ratio (`stride: 4`): [`pretrain_mamba2-1.3b_kv_qwen3-4b_4x.yaml`](../../configs/pretrain/mn5/qwen3_4b/pretrain_mamba2-1.3b_kv_qwen3-4b_4x.yaml)
  - 8x Ratio (`stride: 8`): [`pretrain_mamba2-1.3b_kv_qwen3-4b_8x.yaml`](../../configs/pretrain/mn5/qwen3_4b/pretrain_mamba2-1.3b_kv_qwen3-4b_8x.yaml)
  - 16x Ratio (`stride: 16`): [`pretrain_mamba2-1.3b_kv_qwen3-4b_16x.yaml`](../../configs/pretrain/mn5/qwen3_4b/pretrain_mamba2-1.3b_kv_qwen3-4b_16x.yaml)
- **Encoder + Decoder Scaling (Mamba2-2.7B Compressor + Qwen3-4B Decoder)**:
  - 4x Ratio (`stride: 4`): [`pretrain_mamba2-2.7b_kv_qwen3-4b_4x.yaml`](../../configs/pretrain/mn5/qwen3_4b/pretrain_mamba2-2.7b_kv_qwen3-4b_4x.yaml)
  - 8x Ratio (`stride: 8`): [`pretrain_mamba2-2.7b_kv_qwen3-4b_8x.yaml`](../../configs/pretrain/mn5/qwen3_4b/pretrain_mamba2-2.7b_kv_qwen3-4b_8x.yaml)
  - 16x Ratio (`stride: 16`): [`pretrain_mamba2-2.7b_kv_qwen3-4b_16x.yaml`](../../configs/pretrain/mn5/qwen3_4b/pretrain_mamba2-2.7b_kv_qwen3-4b_16x.yaml)

### Token Control across Scaling Experiments

All scaling configs maintain exact global throughput parity across 32 nodes (128 GPUs):

$$\text{Effective Batch Tokens} = \text{tokens\_per\_batch} (4096) \times \text{grad\_accum} (4) \times \text{world\_size} (128) = 2,097,152 \text{ tokens/step}$$

Hyperparameters (learning rate $1.0 \times 10^{-4}$, warmup steps 50, optimizer weight decay 0.01, gradient clipping 1.0) are strictly held constant across the entire scaling matrix.

---

## 4. Branch Dependencies & Warm-Start Handoff Protocol

### Branch Distribution Requirements

Some objectives are implemented on dedicated branches, so each ablation arm runs on the branch listed below:

| Branch | Target ablation arms | Feature support |
|---|---|---|
| `main` | Mamba1 arms, Mamba2 LM pretrain, `m2_emb_kd`, `m2_kv_kd_noPretrain` | standard KD and LM loss |
| `ablation/kd-pretrain` | `m2_kv_offkd` | `pretrain_loss: kd` |
| `ablation/sft-finetune` | `m2_kv_lm`, `m2_kv_offkd_lm` | `objective: sft` |

### Warm-Start Handoff Protocol

Stage 2 finetuning configs contain the placeholder `<pretrain_run_dir>` in their `start_from` path:

```yaml
# Example from configs/ablations/8x/qwen3_1-7b/finetune/m1_emb_kd.yaml
training:
  start_from: "/path/to/checkpoints/<pretrain_run_dir>/best_model.pt"
```

Once Stage 1 pretraining finishes, replace `<pretrain_run_dir>` with the timestamped run directory it wrote (`<wandb.name>_<start_time>`):

$$\text{Checkpoint Path} = \{\text{checkpoint\_dir}\} / \{\text{pretrain\_wandb\_name\_timestamp}\} / \text{best\_model.pt}$$

---

## 5. Execution & Cluster Launch Commands

### 1. Local Single-Node Execution

Use [`scripts/run_training.sh`](../../scripts/run_training.sh) for local or single-node DDP runs:

```bash
# Local training (1 GPU by default; CUDA_VISIBLE_DEVICES=0,1 runs 2-GPU DDP)
bash scripts/run_training.sh configs/ablations/8x/qwen3_1-7b/pretrain/m2_emb.yaml

# 4 GPUs on one node
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_training.sh configs/ablations/8x/qwen3_1-7b/finetune/m2_emb_kd.yaml
```

### 2. Multi-Node Cluster Submissions

Submit multi-node Slurm jobs (32 nodes x 4 GPUs = 128 GPUs; example scripts, see [../cluster.md](../cluster.md)) using the corresponding architecture Slurm launch scripts:

```bash
# Submit Mamba2 multi-node pretraining job
sbatch scripts/utils/mn5_train_mamba2.slurm \
    configs/ablations/8x/qwen3_1-7b/pretrain/m2_emb.yaml

# Submit Mamba1 multi-node pretraining job
sbatch scripts/utils/mn5_train_mamba.slurm \
    configs/ablations/8x/qwen3_1-7b/pretrain/m1_kv.yaml

# Submit decoder scaling pretraining job (Qwen3-4B + Mamba2-2.7B)
sbatch scripts/utils/mn5_train_mamba2.slurm \
    configs/pretrain/mn5/qwen3_4b/pretrain_mamba2-2.7b_kv_qwen3-4b_8x.yaml
```

### 3. Evaluation Benchmark Launch

Evaluate completed ablation checkpoints with the MRQA evaluation suite (`src/evaluation`) using [`scripts/eval/eval_mrqa.sh`](../../scripts/eval/eval_mrqa.sh):

```bash
CKPT_ROOT=/path/to/checkpoints bash scripts/eval/eval_mrqa.sh <run_dir>   # run directory containing best_model.pt and config.yaml
```

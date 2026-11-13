# 8x Compression Ablation — Run Guide

All configs live under `8x/qwen3_1-7b/`, split into `pretrain/` (stage 1, LM/off-policy-KD
pretraining on FineWeb-EDU) and `finetune/` (stage 2, MRQA). Every arm shares ONE frozen
teacher/LLM: FullFT Qwen3-1.7B. Fine-tuning hyperparameters match the 8x main run (`configs/train/mn5/qwen3_1-7b/mambaL2_pathB_kv_8x_from_pretrain.yaml`)
except `patience` (4 here, 6 there).

**No single branch runs all arms.** Launch each config on the branch listed below.

## Pretrain (stage 1)

| Config | Arm | Branch | Notes |
|---|---|---|---|
| `pretrain/m1_emb.yaml` | Mamba-1 · embed · LM | `main` | |
| `pretrain/m1_kv.yaml` | Mamba-1 · KV · LM | `main` | |
| `pretrain/m2_emb.yaml` | Mamba-2 · embed · LM | `main` | |
| `pretrain/m2_kv_offkd.yaml` | Mamba-2 · KV · off-policy KD | `ablation/kd-pretrain` | Only branch with `pretrain_loss: ce\|kd`. |

> The method run (Mamba-2 · KV · LM pretrain, "Boa") is the main experiment and is not included here.

## Finetune (stage 2, MRQA)

| Config | Arm | Objective | Branch | `start_from` |
|---|---|---|---|---|
| `finetune/m1_emb_kd.yaml` | Mamba-1 · embed | KD | `main` | pretrain/m1_emb |
| `finetune/m1_kv_kd.yaml` | Mamba-1 · KV | KD | `main` | pretrain/m1_kv |
| `finetune/m2_emb_kd.yaml` | Mamba-2 · embed | KD | `main` | pretrain/m2_emb |
| `finetune/m2_kv_kd_noPretrain.yaml` | Mamba-2 · KV, w/o LM pretrain | KD | `main` | `null` (random init) |
| `finetune/m2_kv_lm.yaml` | Mamba-2 · KV | SFT (`objective: sft`) | `ablation/sft-finetune` | completed 8x LM pretrain |
| `finetune/m2_kv_offkd_lm.yaml` | Mamba-2 · KV | SFT (`objective: sft`) | `ablation/sft-finetune` | pretrain/m2_kv_offkd |

> SFT finetune only loads the checkpoint — it does **not** need the KD-pretrain loss code, so
> the off-policy-KD → SFT arm pretrains on `ablation/kd-pretrain` but finetunes on `ablation/sft-finetune`.

## Warm-start handoff

Finetune configs with a `<pretrain_run_dir>` placeholder in `start_from` must be edited after their
pretrain finishes:

```
{checkpoint_dir}/{pretrain_wandb_name}_{start_time}/best_model.pt
```

`start_time` is the runtime timestamp of the pretrain job (the directory name under `checkpoint_dir`).

## Evaluation

Evaluate every arm with the MRQA evaluation suite in `src/evaluation` on `main` (official MRQA
F1/EM, in-domain + OOD).

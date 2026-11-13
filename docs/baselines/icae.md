# Baseline: ICAE

This branch adds an In-Context Autoencoder (ICAE) compressor to the Boa training framework: a LoRA-adapted encoder with learned memory slots (`ICAECompressor` in `src/model.py`) that conditions the same frozen LLM as the Mamba compressor. It is a reimplementation of the method (see [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md)) that reuses the data, loss and evaluation code of `main`. Paths such as `/path/to/...` and `<your_account>` in configs and scripts are placeholders; replace them with your own (the full list is in `docs/setup.md` on `main`).

Install with `uv sync` (prerequisites: `docs/setup.md` on `main`); the quick-start data layout is described in `docs/data/datasets.md` on `main`.

## Configs

Small-scale configs (Qwen2.5-0.5B-Instruct, `n_mem` 52, LoRA rank 128) in `configs/train/`:

| Config | Stage |
|---|---|
| `config_ICAE_ae.yaml`, `config_ICAE_ae_short.yaml` | autoencoding (AE) pretraining |
| `config_ICAE_lm.yaml` | continuation (LM) pretraining |
| `config_ICAE_ae_lm.yaml` (also `configs/pretrain/config_ICAE_pretrain_ae+lm.yaml`) | combined AE + LM pretraining, `L = 0.5 L_AE + 0.5 L_LM` |
| `config_ICAE_ftgt.yaml` | fine-tuning on gold answers (cross-entropy) |
| `config_ICAE_ftdist.yaml` | fine-tuning by KL distillation against the teacher |

Large-scale configs: `configs/pretrain/mn5/pretrain_icae_ae_lm_{4,8,16,32}x.yaml` (FineWeb-Edu pretraining, Qwen3-1.7B teacher) and `configs/train/mn5/icae_finetune_gt_{4,8,16}x.yaml` / `..._bctx.yaml` (MRQA fine-tuning with gold answers; the `_bctx` variants compress the context only and feed the question to the decoder as literal tokens). Fine-tuning configs warm-start from the pretraining run through `start_from: /path/to/checkpoints/<pretrain_run_dir>/best_model.pt`.

## Running

```bash
# single GPU: --gpu, else CUDA_VISIBLE_DEVICES, else GPU 0
scripts/run_training_icae.sh --gpu 0 configs/train/config_ICAE_ae.yaml

# multi-GPU DDP
CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_training.sh configs/train/config_ICAE_ftgt.yaml

# multi-node (example Slurm scripts, see docs/cluster.md on main)
sbatch scripts/utils/mn5_train_icae.slurm configs/pretrain/mn5/pretrain_icae_ae_lm_8x.yaml
```

## Evaluation

```bash
scripts/eval/eval_mrqa_icae.sh --gpu 0 /path/to/checkpoints/<icae_run_dir>
```

This runs the official MRQA span F1/EM evaluation (`src/evaluation`, config `configs/eval/config_eval_MRQA.yaml`); `<icae_run_dir>` must contain `best_model.pt` and `config.yaml`. The LLM-as-judge path (`configs/eval/config_eval_icae.yaml`, `scripts/eval/vllm_judge.sh`) is also available.

## Citation

If you use this code, please cite the Boa paper ([arXiv:2610.33006](https://arxiv.org/abs/2610.33006)):

```bibtex
@misc{serraortega2026lineartime,
  title={Linear-Time Context Compression with State-Space Encoding},
  author={Marc Serra Ortega and Andrey Barsky and Carlos Boned Riera and Tom{\'a}s Ockier and Dimosthenis Karatzas},
  year={2026},
  eprint={2610.33006},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2610.33006}
}
```

ICAE (upstream method):

```bibtex
@inproceedings{ge2024incontext,
  title={In-context Autoencoder for Context Compression in a Large Language Model},
  author={Tao Ge and Jing Hu and Lei Wang and Xun Wang and Si-Qing Chen and Furu Wei},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024},
  url={https://arxiv.org/abs/2307.06945}
}
```

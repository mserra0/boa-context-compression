# Ablation: off-policy KD pretraining

This branch adds the off-policy knowledge-distillation pretraining objective: during LM pretraining on FineWeb-Edu the compressor is trained to match the soft distribution of the frozen teacher (which sees the plaintext prefix) instead of the hard corpus tokens (`model.pretrain_loss: kd`, versus `ce`). The method and its loss are described in [../training/kd-pretraining.md](../training/kd-pretraining.md). Paths such as `/path/to/...` and `<your_account>` in configs and scripts are placeholders; replace them with your own (the full list is in `docs/setup.md` on `main`).

The model section accepts `stride: S` (equivalent to `compression_ratio: 1/S`, as on `main`) as well as the older `compression_ratio`; giving both with different strides is an error.

## Configs

| Config | Purpose |
|---|---|
| `configs/ablations/8x/qwen3_1-7b/pretrain/m2_kv_offkd.yaml` | the 8x KD-pretraining arm (Mamba2, KV, Qwen3-1.7B teacher) |
| `configs/pretrain/mn5/pretrain_mamba2_kd.yaml`, `pretrain_mamba2_kd_kv_qwen3.yaml`, `pretrain_mamba1_kd_kv_qwen3.yaml` | full-scale KD pretraining |
| `configs/pretrain/overfit/config_pretrain_overfit_kd_kv_mamba{1,2}_mn5.yaml` | 200-step overfitting smoke tests |

The fine-tuning stage of the arm (`finetune/m2_kv_offkd_lm.yaml`) is on `ablation/sft-finetune`, because it only loads the pretrained checkpoint.

## Running

Set `data.data_dir` (FineWeb-Edu) and the teacher path (`llm_model_name`, see `docs/training/teacher.md` on `main`), then:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_training.sh configs/ablations/8x/qwen3_1-7b/pretrain/m2_kv_offkd.yaml
# multi-node: sbatch scripts/utils/mn5_train_mamba2.slurm <config>   (see docs/cluster.md on main)
```

The checkpoint is written to `<checkpoint_dir>/<wandb.name>_<timestamp>/best_model.pt`.

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

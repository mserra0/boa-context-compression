# Baseline: Qwen3 full fine-tuning (frozen LLM / teacher)

This branch trains the fully fine-tuned Qwen3 models that serve as the **frozen LLM and KD teacher** for the compressor configs on `main`, and as the uncompressed upper-bound baseline. Training uses the Hugging Face `Trainer` (DDP via `torchrun`) on the six in-domain MRQA training sets. Paths such as `/path/to/...` and `<your_account>` in configs and scripts are placeholders; replace them with your own (the full list is in `docs/setup.md` on `main`).

Install with `uv sync` (prerequisites: `docs/setup.md` on `main`); download MRQA as described in `docs/data/datasets.md` on `main`.

## Train the teacher

```bash
# edit data.data_dir, data.dev_dir, training.output_dir and paths.final_checkpoint_dir first

# Qwen3-1.7B, single node (per-device batch 16, gradient accumulation 4, bf16 with gradient checkpointing)
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_baseline.sh configs/baselines/qwen3_1_7b_fullft.yaml

# multi-node configs (8 nodes x 4 GPUs = 32 GPUs, global batch 512)
sbatch scripts/utils/mn5_baseline_fullft.slurm configs/baselines/mn5_qwen3_1_7b_fullft.yaml
sbatch scripts/utils/mn5_baseline_fullft.slurm configs/baselines/mn5_qwen3_4b_fullft.yaml
```

`run_baseline.sh <config> [train|dump]` launches `src.baselines.train` (default) or `src.baselines.dump_logits` under `torchrun` on the GPUs in `CUDA_VISIBLE_DEVICES`. The best checkpoint by `eval_loss` is saved to `paths.final_checkpoint_dir`, e.g. `/path/to/checkpoints/qwen3_1_7b_fullft/final`, which is the path the configs on `main` use for `llm_model_name`. The `mn5_*` configs are tuned for the multi-node Slurm script (8 nodes x 4 GPUs, global batch 512 = 8 per GPU x 2 accumulation steps for 4B, 16 x 1 for 1.7B; the Qwen3-4B config has no single-node variant, so on fewer GPUs adjust `per_device_train_batch_size` and `gradient_accumulation_steps` to keep the global batch). See `docs/cluster.md` on `main` for the Slurm scripts. `configs/baselines/overfit_qwen3_1_7b_fullft.yaml` is a small early-stopping demonstration.

The `dump` mode writes top-k teacher logits for offline distillation (`dump.output_dir`, consumed by `configs/train/offline_mamba2.yaml`). `src/baselines/coverage_sweep.py` with `configs/baselines/coverage_sweep.yaml` measures how much probability mass the top-K tokens hold on the MRQA dev sets, to choose K for the dump.

## Evaluate

See [../evaluation/baselines.md](../evaluation/baselines.md) for the MRQA evaluation of plain Hugging Face models (zero-shot base vs fine-tuned).

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

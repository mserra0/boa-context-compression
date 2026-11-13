# Frozen LLM / teacher checkpoints

The configs use a **fully fine-tuned Qwen3 model** as the frozen LLM and KD teacher:

- `/path/to/checkpoints/qwen3_1_7b_fullft/final` for the Qwen3-1.7B runs
- `/path/to/checkpoints/qwen3_4b_fullft/final` for the Qwen3-4B runs

These checkpoints are produced by full fine-tuning the Hub models `Qwen/Qwen3-1.7B` / `Qwen/Qwen3-4B` on MRQA with the training code of the `baseline/qwen3-fullft` branch (see `docs/baselines/qwen3-fullft.md` on that branch):

```bash
git checkout baseline/qwen3-fullft
uv sync
bash scripts/run_baseline.sh configs/baselines/qwen3_1_7b_fullft.yaml        # 1.7B teacher, single node
sbatch scripts/utils/mn5_baseline_fullft.slurm configs/baselines/mn5_qwen3_4b_fullft.yaml   # 4B teacher, multi-node
```

Set `data.data_dir`, `paths` and the output directory in the config first (placeholders: [../setup.md](../setup.md)); the final model is written to `paths.final_checkpoint_dir`. The 1.7B config `qwen3_1_7b_fullft.yaml` runs on a single node. The `mn5_*` configs (including the only 4B config) are tuned for the multi-node Slurm script `scripts/utils/mn5_baseline_fullft.slurm` on `baseline/qwen3-fullft` (8 nodes x 4 GPUs, global batch 512; see [../cluster.md](../cluster.md)); on fewer GPUs, adjust `per_device_train_batch_size` and `gradient_accumulation_steps` to keep the global batch.

Configs on `main` then refer to the result through `llm_model_name: "/path/to/checkpoints/qwen3_1_7b_fullft/final"`. The Quick Start does not need a teacher: it uses the public `Qwen/Qwen2.5-0.5B-Instruct` model directly.

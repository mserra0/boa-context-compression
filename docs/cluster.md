# Example Slurm scripts

`scripts/utils/` contains example Slurm scripts for multi-node DDP training on a cluster with 4 GPUs per node, together with the matching multi-node configs under `configs/*/mn5/` (the `mn5` in these names is historical and refers to the cluster they were written for). They are starting points: adapt them to your cluster before use.

| Script | Purpose |
|---|---|
| `mn5_train_mamba2.slurm`, `mn5_train_mamba.slurm` | multi-node training / pretraining with a Mamba2 / Mamba1 compressor |
| `train_mn5.sh` | single-node job with the debug QoS |
| `snapshot_submit.sh` | copies the current commit to a snapshot directory and submits a Slurm script from there |
| `hf_download.sh` | pre-downloads Hugging Face models and datasets for offline compute nodes |

Usage (the scripts write job logs to `logs/`, which must exist before submission):

```bash
mkdir -p logs
sbatch scripts/utils/mn5_train_mamba2.slurm configs/pretrain/mn5/qwen3_1-7b/pretrain_mamba2_kv_qwen3.yaml
bash scripts/utils/snapshot_submit.sh scripts/utils/mn5_train_mamba2.slurm <config.yaml>
```

## What to change

- `#SBATCH --account`, `--partition`, `--qos` (placeholders `<your_account>`, `<your_partition>`, `<your_qos>`, `<your_debug_qos>`), `--nodes`, `--gpus-per-node`, `--cpus-per-task`, `--time`.
- `source /path/to/venv/bin/activate`: the Python environment (see [setup.md](setup.md)).
- `cd /path/to/boa-context-compression/` in `train_mn5.sh`, `EXP_BASE_DIR=/path/to/experiments` in `snapshot_submit.sh`.
- `TRITON_CACHE_DIR` (`/path/to/triton_cache`) and `HF_HOME`: put them on a fast filesystem that is writable from the compute nodes.
- The scripts run in offline mode (`HF_HUB_OFFLINE=1`, `WANDB_MODE=offline`); download models and datasets beforehand, or remove those lines if the compute nodes have internet access.
- `--signal=SIGINT@300` makes `src/train.py` save an `interrupted` checkpoint shortly before the time limit; resubmit with `training.resume_from` set to continue.

The multi-node configs set `checkpoint_dir`, `data_dir` and `start_from` to the placeholders listed in [setup.md](setup.md); replace them with your own paths.

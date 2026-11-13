# Setup

## Prerequisites

- Linux with an NVIDIA GPU and a driver that supports CUDA 13.0 (the locked PyTorch 2.11 is installed from PyPI; its default Linux wheel is built for CUDA 13.0).
- Python 3.11 or newer (`.python-version` pins 3.11) and [uv](https://github.com/astral-sh/uv).
- The **CUDA toolkit (`nvcc`)** and a C++ compiler. `mamba-ssm` and `causal-conv1d` ship as source distributions and are compiled during installation; `nvcc` must match the CUDA version of the installed PyTorch and be on `PATH` (or `CUDA_HOME` must point to the toolkit). The build can take several minutes. If compilation is not an option, install prebuilt wheels of both packages that match your PyTorch/CUDA/Python combination from their upstream release pages.

## Install

```bash
uv sync
```

## Weights & Biases

Training always initialises Weights & Biases. To run without an account or network access:

```bash
export WANDB_MODE=disabled    # or: offline
```

Otherwise log in with `wandb login` and set `wandb.entity` / `wandb.project` in the config (`entity: null` uses your default entity).

## Hugging Face models

Configs refer to models by Hub id (for example `state-spaces/mamba2-1.3b`, `Qwen/Qwen2.5-0.5B-Instruct`); they are downloaded on first use into the Hugging Face cache (`HF_HOME`). On machines without internet access, download them beforehand (see `scripts/utils/hf_download.sh`) and set `HF_HUB_OFFLINE=1`. The exact revisions used for the experiments are listed in [pinned-revisions.md](pinned-revisions.md).

## Data

MRQA and FineWeb-Edu: see [data/datasets.md](data/datasets.md).

## Placeholders used in configs and scripts

| Placeholder | Meaning |
|---|---|
| `/path/to/data/MRQA` | MRQA root containing `train/`, `dev/` and (for evaluation) `dev_ood/` |
| `/path/to/data/fineweb-edu-10B/sample/{10BT,100BT}` | FineWeb-Edu parquet shards used for LM pretraining |
| `/path/to/checkpoints` | checkpoint root; training writes `<root>/<wandb.name>_<timestamp>/` (`best_model.pt`, `config.yaml`) |
| `/path/to/checkpoints/qwen3_1_7b_fullft/final`, `.../qwen3_4b_fullft/final` | fine-tuned Qwen3 frozen LLM / teacher, see [training/teacher.md](training/teacher.md) |
| `<pretrain_run_dir>` | run directory written by the matching pretraining config (`start_from: /path/to/checkpoints/<pretrain_run_dir>/best_model.pt`) |
| `<run_dir>` | any other training run directory under the checkpoint root |
| `/path/to/outputs`, `/path/to/logits` | output / teacher-logit dump directories |
| `/path/to/hf_cache` | Hugging Face cache directory (`HF_HOME`) |
| `/path/to/triton_cache` | Triton kernel cache directory (`TRITON_CACHE_DIR`) |
| `/path/to/venv` | Python environment activated by the Slurm scripts |
| `/path/to/boa-context-compression` | checkout of this repository used by the cluster scripts |
| `/path/to/experiments` | snapshot directory used by `scripts/utils/snapshot_submit.sh` |
| `/path/to/env_variables.sh` | optional file defining `CACHE_ROOT` and other variables (see below) |
| `<your_account>`, `<your_partition>`, `<your_qos>`, `<your_debug_qos>` | Slurm account, partition and QoS names, see [cluster.md](cluster.md) |
| `<your_wandb_entity>` | your Weights & Biases entity (or `null`) |

## Environment variables

| Variable | Used by | Purpose |
|---|---|---|
| `WANDB_MODE` | all training | `disabled`, `offline` or `online` |
| `HF_HOME`, `HF_HUB_OFFLINE`, `HF_DATASETS_OFFLINE` | all | Hugging Face cache location and offline mode |
| `CUDA_VISIBLE_DEVICES` | `scripts/run_training.sh`, `scripts/run_inference.sh` | GPUs to use (default `0`); the number of listed GPUs sets `--nproc_per_node` |
| `ENV_FILE` | `scripts/run_training.sh` | optional file that is `source`d before launching; may define `CACHE_ROOT` |
| `CACHE_ROOT` | `scripts/run_training.sh` | if set, `TRITON_CACHE_DIR` defaults to `$CACHE_ROOT/triton` |
| `NCCL_P2P_DISABLE` | `scripts/run_training.sh` | defaults to `1`; set to `0` on machines with working GPU peer-to-peer |
| `CKPT_ROOT`, `GPU`, `DEVICE`, `CONFIG`, `WANDB` | `scripts/eval/eval_mrqa.sh` | see the header of the script |

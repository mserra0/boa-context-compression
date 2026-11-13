#!/bin/bash
#SBATCH --job-name=boa_sac_sft
#SBATCH --account=<your_account>
#SBATCH --partition=<your_partition>
#SBATCH --qos=<your_qos>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=80
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

module purge
SAC_ENV="${SAC_ENV:-/path/to/venv}"
source "$SAC_ENV/bin/activate"

PROJ_PATH=$SLURM_SUBMIT_DIR
cd "$PROJ_PATH"
mkdir -p logs

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-boa}"
export TOKENIZERS_PARALLELISM=false

CONFIG_PATH="${1:-configs/train/sac/config_sac_qwen3_17b_fullft_fineweb_mrqa_r8.json}"
WORK_DIR="${2:-/path/to/checkpoints/sac/qwen3-17b-fullft/fineweb-mrqa-r8}"
PORT="${3:-14528}"

bash scripts/run_sac_finetune.sh "$CONFIG_PATH" "$WORK_DIR" "$PORT"

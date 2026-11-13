#!/bin/bash
#SBATCH --job-name=boa_sac_eval_r8
#SBATCH --account=<your_account>
#SBATCH --partition=<your_partition>
#SBATCH --qos=<your_qos>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=20
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
export WANDB_NAME="${WANDB_NAME:-sac-r8-mrqa-eval}"
export TOKENIZERS_PARALLELISM=false

CONFIG_PATH="${1:-configs/eval/config_sac_r8_mrqa.yaml}"
DEVICE="${2:-cuda}"

bash scripts/run_sac_eval_protocol.sh "$CONFIG_PATH" "$DEVICE"

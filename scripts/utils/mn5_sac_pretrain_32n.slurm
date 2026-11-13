#!/bin/bash
#SBATCH --job-name=boa_sac_pretrain_32n
#SBATCH --account=<your_account>
#SBATCH --partition=<your_partition>
#SBATCH --qos=<your_qos>
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=80
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --signal=SIGINT@300

# Multi-node SAC pretraining (32 nodes x 4 GPU = 128 ranks).
# This uses torchrun directly; SAC/pretrain/pre_trainer.py has a torchrun path
# that uses global rank/world size for DDP and online FineWeb sharding.

set -ex

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

export TRITON_CACHE_DIR="/path/to/triton_cache"
mkdir -p "$TRITON_CACHE_DIR"

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export NCCL_IB_TIMEOUT=22

NODES=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
export MASTER_ADDR=${NODES[0]}
export MASTER_PORT=${MASTER_PORT:-29500}
export NUM_TRAINERS=4

DEFAULT_CONFIG="configs/pretrain/mn5/pretrain_sac_qwen3_17b_fullft_fineweb_edu_10b_r8_lm_32n.json"
CONFIG_PATH="${1:-$DEFAULT_CONFIG}"
WORK_DIR="${2:-/path/to/checkpoints/sac/r8}"
export CONFIG_PATH="$PROJ_PATH/$CONFIG_PATH"
export WORK_DIR

mkdir -p "$WORK_DIR/output"
cp "$CONFIG_PATH" "$WORK_DIR/config.json"

python - <<'PY'
import json, os
cfg = json.load(open(os.environ['CONFIG_PATH']))
expected = int(os.environ.get('SLURM_JOB_NUM_NODES', '1')) * int(os.environ.get('NUM_TRAINERS', '4'))
actual = int(cfg['pretrain_training_config']['device_count'])
if actual != expected:
    raise SystemExit(f"Config device_count={actual} but Slurm world_size will be {expected}")
print(f"[INFO] SAC config OK: device_count={actual}, token_budget={cfg['pretrain_training_config'].get('token_budget'):,}")
PY

echo "[INFO] Starting SAC DDP pretraining on $SLURM_JOB_NUM_NODES nodes"
echo "[INFO] Master rendezvous node: $MASTER_ADDR:$MASTER_PORT"
echo "[INFO] Config: $CONFIG_PATH"
echo "[INFO] Work dir: $WORK_DIR"
echo "[INFO] Env: $SAC_ENV"

srun bash -c 'torchrun     --nnodes=$SLURM_JOB_NUM_NODES     --nproc_per_node=$NUM_TRAINERS     --node_rank=$SLURM_NODEID     --rdzv_id=$SLURM_JOB_ID     --rdzv_backend=c10d     --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT     SAC/pretrain/pre_trainer.py --work_dir "$WORK_DIR" --port "$MASTER_PORT"'

echo "[SUCCESS] Multi-node SAC pretraining terminated cleanly."

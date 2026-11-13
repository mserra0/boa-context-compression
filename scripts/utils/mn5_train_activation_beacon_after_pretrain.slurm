#!/bin/bash
#SBATCH --job-name=boa_beacon_train
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

# Multi-node Activation Beacon MRQA fine-tuning after FineWeb-Edu pretraining.
# Uses torchrun directly, like the pretraining launchers. Example script: adapt the
# account/partition/QoS and BEACON_ENV to your cluster, see docs/cluster.md on `main`.

set -ex

echo "[INFO] Initializing Multi-Node Beacon Fine-Tuning Job: $SLURM_JOB_ID"

module purge

BEACON_ENV="${BEACON_ENV:-/path/to/venv}"
source "$BEACON_ENV/bin/activate"

PROJ_PATH=$SLURM_SUBMIT_DIR
cd $PROJ_PATH

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export WANDB_MODE="offline"
export WANDB_PROJECT="boa"

export TRITON_CACHE_DIR="/path/to/triton_cache"
mkdir -p "$TRITON_CACHE_DIR" logs

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export NCCL_IB_TIMEOUT=22

NODES=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
export MASTER_ADDR=${NODES[0]}
export MASTER_PORT=29500

DEFAULT_CONFIG="configs/train/config_activation_beacon_qwen3_17b_fullft_after_fineweb_edu_10b_mrqa_official.yaml"
CONFIG_PATH="${1:-$DEFAULT_CONFIG}"
export CONFIG_PATH="$PROJ_PATH/$CONFIG_PATH"
export NUM_TRAINERS=4

echo "[INFO] Starting Beacon DDP fine-tuning on $SLURM_JOB_NUM_NODES nodes..."
echo "[INFO] Master rendezvous node is: $MASTER_ADDR"
echo "[INFO] Config: $CONFIG_PATH"
echo "[INFO] Env: $BEACON_ENV"

srun bash -c 'torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=$NUM_TRAINERS \
    --node_rank=$SLURM_NODEID \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    src/train_activation_beacon.py --config $CONFIG_PATH'

echo "[SUCCESS] Multi-node Beacon fine-tuning pipeline terminated cleanly."

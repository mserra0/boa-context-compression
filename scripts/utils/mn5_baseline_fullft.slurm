#!/bin/bash
#SBATCH --job-name=boa_baseline_fullft
#SBATCH --account=<your_account>
#SBATCH --partition=<your_partition>
#SBATCH --qos=<your_qos>
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=80
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --signal=SIGINT@300

# Abort on error (-e) and trace commands (-x) for post-mortem diagnosis
set -ex

echo "[INFO] Initializing Multi-Node Job (Qwen3 FullFT baseline): $SLURM_JOB_ID"

# 1. Purge defaults
module purge

# 2. Activate the Python environment (replace with your own venv)
source /path/to/venv/bin/activate

# 3. Project Navigation (Slurm snapshot dir)
PROJ_PATH=$SLURM_SUBMIT_DIR
cd $PROJ_PATH

# 4. Offline mode (compute nodes without internet access)
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export WANDB_MODE="offline"
export WANDB_PROJECT="boa"

# Keep the Triton cache on a path that avoids shared-filesystem metadata contention.
export TRITON_CACHE_DIR="/path/to/triton_cache"
mkdir -p $TRITON_CACHE_DIR

# NCCL tuning for multi-node InfiniBand.
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export NCCL_IB_TIMEOUT=22

# 5. Distributed Networking Setup
NODES=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
export MASTER_ADDR=${NODES[0]}
export MASTER_PORT=29500

CONFIG_PATH="${1:?[ERROR] Config path is required. Usage: sbatch mn5_baseline_fullft.slurm <path/to/config.yaml>}"
export CONFIG_PATH="$PROJ_PATH/$CONFIG_PATH"
export NUM_TRAINERS=4

echo "[INFO] Starting Qwen3 FullFT (HF Trainer DDP) on $SLURM_JOB_NUM_NODES nodes x 4 GPU = 32 GPUs, GBS 512..."
echo "[INFO] Master rendezvous node is: $MASTER_ADDR"

# 6. Execution via srun
# bash -c '...' forces dynamic evaluation of SLURM vars on each compute node.
# Entrypoint is the baseline module (-m src.baselines.train), not main.py;
# HF Trainer reads WORLD_SIZE/RANK/LOCAL_RANK from the torchrun env for DDP.
srun bash -c 'torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=$NUM_TRAINERS \
    --node_rank=$SLURM_NODEID \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    -m src.baselines.train --config $CONFIG_PATH'

echo "[SUCCESS] Multi-node Qwen3 FullFT baseline pipeline terminated cleanly."

#!/bin/bash
#SBATCH --job-name=boa_train_mamba
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

# Abort on error (-e) and trace commands (-x) for post-mortem diagnosis
set -ex

echo "[INFO] Initializing Multi-Node Job: $SLURM_JOB_ID"

# 1. Purge defaults
module purge
# Do not source ~/.bashrc in headless Slurm jobs.

# 2. Activate the Python environment (replace with your own venv)
source /path/to/venv/bin/activate

# 3. Project Navigation
# SLURM_SUBMIT_DIR is the directory the job was submitted from (the snapshot directory).
PROJ_PATH=$SLURM_SUBMIT_DIR
cd $PROJ_PATH

# 4. Offline mode (compute nodes without internet access)
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export WANDB_MODE="offline"
export WANDB_PROJECT="boa"

# Multi-node: keep the Triton cache on a path that avoids shared-filesystem metadata contention.
export TRITON_CACHE_DIR="/path/to/triton_cache"
mkdir -p $TRITON_CACHE_DIR

# NCCL tuning for multi-node InfiniBand.
# Async error handling: surfaces transient IB link failures as Python exceptions
# instead of letting NCCL silently deadlock for 10 minutes before SIGABRT.
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# Surface NCCL warnings in the job log for post-mortem diagnosis.
export NCCL_DEBUG=WARN
# IB timeout exponent: default 18 (~0.5 s). Raise to 22 (~8 s) to tolerate
# transient fabric congestion on large jobs without masking real failures.
export NCCL_IB_TIMEOUT=22

# 5. Distributed Networking Setup
NODES=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
export MASTER_ADDR=${NODES[0]}
export MASTER_PORT=29500

CONFIG_PATH="${1:?[ERROR] Config path is required. Usage: sbatch mn5_train_mamba.slurm <path/to/config.yaml>}"
export CONFIG_PATH="$PROJ_PATH/$CONFIG_PATH"
export NUM_TRAINERS=4

echo "[INFO] Starting DDP training on $SLURM_JOB_NUM_NODES nodes..."
echo "[INFO] Master rendezvous node is: $MASTER_ADDR"

# 6. Execution via srun
srun bash -c 'torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=$NUM_TRAINERS \
    --node_rank=$SLURM_NODEID \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    main.py --config $CONFIG_PATH'

echo "[SUCCESS] Multi-node training pipeline terminated cleanly."

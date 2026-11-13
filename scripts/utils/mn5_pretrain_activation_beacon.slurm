#!/bin/bash
#SBATCH --job-name=boa_beacon_pretrain
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

# Multi-node Activation Beacon pretraining. Launches the isolated qwen3_simple Beacon backend
# on whatever transformers version the environment provides (no
# ACTIVATION_BEACON_TRANSFORMERS_VERSION pin). Example script: adapt the account/partition/QoS and
# the environment path (BEACON_ENV) to your cluster, see docs/cluster.md on `main`.

# Abort on error (-e) and trace commands (-x) for post-mortem diagnosis
set -ex

echo "[INFO] Initializing Multi-Node Beacon Job: $SLURM_JOB_ID"

# 1. Purge defaults
module purge
# Do not source ~/.bashrc in headless Slurm jobs.

# 2. Activate the Python environment. Override with BEACON_ENV if your env lives elsewhere.
BEACON_ENV="${BEACON_ENV:-/path/to/venv}"
source "$BEACON_ENV/bin/activate"

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
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export NCCL_IB_TIMEOUT=22

# 5. Distributed Networking Setup
NODES=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
export MASTER_ADDR=${NODES[0]}
export MASTER_PORT=29500

CONFIG_PATH="${1:?[ERROR] Config path is required. Usage: sbatch mn5_pretrain_activation_beacon.slurm <path/to/config.yaml>}"
shift || true
export CONFIG_PATH="$PROJ_PATH/$CONFIG_PATH"
export PRETRAIN_EXTRA_ARGS="$*"
export NUM_TRAINERS=4

echo "[INFO] Starting Beacon DDP pretraining on $SLURM_JOB_NUM_NODES nodes..."
echo "[INFO] Master rendezvous node is: $MASTER_ADDR"
echo "[INFO] Env: $BEACON_ENV"

# 6. Execution via srun -> torchrun -> Accelerate DDP
srun bash -c 'torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=$NUM_TRAINERS \
    --node_rank=$SLURM_NODEID \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    src/pretrain_activation_beacon.py --config "$CONFIG_PATH" $PRETRAIN_EXTRA_ARGS'

echo "[SUCCESS] Multi-node Beacon pretraining pipeline terminated cleanly."

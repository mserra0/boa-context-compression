#!/bin/bash
#SBATCH --job-name=boa_train_icae
#SBATCH --account=<your_account>
#SBATCH --partition=<your_partition>
#SBATCH --qos=<your_qos>
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=80
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --signal=SIGINT@300

# Abort on error (-e) and trace commands (-x) for post-mortem diagnosis
set -ex

echo "[INFO] Initializing Multi-Node Job (ICAE): $SLURM_JOB_ID"

# 1. Purge defaults
module purge
# Do not source ~/.bashrc in headless Slurm jobs.

# Make files created in a shared HF cache group-writable.
umask 002

# 2. Activate the Python environment (replace with your own venv)
source /path/to/venv/bin/activate

# 3. Project Navigation
# SLURM_SUBMIT_DIR is the directory the job was submitted from (the snapshot directory).
# SLURM_SUBMIT_DIR is only populated by sbatch (it's the batch-script's own launch dir);
# fall back to PWD so this script also works when run directly via `bash` inside an
# interactive salloc session (dry-run / debugging before submitting with sbatch).
PROJ_PATH="${SLURM_SUBMIT_DIR:-$PWD}"
cd $PROJ_PATH

# 4. Offline mode (compute nodes without internet access)
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export WANDB_MODE="offline"
export WANDB_PROJECT="boa"

# HF cache location: keep it off the (small) home directory, e.g. on scratch.
export HF_HOME="${HF_HOME:-/path/to/hf_cache}"

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
# Use Bash Arrays to safely extract the master node without triggering SIGPIPE
NODES=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
export MASTER_ADDR=${NODES[0]}
export MASTER_PORT=29500

# SLURM_JOB_NUM_NODES is likewise an sbatch-only convenience var; derive it from the
# node list above so a manual `bash` dry-run inside salloc still gets the right node count.
export SLURM_JOB_NUM_NODES="${SLURM_JOB_NUM_NODES:-${#NODES[@]}}"

CONFIG_PATH="${1:?[ERROR] Config path is required. Usage: sbatch mn5_train_icae.slurm <path/to/config.yaml>}"
export CONFIG_PATH="$PROJ_PATH/$CONFIG_PATH"
export NUM_TRAINERS=4

echo "[INFO] Starting ICAE training on $SLURM_JOB_NUM_NODES nodes..."
echo "[INFO] Master rendezvous node is: $MASTER_ADDR"

# 6. Execution via srun
# Wrapped in bash -c with single quotes to force dynamic evaluation of SLURM variables on the compute nodes
# torchrun comes from the activated environment (no 'uv run').
#
# --signal=SIGINT@300 above fires SIGINT 300s before the walltime limit; src/train.py's
# KeyboardInterrupt handler saves an "interrupted" checkpoint on catch. Resubmit the same
# sbatch command with resume_from: latest (already set in the configs/pretrain/mn5/pretrain_icae_ae_lm_*x.yaml configs) to
# continue — this is the primary checkpointing mechanism for this recipe (see that
# config's header comment for why: the epoch-based driver only checkpoints/evals at
# epoch boundaries, and num_epochs=1 spans the whole corpus).
srun --overlap --mem=0 bash -c 'umask 002; torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=$NUM_TRAINERS \
    --node_rank=$SLURM_NODEID \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    main.py --config $CONFIG_PATH'

echo "[SUCCESS] Multi-node ICAE pretraining pipeline terminated cleanly."

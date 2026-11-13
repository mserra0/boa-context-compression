#!/bin/bash
#SBATCH --job-name=boa_train
#SBATCH --account=<your_account>
#SBATCH --partition=<your_partition>
#SBATCH --qos=<your_debug_qos>
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=80
#SBATCH --time=02:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

echo "[INFO] Initializing Job: ${SLURM_JOB_ID:-Interactive}"

# 1. Purge default modules and load the bash environment
# Strict error checking is disabled while sourcing: bashrc and activate scripts often
# contain harmless non-zero exit codes.
set +e

module purge
source ~/.bashrc

echo "[INFO] Activating offline environment..."
source /path/to/venv/bin/activate

# Re-enable strict error checking.
set -e

# 3. Weights & Biases Offline Configuration
export WANDB_MODE="offline"
export WANDB_PROJECT="boa"

# 4. Input Parsing
CONFIG_PATH=$1
NUM_TRAINERS=${2:-4} 

if [ -z "$CONFIG_PATH" ]; then
    echo "[ERROR] You must provide a configuration file path."
    echo "Usage: bash train_mn5.sh <config_path> [num_trainers]"
    exit 1
fi

if [ ! -f "$CONFIG_PATH" ]; then
    echo "[ERROR] Configuration file not found at: $CONFIG_PATH"
    exit 1
fi

# 5. Navigate to the project root directory
cd /path/to/boa-context-compression/

# 6. Distributed Execution
echo "[INFO] Launching DDP training across $NUM_TRAINERS GPUs..."

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NUM_TRAINERS \
    main.py --config "$CONFIG_PATH"

echo "[SUCCESS] Training pipeline terminated cleanly."

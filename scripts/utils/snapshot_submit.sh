#!/bin/bash

# Ensure a target script was provided
if [ -z "$1" ]; then
    echo "Error: Please provide a target slurm script."
    exit 1
fi

TARGET_SLURM_SCRIPT="$1"
shift

# We do the Git and Path resolution BEFORE changing directories
COMMIT_HASH=$(git rev-parse --short HEAD)

# Derive a human-readable label for the snapshot directory.
# Primary: wandb.name from the YAML config (first remaining arg).
# Fallback: sanitized git commit subject.
LABEL=""
CONFIG_ARG="$1"
if [[ -n "$CONFIG_ARG" && -f "$CONFIG_ARG" ]]; then
    LABEL=$(python3 -c \
        "import yaml,sys; d=yaml.safe_load(open(sys.argv[1])); print(d.get('wandb',{}).get('name',''))" \
        "$CONFIG_ARG" 2>/dev/null)
fi
if [[ -z "$LABEL" ]]; then
    LABEL=$(git log -1 --format="%s" | tr ' /' '_' | tr -dc '[:alnum:]_-' | cut -c1-50)
fi

RUN_ID="${LABEL}_${COMMIT_HASH}_$(date +%d_%m_%H_%M)"

EXP_BASE_DIR="/path/to/experiments"
DEST_DIR="${EXP_BASE_DIR}/${RUN_ID}"

echo "--- Creating snapshot in: $DEST_DIR ---"
mkdir -p "$DEST_DIR"

# Archive the current repo state into the destination
git archive --format=tar HEAD | (cd "$DEST_DIR" && tar xf -)

# Move into the new snapshot directory
cd "$DEST_DIR"
mkdir -p logs

# Find the exact path of the slurm script inside the new snapshot
SNAPSHOT_SLURM_SCRIPT=$(find . -name "$(basename "$TARGET_SLURM_SCRIPT")" | head -n 1)

if [ -z "$SNAPSHOT_SLURM_SCRIPT" ]; then
    echo "Error: Could not locate slurm script inside the snapshot!"
    exit 1
fi

echo "Submitting $SNAPSHOT_SLURM_SCRIPT from $DEST_DIR..."

sbatch -A <your_account> --qos <your_qos> "$SNAPSHOT_SLURM_SCRIPT" "$@"

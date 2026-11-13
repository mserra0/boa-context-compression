#!/bin/bash
# Pre-download the Hugging Face models used by the configs in this repository (for offline compute nodes).
# Datasets: see docs/data/datasets.md (MRQA via scripts/download_mrqa.sh on main; FineWeb-Edu via `hf download`).
# Set HF_HOME first if the cache should live elsewhere than ~/.cache/huggingface.
set -e

echo "Downloading state-spaces/mamba2-370m..."
hf download state-spaces/mamba2-370m

echo "Downloading state-spaces/mamba2-1.3b..."
hf download state-spaces/mamba2-1.3b

echo "Downloading state-spaces/mamba-370m-hf..."
hf download state-spaces/mamba-370m-hf

echo "Downloading state-spaces/mamba-1.4b-hf..."
hf download state-spaces/mamba-1.4b-hf

echo "Downloading EleutherAI/gpt-neox-20b..."
hf download EleutherAI/gpt-neox-20b

echo "Downloading Qwen/Qwen2.5-0.5B-Instruct..."
hf download Qwen/Qwen2.5-0.5B-Instruct

echo "All models have been downloaded."

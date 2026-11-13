#!/bin/bash
NCCL_P2P_DISABLE=1 \
HF_HUB_CACHE=/path/to/hf_cache/hub \
CUDA_VISIBLE_DEVICES=${JUDGE_GPUS:-0,1} \
uv run vllm serve Qwen/Qwen3-32B \
  --port 8080 \
  --tensor-parallel-size 2 \
  --disable-custom-all-reduce
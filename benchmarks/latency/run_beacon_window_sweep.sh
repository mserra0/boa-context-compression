#!/usr/bin/env bash
# Three concurrent five-way latency sweeps, one per GPU.
#
# The variable under test is Activation Beacon's window size: the 1.7B backbone has checkpoints at
# window 128 and window 2048, so both are swept against the same set of baselines and the results
# files are named by window. The 4B backbone only has a window-128 checkpoint, so it contributes
# one run.
#
# Configuration (environment variables):
#   BOA_17B_RUN   required  run directory (config.yaml + best_model.pt) of the 1.7B checkpoint
#   BOA_4B_RUN    required  same, for the 4B checkpoint
#   CKPT_ROOT     default /path/to/checkpoints
#   BEACON_ROOT   default /path/to/checkpoints/activation_beacon
#   SAC_17B       default /path/to/checkpoints/<sac_ckpt>
#   GPUS          default "0 1 2"   three GPU indices (1.7B w2048, 1.7B w128, 4B w128)
set -u

cd "$(dirname "$0")/../.." || exit 1

: "${BOA_17B_RUN:?set BOA_17B_RUN (see the header of this script)}"
: "${BOA_4B_RUN:?set BOA_4B_RUN}"

CKPT_ROOT=${CKPT_ROOT:-/path/to/checkpoints}
BEACON_ROOT=${BEACON_ROOT:-/path/to/checkpoints/activation_beacon}
read -r GPU_A GPU_B GPU_C <<< "${GPUS:-0 1 2}"

BOA_17B=$BOA_17B_RUN
BOA_4B=$BOA_4B_RUN
QWEN_17B=$CKPT_ROOT/qwen3_1_7b_fullft/final
QWEN_4B=$CKPT_ROOT/qwen3_4b_fullft/final

# --beacon-base is passed because the checkpoints' own base weights may not be readable. Only the
# beacon_* tensors train, so any copy of the same base backbone loads identical weights.
BEACON_17B_W128=${BEACON_17B_W128:-$BEACON_ROOT/qwen3-17b-fullft/window_128/checkpoint-epoch-1}
BEACON_17B_W2048=${BEACON_17B_W2048:-$BEACON_ROOT/qwen3-17b-fullft/window_2048/checkpoint-epoch-1}
BEACON_4B_W128=${BEACON_4B_W128:-$BEACON_ROOT/qwen3-4b-fullft/window_128/checkpoint-epoch-1}

SAC_17B=${SAC_17B:-/path/to/checkpoints/<sac_ckpt>}

OUT_17B=benchmarks/latency/outputs/qwen3_1-7b
OUT_4B=benchmarks/latency/outputs/qwen3_4b
mkdir -p "$OUT_17B" "$OUT_4B"

launch() {
  local gpu=$1 out=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu nohup uv run python -m benchmarks.latency.bench \
    --bench-config benchmarks/latency/config_sota.yaml \
    --dtype bf16 \
    --out "$out.jsonl" \
    "$@" >"$out.log" 2>&1 &
  echo "gpu $gpu -> $out.jsonl  (pid $!)"
}

# 1.7B, Beacon window 2048.
launch "$GPU_A" "$OUT_17B/results_sota_1-7b_beacon_w2048" \
  --config "$BOA_17B/config.yaml" --checkpoint "$BOA_17B/best_model.pt" \
  --beacon-checkpoint "$BEACON_17B_W2048" --beacon-base "$QWEN_17B" --beacon-ratio 8 \
  --sac-checkpoint "$SAC_17B" --sac-ratio 8 --sac-chunk 512 \
  --icae-base "$QWEN_17B" --icae-ratio 8

# 1.7B, Beacon window 128, same host conditions.
launch "$GPU_B" "$OUT_17B/results_sota_1-7b_beacon_w128" \
  --config "$BOA_17B/config.yaml" --checkpoint "$BOA_17B/best_model.pt" \
  --beacon-checkpoint "$BEACON_17B_W128" --beacon-base "$QWEN_17B" --beacon-ratio 8 \
  --sac-checkpoint "$SAC_17B" --sac-ratio 8 --sac-chunk 512 \
  --icae-base "$QWEN_17B" --icae-ratio 8

# 4B, Beacon window 128. No window-2048 checkpoint exists for this backbone.
# SAC has no 4B checkpoint either; --sac-base alone is equivalent for latency
# (the trained tensors change no matmul shape).
launch "$GPU_C" "$OUT_4B/results_sota_4b_beacon_w128" \
  --config "$BOA_4B/config.yaml" --checkpoint "$BOA_4B/best_model.pt" \
  --beacon-checkpoint "$BEACON_4B_W128" --beacon-base "$QWEN_4B" --beacon-ratio 8 \
  --sac-base "$QWEN_4B" --sac-ratio 8 --sac-chunk 512 \
  --icae-base "$QWEN_4B" --icae-ratio 8

wait

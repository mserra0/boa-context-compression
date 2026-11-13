#!/usr/bin/env bash
# Latency sweeps behind `paper_panels.py`: four results files, two decoder scales, both
# Activation Beacon windows.
#
# Only three sweeps are launched here. The fourth file -- 1.7B at Beacon window 128 -- is a
# five-way sweep produced separately with `bench.py` and `config_sota.yaml`; point
# RESULTS_17B_W128 at it and it is copied into place (never overwriting an existing file).
#
# Every sweep behind one set of figures should run on the same GPU model, because
# `paper_panels.py` shares one y range per metric across both decoder scales; ranges derived
# from runs on different cards would compare hardware rather than methods.
#
# Configuration (environment variables):
#   BOA_17B_RUN        required  run directory (config.yaml + best_model.pt) of the 1.7B checkpoint
#   BOA_4B_RUN         required  same, for the 4B checkpoint
#   RESULTS_17B_W128   required  existing five-way sweep for 1.7B at Beacon window 128
#   CKPT_ROOT          default /path/to/checkpoints      (frozen-LLM checkpoints qwen3_*_fullft/final)
#   BEACON_ROOT        default /path/to/checkpoints/activation_beacon
#   SAC_17B            default /path/to/checkpoints/<sac_ckpt>
#   GPU_FIVEWAY        default 0   GPU for the five-way 4B sweep
#   GPU_BEACON         default 1   GPU for the two Beacon-only sweeps
#
# Progress is written to the .log files next to each results file.
set -u

cd "$(dirname "$0")/../.." || exit 1

: "${BOA_17B_RUN:?set BOA_17B_RUN (see the header of this script)}"
: "${BOA_4B_RUN:?set BOA_4B_RUN}"
: "${RESULTS_17B_W128:?set RESULTS_17B_W128}"

CKPT_ROOT=${CKPT_ROOT:-/path/to/checkpoints}
BEACON_ROOT=${BEACON_ROOT:-/path/to/checkpoints/activation_beacon}
GPU_FIVEWAY=${GPU_FIVEWAY:-0}
GPU_BEACON=${GPU_BEACON:-1}

BOA_17B=$BOA_17B_RUN
BOA_4B=$BOA_4B_RUN
QWEN_17B=$CKPT_ROOT/qwen3_1_7b_fullft/final
QWEN_4B=$CKPT_ROOT/qwen3_4b_fullft/final

# --beacon-base is passed because a Beacon checkpoint's own base weights may not be readable
# (safetensors reports that as a misleading FileNotFoundError). Only the beacon_* tensors train,
# so any copy of the same base backbone loads identical weights.
BEACON_17B_W2048=${BEACON_17B_W2048:-$BEACON_ROOT/qwen3-17b-fullft/window_2048/checkpoint-epoch-1}
BEACON_4B_W128=${BEACON_4B_W128:-$BEACON_ROOT/qwen3-4b-fullft/window_128/checkpoint-epoch-1}

SAC_17B=${SAC_17B:-/path/to/checkpoints/<sac_ckpt>}

OUT_17B=benchmarks/latency/outputs/qwen3_1-7b
OUT_4B=benchmarks/latency/outputs/qwen3_4b
mkdir -p "$OUT_17B" "$OUT_4B"

W128_17B=$OUT_17B/results_sota_1-7b_beacon_w128.jsonl
if [ -f "$W128_17B" ]; then
  echo "keeping existing $W128_17B"
else
  cp "$RESULTS_17B_W128" "$W128_17B"
  echo "copied $RESULTS_17B_W128 -> $W128_17B"
fi

# Runs in the foreground of the calling shell; the caller backgrounds the whole call. Because
# `uv run` spawns a child python instead of exec'ing it, backgrounding here would give the pid
# of the uv wrapper: `wait` could return while the sweep is still on the card and the next sweep
# in a chain would land on a busy GPU.
run() {
  local gpu=$1 out=$2 cfg=$3 rc; shift 3
  echo "gpu $gpu -> $out.jsonl  ($cfg)"
  CUDA_VISIBLE_DEVICES=$gpu uv run python -m benchmarks.latency.bench \
    --bench-config "benchmarks/latency/$cfg" \
    --dtype bf16 \
    --out "$out.jsonl" \
    "$@" >"$out.log" 2>&1
  rc=$?
  echo "gpu $gpu done ($rc) -> $out.jsonl"
  # A nonzero exit is expected on the five-way sweep: `ssm_kv` at 262144 overflows int32
  # indexing in the mamba2 Triton kernels and the illegal memory access kills the process. Rows
  # stream to disk as they complete, so the file is complete up to that cell.
  return 0
}

# Five-way 4B sweep. Everything the 4B figures draw except the second Beacon window comes from
# this one file, so all five methods share one process and one set of host conditions.
#
# SAC has no 4B checkpoint here; --sac-base alone is equivalent for latency, since the trained
# tensors change no matmul shape (the same argument holds for sweeping ICAE without one).
run "$GPU_FIVEWAY" "$OUT_4B/results_sota_4b_beacon_w128" config_sota.yaml \
  --config "$BOA_4B/config.yaml" --checkpoint "$BOA_4B/best_model.pt" \
  --beacon-checkpoint "$BEACON_4B_W128" --beacon-base "$QWEN_4B" --beacon-ratio 8 \
  --sac-base "$QWEN_4B" --sac-ratio 8 --sac-chunk 512 \
  --icae-base "$QWEN_4B" --icae-ratio 8 &

# The two second-window runs, back to back. Beacon only: the figures take just the `beacon` rows
# out of a window-2048 file, so re-running the four baselines would re-measure what the window-128
# files already hold.
#
# The 1.7B window-2048 checkpoint is trained at that window. The 4B one is the window-128
# checkpoint run at 2048 (`--beacon-window`), because no 4B window-2048 checkpoint exists. That is
# sound for latency -- no beacon tensor's shape depends on the window -- and for nothing else.
# bench.py records `beacon_window_overridden: true`, paper_panels.py prints a warning that
# applies to the 4B figure.
(
  run "$GPU_BEACON" "$OUT_17B/results_sota_1-7b_beacon_w2048" config_beacon_only.yaml \
    --beacon-checkpoint "$BEACON_17B_W2048" --beacon-base "$QWEN_17B" --beacon-ratio 8

  run "$GPU_BEACON" "$OUT_4B/results_beacon_4b_w2048" config_beacon_only.yaml \
    --beacon-checkpoint "$BEACON_4B_W128" --beacon-base "$QWEN_4B" --beacon-ratio 8 \
    --beacon-window 2048
) &

wait
echo "all sweeps finished"

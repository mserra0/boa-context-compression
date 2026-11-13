# End-to-End Latency Benchmark

`benchmarks/latency/` measures **end-to-end latency = Compression Time + Generation Time** across sequence lengths.

Most SOTA compression methods (Activation Beacon, SAC, ICAE) report only decoder inference time *after* the KV cache is filled, ignoring the cost of building the compressed representation. This suite measures both stages so the heterogeneous setup (linear-time SSM encoder + Transformer LLM) can be contrasted against feeding the full context to the LLM (quadratic prefill).

All three of those baselines are implemented here as adapters, and `config_sota.yaml` runs the five-way comparison at a uniform **8x** compression rate. Every method must be launched at 8x for the numbers to mean anything — the `C` column of the summary table is the check, and it has to read `seq_len / 8` on every row.

```bash
CKPT=/path/to/checkpoints/<run_dir>
CUDA_VISIBLE_DEVICES=0 uv run python -m benchmarks.latency.bench \
  --config "$CKPT/config.yaml" \
  --checkpoint "$CKPT/best_model.pt" \
  --bench-config benchmarks/latency/config.yaml
```

`--checkpoint` is the `best_model.pt` file, **not** the run directory. `--config` reuses the checkpoint's own `config.yaml`. Loading goes through `load_resources` (`src/utils/inference_utils.py`), the same path as `src/inference.py`.

Inference runs in **bf16 by default** (`--dtype bf16`). The whole model is cast with a plain `model.to(torch.bfloat16)` rather than `torch.autocast`: mamba2's custom SSM kernels follow the parameter dtype directly, and `generate_compressed` derives its mask dtype from the LLM parameters, so a straight cast propagates correctly through both stages. Pass `--dtype fp32` to disable.

---

## What It Compares

| Adapter | Encoder stage | Generation stage |
|---|---|---|
| `ssm` | Real `MambaCompressor.compress_single` → soft prompts `[1, C, H]` | `generate_compressed` over `C` soft tokens |
| `ssm_kv` | Fused into generation (see below) | `generate_compressed_kv` — SSM forward → per-layer K/V → injected KV cache → decode |
| `full_ctx` | None (no-op) | `llm.generate` over the full `L` raw tokens |
| `beacon` | Fused into generation (see below) | Activation Beacon's recurrent window loop, then decode against beacon memory |
| `sac` | `CompressLLM.compress` — per-chunk bidirectional encode, keep K/V at strided positions | Greedy decode from the `[LM]` token against the compressed KV memory |
| `icae` | `ICAECompressor.compress` — one pass over `[context ‖ MEM]`, keep the hidden states at the MEM positions | `generate_compressed` over the `n_mem` memory slots |

Pick the adapter that matches the checkpoint's `prompt_mode`: `ssm` for `embed` checkpoints, `ssm_kv` for `kv` ones. Running the embed path against a KV-trained checkpoint would exercise an untrained projector and measure the wrong pipeline.

`full_ctx` pays no compression cost; its entire cost lands in the prefill inside generation. That is the point — the quadratic prefill is the baseline being beaten.

`ssm_kv` also reports `t_compress = 0`, but for a different reason: `generate_compressed_kv` fuses the SSM forward, the KV-cache build and decoding into one call, so there is no separable compress stage without reaching into model internals. The compression cost lands in `t_ttft` instead — which is the honest reading anyway, since `t_ttft` is then the true time-to-first-token including compression, directly comparable to `full_ctx`'s prefill.

---

## Activation Beacon

`beacon` is the homogeneous SOTA comparison: the same Transformer both compresses and generates. Its accounting is described below.

```bash
BEACON=/path/to/checkpoints/<beacon_ckpt>/checkpoint-epoch-1
QWEN3=/path/to/checkpoints/qwen3_1_7b_fullft/final

CUDA_VISIBLE_DEVICES=0 uv run python -m benchmarks.latency.bench \
  --beacon-checkpoint "$BEACON" \
  --beacon-base "$QWEN3" \
  --beacon-ratio 8 \
  --bench-config benchmarks/latency/config.yaml
```

Add `beacon` to `adapters` in the bench config to include it. It is **off by default**: its window loop is serial, ttft is already ~550 ms at `L=2048`, and the sweep's long tail would take hours.

Four things about this baseline are not obvious:

- **The backend is vendored.** `benchmarks/latency/beacon/qwen3_official.py` is vendored from `src/baselines/activation_beacon/qwen3_official.py` (the full training code lives on the `baseline/activation-beacon` branch). That branch's version adds training-only features this inference-only benchmark does not use: gradient checkpointing (active only when `self.training`) and a `fixed` ratio mix (unreachable with a single `--beacon-ratio`, which returns before it). `support.py` holds the three model-side helpers it imports from that branch's `training.py` (the rest of that module drags in the MRQA data pipeline).
- **`--beacon-base` may be required.** If the checkpoint's own base weights (`model.safetensors`) are not readable or not included, pass `--beacon-base` pointing to the base model; only the beacon adapter `.pt` and the configs are needed from the checkpoint. Activation Beacon never updates the backbone — the checkpoint's 85 trainable tensors are all `beacon_*` — so pointing at any copy of the same Qwen3 base loads identical weights. The checkpoint's own `activation_beacon_training.json` names this exact path as `beacon_code_model`.
- **`--beacon-ratio` must be a single value.** Checkpoints are trained with a ratio *list* (`[2, 4, 8, 16]`) sampled per step. Left as a list, both the compression rate and the latency would vary between repeats, and the median would be meaningless. A one-element list short-circuits the sampler.
- **Compression is not separable**, so `t_compress = 0` and the cost lands in `t_ttft` — same accounting as `ssm_kv`. Beacon's `generate` runs the whole windowed compression inside the first forward pass. This is exactly the cost the Activation Beacon paper omits by reporting only post-KV-fill decode.

Only one method's model is resident at a time. The runner groups adapters by `loader`, runs a group, then frees it — otherwise Beacon's backbone and our compressor+LLM would share the device and every `peak_mem_mb` reading would be inflated by the other method's weights.

---

## SAC

`sac` is the second homogeneous baseline, and the closest one to a chunked encoder. Each `chunk_size` window of context is encoded independently under **full bidirectional attention** by a LoRA-adapted copy of the LLM; the K/V at `chunk_size // ratio` uniformly strided positions inside that window survives as the chunk's memory, and a **separate frozen copy** of the same model decodes against the concatenation.

**Chunks never see each other** — no `past_key_values` is threaded through the loop — so the encoder is `O(chunk · L)`, *linear*, not quadratic. This is the single most counter-intuitive thing about the method's measured cost; the [FLOP accounting](#why-sac-is-linear-and-why-it-is-that-fast) below verifies it against the code and the runtime.

```bash
SAC=/path/to/checkpoints/<sac_ckpt>

CUDA_VISIBLE_DEVICES=0 uv run python -m benchmarks.latency.bench \
  --sac-checkpoint "$SAC" \
  --sac-ratio 8 \
  --sac-chunk 512 \
  --bench-config benchmarks/latency/config_sota.yaml
```

- **The backend is vendored**, like Beacon's: `benchmarks/latency/sac/modeling.py` is a modified copy of `SAC/model/modeling.py` from the `baseline/sac` branch. Five deviations are each marked `# BENCH:` in the file or its header: the `path_config` bootstrap is dropped and the LoRA import repointed, the encoder's attention mask follows the input dtype (so `--dtype fp32` runs), the LoRA layers take their dtype from the wrapped weight, the per-chunk hidden-state collection that only `ae_inference` uses can be switched off (the timed QA/LM path never reads it), and the KV-cache helpers are rewritten for Transformers 5. The KV-cache helpers use the Transformers 5 `Cache` API.
- **LoRA does not come from PEFT.** PEFT is not a dependency here, so `benchmarks/latency/lora.py` provides SAC's `LinearLoraLayer` with the same layer math, plus an `enabled` toggle and a `dtype` argument (rank 128 on `q_proj`/`v_proj`, `scale = 2`). SAC's trained `adapter.pt` loads onto it unchanged.
- **The checkpoint is optional.** The trained tensors are the [MEM] role embeddings, two special tokens and the LoRA factors — none of them change the shape of a single matmul, so latency is identical to a randomly initialised model. Pass `--sac-base <model>` alone if the checkpoint is unavailable. When a checkpoint *is* given, the SFT-stage `instruction_adapter.pt` is preferred over the pretrain `adapter.pt`.
- **`--sac-ratio` and `--sac-chunk` override the checkpoint**, so one checkpoint can be swept at several rates and the rate that gets measured is always the one that was asked for. Memory slots per chunk = `chunk // ratio`, so 8x at the trained chunk size of 512 is 64 slots per chunk, at every sequence length.
- **Compression is separable**, unlike `ssm_kv` and `beacon`: `compress()` is its own call, so `sac` reports a real `t_compress` and its `t_ttft` covers the decoder prefill and the first token.
- **The question stays literal on the decoder side.** SAC's own evaluator calls `lm_inference`, which prefills `[memory ‖ LM ‖ question]` — the question is *not* compressed. The adapter therefore prefills a `--question-len` block (default 32) exactly as `ssm_kv` does, so the two methods that keep a literal question pay the same prefill. Omitting it under-reports SAC's time-to-first-token by ~2 ms.
- **The reference `compress()` builds a result it throws away.** It collects `output_hidden_states` from every chunk, stacks them and concatenates — then `return`s `None` in that slot. Those tensors (`compress_token`, `encoder_hidden_states`) feed only `ae_inference`; the QA/LM path never reads them. Producing them cost **~1 GB of peak memory at `L=32k` (11%), growing linearly**, for no effect on the output: with the collection off, the returned KV memory is bit-identical and `encoder_mem_size` is unchanged. The loader sets `model.collect_hidden_states = False` so the memory column measures the method rather than a leftover of its training code. Set it back to `True` to reproduce the reference exactly.
- **SAC holds two full model copies** (LoRA encoder + frozen decoder), which is why its `peak_mem_mb` baseline is ~6.9 GB against ICAE's ~3.4 GB for the same 1.7B backbone.

---

## ICAE

`icae` appends `n_mem` learnable [MEM] embeddings to the context and reads the whole thing — `[context ‖ MEM]` — in **one** causal pass through a LoRA-adapted copy of the LLM. The hidden states at the MEM positions are the compressed representation; the same weights, with LoRA switched off, act as the frozen decoder.

```bash
QWEN3=/path/to/checkpoints/qwen3_1_7b_fullft/final

CUDA_VISIBLE_DEVICES=0 uv run python -m benchmarks.latency.bench \
  --icae-base "$QWEN3" \
  --icae-ratio 8 \
  --bench-config benchmarks/latency/config_sota.yaml
```

- **The encoder is quadratic in `L`, with no chunking and no recurrence.** It is the only compression baseline here that pays full attention over the entire context in a single pass, which is why `config_sota.yaml` schedules it first: expect it to OOM well before the tail. The reference implementation never runs it past a 1024-token context.
- **The question is compressed *with* the context**, unlike SAC's. `_run_student_icae` builds its compressor input from `_build_aligned_batch(context, question)` with the default `compress_question=True`, and then calls `generate_compressed(soft_prompts)` — memory slots only, no literal prompt block. The adapter matches: compress `L` tokens, decode from memory alone. So ICAE gets no `--question-len` block, and that is correct rather than an omission.
- **`n_mem` is set per cell, not once.** The `baseline/icae` branch computes `n_mem = round(max_ctx * compression_ratio)` at startup — a checkpoint trained at `max_ctx=1024` and ratio 8 always emits 128 slots, whatever the input length. Held fixed across a sweep that spans 512 → 256k tokens, the compression rate would drift by three orders of magnitude and the comparison would be meaningless. Here `n_mem = ceil(seq_len / --icae-ratio)`, so the rate is exactly 8x in every cell.
- **No checkpoint is needed, and one barely helps.** Slot embeddings and LoRA factors do not change any matmul shape. Worse, a checkpoint's [MEM] bank is *fixed at the trained `n_mem`*, so it can only ever populate the one cell whose slot count happens to match. `--icae-checkpoint` exists but loads non-strictly and is off the critical path.
- **The code is vendored, not imported.** `benchmarks/latency/icae/modeling.py` lifts the inference paths of `ICAECompressor.forward` and `ICAEDestiller.generate_compressed` from the `baseline/icae` branch. Importing that branch's `src/model.py` would drag in its Mamba compressors, KD losses and training wrappers, all of which would then have to be kept in sync with this branch's own `src/`.
- **Encoder and decoder share one set of weights** (LoRA on vs. off), halving VRAM against SAC's two copies. The `enabled` flag on `LinearLoraLayer` replaces PEFT's `enable_adapter_layers()` / `disable_adapter_layers()`.
- **Compression is separable**, so like `sac`, `icae` reports a real `t_compress`.

---

## Measurement

Per `(adapter, seq_len)` cell:

| Metric | How |
|---|---|
| `t_compress` | Time of the encoder stage (SSM forward; ~0 for `full_ctx`). |
| `t_ttft` | Time-to-first-token: `generate` with `max_new_tokens=1` (prefill + 1 token). |
| `t_decode` | `t_total_gen − t_ttft` — amortised per-token decode. |
| `t_total` | `t_compress + t_total_gen` — the headline end-to-end number. |
| `peak_mem_mb` | `torch.cuda.max_memory_allocated` for the cell. |

Rigor:

- Every stage is CUDA-synced (`torch.cuda.synchronize`) around `perf_counter`.
- `warmup` discarded runs + `repeats` timed runs per stage; the reported value is the **median** (robust to scheduler noise).
- `batch=1` (latency, not throughput), greedy decoding.
- Inputs are **synthetic random token ids** at the exact requested length. Token identity is irrelevant to latency, so this keeps the sweep reproducible, dataset-free, and gives exact control over the sequence-length axis.
- **Decode is forced to full length.** Random inputs would otherwise hit an EOS id and stop early, making `t_decode` meaningless. HF generation gets `min_new_tokens == max_new_tokens`; the compressed paths get `eos_token_id=[-1]`, which can never match a real argmax.
- **`t_gen_full >= t_ttft` is checked.** Generating `max_new_tokens` cannot be faster than generating one, so a negative `t_decode` means the two stages were not measured under comparable conditions. Such rows are flagged `suspect` and dropped by `plot.py` (which re-derives the check, so older result files are covered too). It can trigger for methods whose first-token path is partly CPU-bound: Beacon's window loop runs 2048 Python-level iterations per forward at 256k with CPU-side tensor concatenation in the memory controller, so heavy host load affects it more than the GPU-bound methods.
- **Rows stream to disk as they complete.** Only OOM is recoverable; any other CUDA fault poisons the context and must kill the process. Writing rows incrementally means a fault costs one cell instead of the whole sweep.
- **A global warmup runs per adapter before the timed loop.** One-time CUDA / cuBLAS / kernel-autotune allocations otherwise land inside the first measured cell and inflate its `peak_mem_mb` (~3%, enough to make the smallest sequence length report *more* memory than the next one up). With the warmup, peak memory is monotonic in `L`.
- A CUDA OOM records `NaN` and continues — `full_ctx` OOMing before `ssm` is itself a result.

The `t_ttft` / `t_decode` split comes from calling generation twice (`max_new_tokens=1`, then the full length) rather than hooking model internals — simpler and adapter-agnostic.

---

## Files

| File | Responsibility |
|---|---|
| `adapters.py` | Per-method functions `prepare` / `compress` / `generate` + the `ADAPTERS` registry. Plain functions and dicts — no class hierarchy. |
| `runner.py` | The sweep loop and staged timing. Knows nothing about specific models — only the adapter protocol and the `loader` grouping. |
| `bench.py` | CLI entry: build the loader table, run sweep, write `results.jsonl`, print a summary table. |
| `beacon/` | Vendored Activation Beacon backend + its loader. Benchmark-only; not part of `src/`. |
| `sac/` | Vendored SAC backend + its loader and decode loop. Benchmark-only. |
| `icae/` | Vendored ICAE encoder/decoder paths + its loader. Benchmark-only. |
| `lora.py` | The one `LinearLoraLayer` shared by SAC and ICAE (PEFT is not a dependency). |
| `plot.py` | Read `results.jsonl` → one metric vs sequence length, one figure per metric. |
| `config.yaml` | Sweep parameters: `seq_lens` (512 → 512k), `adapters`, `max_new_tokens`, `warmup`, `repeats`. |
| `config_beacon.yaml` | Three-way sweep: ours vs Activation Beacon vs full context. |
| `config_sota.yaml` | Five-way sweep: ours vs ICAE vs SAC vs Activation Beacon vs full context, all at 8x. |
| `config_beacon_only.yaml` | Activation-Beacon-only sweep (adds a second Beacon window to results that already hold the other methods). |
| `paper_figure*.py`, `paper_panels.py` | Figure scripts built on `plot.py`: end-to-end figures per decoder scale and per-metric panels. |
| `run_paper_sweeps.sh`, `run_beacon_window_sweep.sh` | Sweep launchers for those figures (see Sweep scripts below). |

### Adapter protocol

```python
prepare(model, ctx, seq_len) -> state              # build inputs at exact length
compress(model, state)       -> gen_state          # timed stage 1 (encoder)
generate(model, gen_state, max_new_tokens) -> int  # timed stage 2, returns #new tokens
```

Each `ADAPTERS` entry also names a `loader`, and `bench.py` maps that name to a callable returning `(model, ctx)`. Methods that run on our checkpoint use `boa`; a method that brings its own backbone declares its own loader. `ctx` is per-loader — every loader supplies `device`, plus:

| Loader | Extra `ctx` keys |
|---|---|
| `boa` | `llm_vocab`, `compressor_vocab`, `pad_token_id`, `question_len` |
| `beacon` | `beacon_vocab` |
| `sac` | `sac_vocab`, `question_len` |
| `icae` | `icae_vocab`, `icae_ratio` (`n_mem` is derived per cell from it) |

`question_len` (`--question-len`, default 32) is shared by the two methods that keep the question **uncompressed** on the decoder side — `ssm_kv` (Design B-ctx) and `sac` — so their prefill costs stay comparable. `ssm`, `icae`, `beacon` and `full_ctx` all fold the question into the `L` tokens they process, so they take no block.

`gen_state` is built once per cell and reused across timings, so generation must not mutate it. The compressed-KV methods need care here: SAC rebuilds a fresh `Cache` from its stored per-layer memory on every `generate` call, because a decode pass would otherwise `cat` new tokens onto the memory it is supposed to start from.

---

## Output

Results are written to `benchmarks/latency/outputs/results.jsonl` (gitignored). Line 1 is a `meta` record (checkpoint, device, `max_new_tokens`, `warmup`, `repeats`); each following line is one cell:

```json
{"adapter": "ssm", "seq_len": 1024, "compressed_tokens": 103, "gen_tokens": 64,
 "t_compress_ms": 37.6, "t_ttft_ms": 13.8, "t_decode_ms": 89.7, "t_total_ms": 141.1,
 "peak_mem_mb": 3638, "oom": false}
```

At short `L`, `full_ctx` wins — the SSM compression cost is not yet amortised. The crossover appears at long `L`, where full-context prefill grows quadratically. The crossover point depends on the compressor and the decoder.

### Plotting

```bash
uv run python -m benchmarks.latency.plot \
  --results benchmarks/latency/outputs/results.jsonl \
  --out benchmarks/latency/outputs/latency_curve.png
```

One metric (y) vs sequence length (x, log2 axis), one curve per adapter, one metric per figure. OOM / NaN cells are dropped from a curve. `--metric` selects the field: latency fields (`t_total_ms` default, `t_compress_ms`, `t_ttft_ms`, `t_decode_ms`, plotted in ms) or `peak_mem_mb` (plotted as a standalone memory figure in GB). `--logy` log-scales the y axis.

For the summary view of a whole sweep — every method, every metric, one figure:

```bash
uv run python -m benchmarks.latency.plot --panels \
  --results benchmarks/latency/outputs/results_sota.jsonl \
  --out benchmarks/latency/outputs/sota_panels.png \
  --title "Ours vs ICAE vs SAC vs Activation Beacon"
```

`--panels` draws a 2×2 grid — `t_compress_ms`, `t_ttft_ms`, `t_total_ms`, `peak_mem_mb` — in a wide figure, with one shared legend below. Colours and markers are assigned **once from every row in the file**, not per panel, so a method that OOMs on one metric but survives on another keeps its identity across the grid. Methods are ordered ours → compression baselines → uncompressed reference.

Use the single-metric form for one specific comparison, and `--panels` for the overview.

### Sweep scripts

`benchmarks/latency/run_paper_sweeps.sh` and `benchmarks/latency/run_beacon_window_sweep.sh` run the sweeps behind `paper_panels.py` and the Beacon-window figure (two decoder scales, both Beacon windows). They read the checkpoint locations and GPU indices from environment variables documented in the header of each script. Both read `BOA_17B_RUN`, `BOA_4B_RUN`, `CKPT_ROOT`, `BEACON_ROOT` and `SAC_17B`; `run_paper_sweeps.sh` selects GPUs with `GPU_FIVEWAY` and `GPU_BEACON` (and takes `RESULTS_17B_W128`), while `run_beacon_window_sweep.sh` uses `GPUS` (three indices). All sweeps behind one set of figures should run on the same GPU model.

Figures use the [SciencePlots](https://github.com/garrettj403/SciencePlots) `science + nature` style with `no-latex` (no LaTeX toolchain is required; the modifier must come after the base styles to win).

---

## Measured Results

Measurements: single NVIDIA GPU with 97 GB of memory.

### Five-way: ours vs ICAE vs SAC vs Activation Beacon vs full context (bf16, **8x**, 512 → 256k)

`config_sota.yaml`, Qwen3-1.7B backbone throughout, `warmup 1 / repeats 3`, `max_new_tokens 64`. 54 of 55 cells completed; every compression method held **exactly `L/8`** on every row, and no row tripped the `t_gen_full >= t_ttft` guard. The plotting step below generates `outputs/sota_panels.png` (combined) and `outputs/sota_<metric>.png`; these figures are generated locally and are not shipped.

At **245,760 tokens** — the largest length every method reaches:

| Method | Context (compress + TTFT) | Total | Peak mem | Fitted exponent |
|---|---|---|---|---|
| **`ssm_kv` (ours)** | **4.5 s** | **5.6 s** | 23.2 GB | **1.07** |
| `sac` | 12.6 s | 13.7 s | 16.5 GB | **1.02** |
| `full_ctx` | 33.2 s | 37.2 s | 44.5 GB | 1.80 |
| `icae` | 45.1 s | 46.2 s | 48.9 GB | 1.84 |
| `beacon` | 112.6 s | 116.2 s | **10.5 GB** | 1.60 |

Exponents are fitted over 131,072 → 245,760. Ours: **2.8× faster than SAC, 7.3× than full context, 9.9× than ICAE, 24.8× than Beacon** on context processing; 2.4× / 6.6× / 8.2× / 20.6× on end-to-end total.

**Read `t_context`, not `t_compress`, when comparing methods.** The methods split that cost differently: `sac` and `icae` expose a separable `compress()` and report it, while `ssm_kv`, `beacon` and `full_ctx` fuse compression into the first forward pass and report `t_compress = 0`. Only the sum — raw tokens to first generated token — is the same quantity for all five, which is why it is the combined figure's first panel.

**Two methods are linear, three are not.** Ours (1.07) and SAC (1.02) scale with `L`; ICAE (1.84) and full context (1.80) are near-quadratic, and Beacon sits at 1.60. SAC's linearity is not an artifact — it encodes independent 512-token chunks, so its cost is `O(chunk · L)`; see the FLOP accounting below, which also explains why our margin over it is a constant factor rather than an asymptotic one. Beacon's is structural — each window attends over all accumulated beacon memory, so the cost is `O(L/window × L/ratio)`. **ICAE is the slowest compression method past 128k and the most memory-hungry of all five at 245k (48.9 GB, above even the uncompressed baseline)**: its encoder pays full attention over `L + n_mem` in a single pass with no chunking and no recurrence. Its own evaluation never exceeds a 1024-token context.

**Compressing does not automatically beat not compressing.** ICAE and Beacon are both *slower end-to-end than `full_ctx`* across this entire sweep — at 245k, Beacon's compression alone costs 3.4× the quadratic prefill it replaces. Compression pays off only when the encoder is cheaper than the prefill it removes, which is what the linear methods deliver and the homogeneous ones do not.

**Memory ranks differently from latency.** Beacon wins it (10.5 GB) — one 1.7B backbone, no raw KV retained — at 20.6× the latency. SAC is second (16.5 GB) on two model copies but chunk-bounded activations. We sit third (23.2 GB) carrying a compressor *and* an LLM. ICAE and full context are far behind. Our memory win is against the uncompressed baseline (1.9×), not against every baseline.

**SAC is the strongest baseline here**: same linear scaling as ours, less memory, 2.4–2.8× slower. Its `t_ttft` stays flat at ~20–90 ms across the whole sweep because its memory arrives as a **prebuilt KV cache**, so the first token is one decode step. ICAE's memory is a block of **embeddings**, so its decoder must prefill all `n_mem` slots first — 19 → 3341 ms as `n_mem` grows 64 → 30,720.

#### Why SAC is linear, and why it is that fast

SAC's measured exponent (1.02) is linear even though its encoder is a Transformer. This follows from the code, and the FLOP accounting predicts the runtime to rounding.

**SAC's chunks are independent.** The encoder's `forward` runs `ceil(L/512)` separate passes of exactly 512 tokens each, with a `(1, 1, 512, 512)` mask:

```
L= 2048: encoder forwards=  4  input_len set=[512]  total tokens=2048
L= 8192: encoder forwards= 16  input_len set=[512]  total tokens=8192
L=32768: encoder forwards= 64  input_len set=[512]  total tokens=32768
```

In the chunk loop `self.model(...)` receives `position_ids`, `inputs_embeds` and `attention_mask` — **never `past_key_values`**. Each chunk's KV is trimmed to its strided slots and appended to a list; it is never fed into the next chunk's forward. The SFT trainer/evaluator import `model.modeling`, the file vendored here. So the encoder is `O(L/chunk × chunk²) = O(chunk · L)` — **linear, with `chunk_size = 512` as the constant**. That chunk size is a trained hyperparameter stored in the checkpoint config, not a benchmark setting.

**The speed then follows from where the FLOPs actually are.** At `L = 245,760` on the 1.7B backbone:

| | dense (MLP + proj) | attention | LoRA | total | measured | achieved |
|---|---|---|---|---|---|---|
| `full_ctx` | 693 TFLOP | **6927 TFLOP** | — | 7620 | 33.2 s | 230 TF/s |
| `sac` | 693 TFLOP | **29 TFLOP** | 13 | 734 | 12.5 s | 59 TF/s |

A full-context prefill at this length is **91 % attention**. Capping attention at 512-token windows cuts that term by 239× and total work by 10.4×. SAC hands most of it back in efficiency — 59 vs 230 TF/s, **3.9× worse utilization**, because it runs 480 sequential small forwards through Python with an explicit 4D mask that defeats the flash-attention path. Net 10.4 / 3.9 = **2.7×**, matching the measured 33.2 / 12.5 = **2.7×**.

**The dense term is identical for both (693 TFLOP)** — direct evidence that no layers are skipped: every token traverses the full 1.7B exactly once either way. SAC spends 11.7 s of its 12.5 s on that unavoidable floor. Quadratic attention is a smaller share of the total than might be expected, and chunking removes it.

#### Our margin over SAC is a constant factor, not an asymptotic one

Running the same accounting on our side (compressor non-embedding params measured from the checkpoint at 1.348 B, plus the KV projector over the `L/8` compressed positions):

| | total FLOPs | measured | achieved |
|---|---|---|---|
| `ssm_kv` (ours) | 670 TFLOP | 4.5 s | **147 TF/s** |
| `sac` | 734 TFLOP | 12.5 s | 59 TF/s |

We do only **1.10× fewer FLOPs** than SAC. The 2.75× speedup decomposes as `1.10 × 2.51` — almost all of it is **2.5× better hardware utilization**, from one packed 245k-token SSM forward through fused Triton kernels versus 480 small Transformer forwards.

Against ICAE (1.84), Beacon (1.60) and full context (1.80), the linear-vs-quadratic gap is real and grows without bound. **Against SAC it is not a complexity argument** — both are linear (1.07 vs 1.02) with comparable constants, and the difference comes from implementation efficiency plus a smaller backbone.

**Scope.** This section describes the SAC *implementation* used here — the one that trained the measured checkpoint — which encodes chunks independently. A single full-context encoder pass would be `O(L²)`.

**Our ceiling remains 252,288 tokens per packed sequence** — `ssm_kv` is the one method with no 262,144 point. mamba_ssm's Triton kernels index the fused projection with int32 at row stride `d_in_proj = 8512`, so a single sequence overflows at `2³¹ / 8512`. It surfaces as an illegal memory access (at the first device→host sync after the backbone call), not as an OOM — 245k peaked at 23.2 GB of 97 GB. Because that fault poisons the CUDA context and kills the process, `config_sota.yaml` schedules `ssm_kv` last and rows stream to disk; the crash loses one cell.

### Three-way: ours vs Activation Beacon vs full context (bf16, ratio 8, 512 → 256k)

`config_beacon.yaml`. At **245,760 tokens** — the largest length all three methods reach:

| Method | ttft | total | peak mem | compressed tokens |
|---|---|---|---|---|
| **`ssm_kv` (ours)** | **4.5 s** | **5.6 s** | 23.8 GB | 24,576 |
| `beacon` | 113.5 s | 116.3 s | **10.8 GB** | 30,720 |
| `full_ctx` | 33.2 s | 37.3 s | 45.6 GB | 245,760 |

Two results matter here.

**Beacon is not linear, and it is slower than not compressing at all.** Its ttft doubles at exactly 2.00× per doubling up to 64k, then breaks: 2.30× at 128k, and a fitted exponent of **1.64** from 128k → 240k. The mechanism is structural — each window attends over all accumulated beacon memory, so the true cost is `O(L/window × L/ratio)`, quadratic with a small constant. Below 64k the per-window fixed cost hides it. At 240k Beacon's compression costs **3.4× more than the quadratic prefill it replaces**. Compression time is not free, and for this method at this scale it does not pay for itself in latency.

**Beacon's real win is memory**: 10.8 GB vs our 23.8 GB at 240k, because it carries one 1.7B backbone while we carry a compressor *and* an LLM, and it discards raw KV. That advantage is genuine and costs 25× in latency.

Ours is the only linear method in the comparison: fitted exponent **1.07** over the same 128k → 240k range. `full_ctx` measures **1.87** — near-quadratic, as expected.

**Our ceiling is 252,288 tokens per packed sequence.** mamba_ssm's Triton kernels index the fused projection with int32, and its row stride is `d_in_proj = 8512`, so a single sequence overflows at `2³¹ / 8512`. Measured: 250,112 passes, 254,208 raises an illegal memory access. This is a kernel indexing limit, not memory or algorithm — 128k peaked at 15.5 GB  — and it is fixable upstream. It is why `ssm_kv` has no 262,144 point. Splitting into two packed segments would dodge it but would give two independent SSM states rather than one continuous compression, so it is not the same method and is not plotted as one.

### Qwen3-1.7B LLM + mamba2-1.3B compressor, KV path, bf16

Checkpoint: Mamba2 1.3B + Qwen3-1.7B, KV path (`<run_dir>`), `ssm_kv` vs `full_ctx`, 512 → 128k, no OOM.

| `L` | ttft `ssm_kv` / `full_ctx` | total `ssm_kv` / `full_ctx` | peak mem `ssm_kv` / `full_ctx` |
|---|---|---|---|
| 32k | 592 / 1181 ms (**2.0×**) | 1658 / 2265 ms (1.4×) | 8.5 / 11.4 GB |
| 64k | 1166 / 3372 ms (**2.9×**) | 2245 / 4711 ms (2.1×) | 10.8 / 16.7 GB |
| 128k | 2286 / 10732 ms (**4.7×**) | 3381 / 13045 ms (**3.9×**) | 15.6 / 27.2 GB (**43 % less**) |

The asymptotics come out of the measurement directly. Per doubling of `L`, `ssm_kv` time-to-first-token goes 592 → 1166 → 2286 ms — exactly 2× each step, i.e. linear in `L`. `full_ctx` goes 1181 → 3372 → 10732 ms — 2.9× then 3.2×, i.e. super-linear.

Compression also helps *decode*: `full_ctx` per-token decode degrades (1084 → 1338 → 2313 ms) because every generated token attends over a 128k KV cache, while `ssm_kv` decode stays flat at ~1080 ms against its 10× smaller cache.

**Caveat:** Qwen3's trained context length is 32768. The 64k and 128k points measure valid *compute*, but generation there is not semantically meaningful — cite them as throughput/scaling evidence only.

### Qwen2.5-0.5B LLM + mamba2-1.3B compressor, embed path

With a 0.5B LLM, `ssm` is slower than `full_ctx` on end-to-end total across the whole range. Two factors contribute:

1. Total is dominated by a fixed ~760 ms decode for 64 tokens, identical for both methods, which dilutes any prefill saving.
2. The compressor (1.3B) is **larger than the LLM** (0.5B), so the SSM forward at 64k (1012 ms) costs about as much as the entire quadratic prefill it replaces (1103 ms).

The compressed path pays off when the LLM is large enough that its prefill dominates the compressor's cost, as in the Qwen3-1.7B results above.

### Why the memory win is asymptotic

At short `L` the compressed path can use *more* memory than full context. Model weights are resident either way (both paths load the same checkpoint: compressor **and** LLM), and `ssm_kv` additionally pays to run the compressor and build the cache. Compression only shrinks KV/attention memory, which is negligible until the context is long. Net memory win = KV saved − compressor overhead: negative below ~2k, decisively positive by 32k.

---

## Extending

Add a new method without touching `runner.py`:

1. Write three functions `<name>_prepare`, `<name>_compress`, `<name>_generate` in `adapters.py`.
2. Register them in the `ADAPTERS` dict with a `loader` and a `compressed_tokens` lambda.
3. Add the name to `adapters` in `config.yaml`.

If the method runs on our checkpoint, reuse `loader: "boa"` and you are done. If it brings its own model (as Beacon, SAC and ICAE do), add one loader function to `bench.py` returning `(model, ctx)` and register it in the `loaders` dict. The core loop calls only the adapter protocol.

Two rules that apply to every method added this way, both worth following:

- **Force full-length decode.** Random benchmark inputs will hit an EOS id and stop early, which makes `t_decode` meaningless. Use `min_new_tokens == max_new_tokens` on HF generation, or an EOS id that can never match an argmax on a hand-written loop.
- **Keep the ratio fixed across the sweep.** A method whose compressed size is a trained constant rather than a function of `L` (ICAE's `n_mem`) must have that constant re-derived per cell, or the comparison silently stops being a comparison at one compression rate. The `C` column in the summary table is the check.

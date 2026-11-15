// Regenerates site/data/latency.json from the benchmark outputs of the paper.
//
//   node site/tools/build-latency.mjs /path/to/resources_boa_ssm/outputs > site/data/latency.json
//
// The argument is the directory that holds results_sota.jsonl, qwen3_1-7b/ and qwen3_4b/.
// Only derived numbers are written; the raw benchmark files are never copied into the repository.
// This script is not run in CI.
//
// Conversions (one row per adapter and sequence length):
//   endToEndS = t_total_ms / 1000
//   contextS  = (t_compress_ms + t_ttft_ms) / 1000
//   memoryGB  = peak_mem_mb / 1024   (the values are GiB; the paper's axis says GB)
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const SIGNIFICANT = 4;
const round = (x) => Number(x.toPrecision(SIGNIFICANT));

export function rowsToMetrics(rows, adapter, lengths) {
  const metrics = { endToEndS: [], contextS: [], memoryGB: [] };
  for (const length of lengths) {
    const found = rows.filter((r) => r.adapter === adapter && r.seq_len === length);
    if (found.length !== 1) throw new Error(`${found.length === 0 ? "Missing" : "Duplicate"} row: ${adapter} at ${length} tokens`);
    const r = found[0];
    if (r.oom) throw new Error(`OOM row: ${adapter} at ${length} tokens`);
    const values = [r.t_total_ms, r.t_compress_ms + r.t_ttft_ms, r.peak_mem_mb];
    if (!values.every((v) => Number.isFinite(v) && v > 0)) throw new Error(`Non-positive value: ${adapter} at ${length} tokens`);
    metrics.endToEndS.push(round(r.t_total_ms / 1000));
    metrics.contextS.push(round((r.t_compress_ms + r.t_ttft_ms) / 1000));
    metrics.memoryGB.push(round(r.peak_mem_mb / 1024));
  }
  return metrics;
}

export function buildLatency(config, readRows) {
  const { lengths, methods, decoders, ratio, generatedTokens, dtype, source, annotatedSpeedup } = config;
  const out = { annotatedSpeedup, ratio, generatedTokens, dtype, source, lengths, methods, decoders: {} };
  for (const decoder of decoders) {
    const entry = { label: decoder.label };
    if (decoder.gpu) entry.gpu = decoder.gpu;
    if (decoder.note) entry.note = decoder.note;
    entry.endToEndS = {};
    entry.contextS = {};
    entry.memoryGB = {};
    for (const { id } of methods) {
      const { file, adapter } = decoder.sources[id];
      const m = rowsToMetrics(readRows(file), adapter, lengths);
      entry.endToEndS[id] = m.endToEndS;
      entry.contextS[id] = m.contextS;
      entry.memoryGB[id] = m.memoryGB;
    }
    out.decoders[decoder.id] = entry;
  }
  return out;
}

const SOTA_17 = "results_sota.jsonl";
const BEACON2048_17 = "qwen3_1-7b/results_sota_1-7b_beacon_w2048.jsonl";
const SOTA_4 = "qwen3_4b/results_sota_4b_beacon_w128.jsonl";
const BEACON2048_4 = "qwen3_4b/results_beacon_4b_w2048.jsonl";
const sources = (sota, beacon2048) => ({
  boa: { file: sota, adapter: "ssm_kv" },
  sac: { file: sota, adapter: "sac" },
  beacon128: { file: sota, adapter: "beacon" },
  beacon2048: { file: beacon2048, adapter: "beacon" },
  full: { file: sota, adapter: "full_ctx" },
});

export const CONFIG = {
  // The paper's own annotation on Figure 3 (Boa vs Activation Beacon w=128 at its longest length).
  annotatedSpeedup: { "qwen3-1.7b": 19, "qwen3-4b": 39 },
  ratio: 8,
  generatedTokens: 64,
  dtype: "bf16",
  source: [
    "Derived from the paper's benchmark outputs, median of 3 timed runs after 1 warmup, 8x compression, 64 generated tokens, bf16.",
    "Qwen3-1.7B: results_sota.jsonl (Boa, SAC, Activation Beacon w=128, full context) and qwen3_1-7b/results_sota_1-7b_beacon_w2048.jsonl (Beacon w=2048), on an NVIDIA RTX PRO 6000 (Blackwell) as stated in the paper's supplementary material.",
    "Qwen3-4B: qwen3_4b/results_sota_4b_beacon_w128.jsonl and qwen3_4b/results_beacon_4b_w2048.jsonl; the GPU is not stated for this decoder.",
    "Qwen3-4B caveats: the SAC run records no checkpoint (sac_checkpoint is null), and the Beacon w=2048 run reuses the w=128 checkpoint with the window overridden.",
    "The Qwen3-4B curves come from a later run than the paper's figure and differ from the published plot by up to about 18%.",
    "End to end = t_total_ms; context processing = t_compress_ms + t_ttft_ms; memory = peak_mem_mb / 1024.",
    "The last length is 245,760 tokens (the paper calls it 240k).",
  ].join(" "),
  lengths: [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 245760],
  methods: [
    { id: "boa", label: "Boa" },
    { id: "sac", label: "SAC" },
    { id: "beacon128", label: "Activation Beacon (w=128)" },
    { id: "beacon2048", label: "Activation Beacon (w=2048)" },
    { id: "full", label: "Full context" },
  ],
  decoders: [
    { id: "qwen3-1.7b", label: "Qwen3-1.7B", gpu: "NVIDIA RTX PRO 6000 (Blackwell)", sources: sources(SOTA_17, BEACON2048_17) },
    {
      id: "qwen3-4b",
      label: "Qwen3-4B",
      note: "These curves come from a later benchmark run than the paper's figure, so the values differ slightly from the published plot.",
      sources: sources(SOTA_4, BEACON2048_4),
    },
  ],
};

const isMain = process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href;
if (isMain) {
  const dir = process.argv[2];
  if (!dir) {
    console.error("usage: node site/tools/build-latency.mjs <outputs-directory> > site/data/latency.json");
    process.exit(1);
  }
  const readRows = (file) => readFileSync(join(dir, file), "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l)).filter((r) => r.adapter);
  // Arrays of numbers stay on one line so the file diffs by series.
  const json = JSON.stringify(buildLatency(CONFIG, readRows), null, 1).replace(/\[\s+([\d.,\s]+?)\s+\]/g, (_, body) => `[${body.replace(/\s*\n\s*/g, " ")}]`);
  process.stdout.write(`${json}\n`);
}

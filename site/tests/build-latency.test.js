import test from "node:test";
import assert from "node:assert/strict";
import { buildLatency, rowsToMetrics } from "../tools/build-latency.mjs";

const row = (adapter, seq_len, over = {}) => ({
  adapter, seq_len, t_compress_ms: 100, t_ttft_ms: 400, t_decode_ms: 1500, t_total_ms: 2000, peak_mem_mb: 2048, oom: false, suspect: false, ...over,
});

test("rowsToMetrics converts ms to s and MiB to GiB, in length order", () => {
  const rows = [row("a", 2048, { t_total_ms: 3000 }), row("a", 1024), row("b", 1024)];
  const m = rowsToMetrics(rows, "a", [1024, 2048]);
  assert.deepEqual(m.endToEndS, [2, 3]);
  assert.deepEqual(m.contextS, [0.5, 0.5]);
  assert.deepEqual(m.memoryGB, [2, 2]);
});

test("rowsToMetrics keeps four significant figures", () => {
  const m = rowsToMetrics([row("a", 1, { t_total_ms: 1234.5678, peak_mem_mb: 1000 })], "a", [1]);
  assert.equal(m.endToEndS[0], 1.235);
  assert.equal(m.memoryGB[0], 0.9766);
});

test("rowsToMetrics rejects a missing length, an OOM row and a negative time", () => {
  assert.throws(() => rowsToMetrics([row("a", 1)], "a", [1, 2]), /missing/i);
  assert.throws(() => rowsToMetrics([row("a", 1, { oom: true })], "a", [1]), /oom/i);
  assert.throws(() => rowsToMetrics([row("a", 1, { t_decode_ms: -5, t_total_ms: -5 })], "a", [1]), /negative|positive/i);
});

test("buildLatency assembles decoders from the configured files", () => {
  const files = {
    "x.jsonl": [row("full_ctx", 1), row("ssm_kv", 1, { t_total_ms: 1000 })],
  };
  const out = buildLatency({
    lengths: [1],
    methods: [{ id: "boa", label: "Boa" }, { id: "full", label: "Full context" }],
    decoders: [{ id: "d", label: "D", gpu: "GPU", sources: { boa: { file: "x.jsonl", adapter: "ssm_kv" }, full: { file: "x.jsonl", adapter: "full_ctx" } } }],
    ratio: 8, generatedTokens: 64, dtype: "bf16", source: "note",
  }, (file) => files[file]);
  assert.deepEqual(out.decoders.d.endToEndS, { boa: [1], full: [2] });
  assert.deepEqual(out.decoders.d.memoryGB, { boa: [2], full: [2] });
  assert.equal(out.decoders.d.gpu, "GPU");
  assert.equal(out.approximate, undefined);
  assert.equal(out.source, "note");
});

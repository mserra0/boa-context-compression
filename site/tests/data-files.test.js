import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { loadData, DataLoadError } from "../js/lib/data.js";
import { speedup, measurement } from "../js/lib/latency.js";

const read = (name) => JSON.parse(readFileSync(new URL(`../data/${name}`, import.meta.url), "utf8"));
const latency = read("latency.json");
const results = read("results.json");
const metrics = ["endToEndS", "contextS", "memoryGB"];

test("latency arrays align with lengths and are positive finite numbers", () => {
  for (const decoder of Object.values(latency.decoders)) {
    for (const metric of metrics) {
      if (!decoder[metric]) continue;
      for (const [id, values] of Object.entries(decoder[metric])) {
        assert.equal(values.length, latency.lengths.length, `${metric}.${id}`);
        assert.ok(values.every((v) => Number.isFinite(v) && v > 0), `${metric}.${id}`);
      }
    }
  }
});

test("every method id is present in every metric object", () => {
  for (const decoder of Object.values(latency.decoders)) {
    for (const metric of metrics) {
      if (!decoder[metric]) continue;
      for (const { id } of latency.methods) assert.ok(id in decoder[metric], `${metric}.${id}`);
    }
  }
});

test("latency.json is measured data, not plot-read values", () => {
  assert.notEqual(latency.approximate, true);
  assert.equal(latency.lengths.at(-1), 245760);
  assert.equal(latency.dtype, "bf16");
  assert.equal(typeof latency.source, "string");
});

test("latency anchors at the longest measured length", () => {
  const last = latency.lengths.at(-1);
  const at = (decoder) => speedup(latency, { decoder, metric: "endToEndS", length: last, method: "boa", baseline: "beacon128" });
  assert.ok(at("qwen3-1.7b") >= 20 && at("qwen3-1.7b") <= 21);
  assert.ok(at("qwen3-4b") >= 37 && at("qwen3-4b") <= 38);
  const mem = (decoder, method) => measurement(latency, { decoder, metric: "memoryGB", method, length: last });
  assert.ok(mem("qwen3-1.7b", "full") >= 44 && mem("qwen3-1.7b", "full") <= 45);
  assert.ok(mem("qwen3-4b", "full") >= 62 && mem("qwen3-4b", "full") <= 63);
  assert.ok(mem("qwen3-4b", "boa") >= 27 && mem("qwen3-4b", "boa") <= 28);
});

test("both decoders have complete memory and context-processing data", () => {
  for (const decoder of Object.values(latency.decoders)) {
    for (const metric of ["contextS", "memoryGB"]) {
      for (const { id } of latency.methods) assert.equal(decoder[metric]?.[id]?.length, latency.lengths.length, `${decoder.label} ${metric}.${id}`);
    }
  }
});

test("results.json holds the paper reference values", () => {
  assert.equal(results.inDomain.byRatio["16"].boa.normF1, 94.91);
  assert.equal(results.inDomain.full.f1, 83.74);
  assert.equal(results.outOfDomain.byRatio["16"].boa.f1, 61.91);
  for (const ratio of results.ratios) {
    for (const { id } of results.methods) {
      assert.ok(results.inDomain.byRatio[ratio][id], `inDomain ${ratio} ${id}`);
      assert.ok(results.outOfDomain.byRatio[ratio][id], `ood ${ratio} ${id}`);
    }
  }
});

test("loadData rejects with DataLoadError carrying the url", async () => {
  const original = globalThis.fetch;
  globalThis.fetch = async () => ({ ok: false });
  try {
    await assert.rejects(loadData("data/"), (err) => {
      assert.ok(err instanceof DataLoadError);
      assert.equal(err.url, "data/results.json");
      return true;
    });
  } finally {
    globalThis.fetch = original;
  }
});

test("latency.json holds the speedups the paper reports", () => {
  assert.deepEqual(latency.annotatedSpeedup, { "qwen3-1.7b": 19, "qwen3-4b": 39 });
  for (const id of Object.keys(latency.annotatedSpeedup)) assert.ok(id in latency.decoders, id);
});

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

test("latency sanity anchors at 240000 tokens", () => {
  const at = (decoder) => speedup(latency, { decoder, metric: "endToEndS", length: 240000, method: "boa", baseline: "beacon128" });
  assert.ok(at("qwen3-1.7b") >= 18 && at("qwen3-1.7b") <= 20);
  assert.ok(at("qwen3-4b") >= 38 && at("qwen3-4b") <= 40);
  const mem = measurement(latency, { decoder: "qwen3-1.7b", metric: "memoryGB", method: "full", length: 240000 });
  assert.ok(mem >= 45 && mem <= 49);
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

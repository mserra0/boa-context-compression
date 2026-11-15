import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { headlines } from "../js/lib/headlines.js";

const read = (name) => JSON.parse(readFileSync(new URL(`../data/${name}`, import.meta.url), "utf8"));
const results = read("results.json");
const latency = read("latency.json");

test("retained F1 at the highest ratio is rounded from the data", () => {
  const h = headlines(results, latency);
  assert.equal(h.retained.ratio, 16);
  assert.equal(h.retained.percent, 95);
});

test("EM loss runs from the lowest to the highest ratio", () => {
  const h = headlines(results, latency);
  assert.equal(h.emDrop.from, 4);
  assert.equal(h.emDrop.to, 16);
  assert.equal(h.emDrop.points, 1.4);
});

test("speedups are the paper's annotated factors while latency is approximate", () => {
  const h = headlines(results, latency);
  assert.deepEqual(h.speedups.map((s) => [s.decoder, s.factor]), [["Qwen3-1.7B", 19], ["Qwen3-4B", 39]]);
  assert.equal(h.speedups[0].tokens, 240000);
});

test("speedups fall back to the plotted values when nothing is annotated", () => {
  const plain = { ...latency };
  delete plain.annotatedSpeedup;
  const h = headlines(results, plain);
  assert.ok(h.speedups.every((s) => Number.isFinite(s.factor)));
});

test("a decoder without the baseline is left out", () => {
  const sparse = { ...latency, annotatedSpeedup: undefined, decoders: { a: { label: "A", endToEndS: { boa: [1], beacon128: [2] } } }, lengths: [240000] };
  const h = headlines(results, sparse);
  assert.equal(h.speedups.length, 1);
  assert.equal(h.speedups[0].factor, 2);
});

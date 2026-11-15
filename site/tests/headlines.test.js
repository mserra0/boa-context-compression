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

test("speedups are the paper's reported factors, at the longest measured length", () => {
  const h = headlines(results, latency);
  assert.deepEqual(h.speedups.map((s) => [s.decoder, s.factor, s.reported]), [["Qwen3-1.7B", 19, true], ["Qwen3-4B", 39, true]]);
  assert.equal(h.speedups[0].tokens, 245760);
});

test("speedups are computed from the data when the paper reports none", () => {
  const plain = { ...latency };
  delete plain.annotatedSpeedup;
  const h = headlines(results, plain);
  assert.deepEqual(h.speedups.map((s) => [s.factor, s.reported]), [[21, false], [37, false]]);
});

test("the headline length comes from the data, not a constant", () => {
  const short = { ...latency, lengths: [512, 1024], annotatedSpeedup: undefined, decoders: { a: { label: "A", endToEndS: { boa: [1, 1], beacon128: [2, 6] } } } };
  const h = headlines(results, short);
  assert.equal(h.speedups[0].tokens, 1024);
  assert.equal(h.speedups[0].factor, 6);
});

test("a decoder without the baseline is left out", () => {
  const sparse = { ...latency, annotatedSpeedup: undefined, decoders: { a: { label: "A", endToEndS: { boa: [1], beacon128: [2] } } }, lengths: [240000] };
  const h = headlines(results, sparse);
  assert.equal(h.speedups.length, 1);
  assert.equal(h.speedups[0].factor, 2);
});

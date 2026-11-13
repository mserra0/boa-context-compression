import test from "node:test";
import assert from "node:assert/strict";
import { formatSeconds, formatGigabytes, formatTokens, formatFactor } from "../js/lib/format.js";

test("formatSeconds", () => {
  assert.equal(formatSeconds(0.082), "82 ms");
  assert.equal(formatSeconds(1.234), "1.2 s");
  assert.equal(formatSeconds(12.6), "13 s");
  assert.equal(formatSeconds(125), "125 s");
});

test("formatGigabytes", () => {
  assert.equal(formatGigabytes(6.04), "6.0 GB");
  assert.equal(formatGigabytes(47), "47 GB");
});

test("formatTokens", () => {
  assert.equal(formatTokens(512), "512");
  assert.equal(formatTokens(1024), "1k");
  assert.equal(formatTokens(131072), "128k");
  assert.equal(formatTokens(240000), "240k");
});

test("formatFactor", () => {
  assert.equal(formatFactor(19.3), "19×");
  assert.equal(formatFactor(1.84), "1.8×");
});

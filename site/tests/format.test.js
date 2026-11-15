import test from "node:test";
import assert from "node:assert/strict";
import { formatSeconds, formatGigabytes, formatTokens, formatTokensExact, formatFactor, ordinal } from "../js/lib/format.js";

test("formatSeconds", () => {
  assert.equal(formatSeconds(0.082), "82 ms");
  assert.equal(formatSeconds(1.234), "1.2 s");
  assert.equal(formatSeconds(12.6), "13 s");
  assert.equal(formatSeconds(125), "125 s");
});

test("formatSeconds does not round up into the next unit", () => {
  assert.equal(formatSeconds(0.9996), "1.0 s");
  assert.equal(formatSeconds(0.999), "999 ms");
  assert.equal(formatSeconds(9.96), "10 s");
  assert.equal(formatSeconds(9.94), "9.9 s");
});

test("formatGigabytes", () => {
  assert.equal(formatGigabytes(6.04), "6.0 GB");
  assert.equal(formatGigabytes(47), "47 GB");
  assert.equal(formatGigabytes(9.96), "10 GB");
});

test("formatTokens", () => {
  assert.equal(formatTokens(512), "512");
  assert.equal(formatTokens(1024), "1k");
  assert.equal(formatTokens(131072), "128k");
  assert.equal(formatTokens(240000), "240k");
  assert.equal(formatTokens(245760), "246k");
});

test("formatTokensExact", () => {
  assert.equal(formatTokensExact(245760), "245,760");
  assert.equal(formatTokensExact(131072), "131,072");
  assert.equal(formatTokensExact(512), "512");
});

test("ordinal", () => {
  assert.deepEqual([1, 2, 3, 4, 8, 11, 12, 13, 16, 21, 22].map(ordinal), ["1st", "2nd", "3rd", "4th", "8th", "11th", "12th", "13th", "16th", "21st", "22nd"]);
});

test("formatFactor", () => {
  assert.equal(formatFactor(19.3), "19×");
  assert.equal(formatFactor(1.84), "1.8×");
  assert.equal(formatFactor(9.96), "10×");
});

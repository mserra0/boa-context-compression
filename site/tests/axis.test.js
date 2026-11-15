import test from "node:test";
import assert from "node:assert/strict";
import { labelStep, logTicks, niceLinear, niceLogDomain } from "../js/lib/axis.js";

test("niceLogDomain widens to the nearest 1, 2 or 5 times a power of ten", () => {
  assert.deepEqual(niceLogDomain(1.15, 110), [1, 200]);
  assert.deepEqual(niceLogDomain(1.5, 230), [1, 500]);
  assert.deepEqual(niceLogDomain(0.02, 36), [0.02, 50]);
  assert.deepEqual(niceLogDomain(3.9, 47), [2, 50]);
  assert.deepEqual(niceLogDomain(2, 50), [2, 50]);
});

test("niceLogDomain never returns an empty domain", () => {
  const [lo, hi] = niceLogDomain(5, 5);
  assert.ok(lo <= 5 && hi >= 5 && lo < hi);
});

test("logTicks lists 1-2-5 values, or decades when that would be too many", () => {
  assert.deepEqual(logTicks(1, 200), [1, 2, 5, 10, 20, 50, 100, 200]);
  assert.deepEqual(logTicks(1, 500), [1, 2, 5, 10, 20, 50, 100, 200, 500]);
  assert.deepEqual(logTicks(0.01, 1000), [0.01, 0.1, 1, 10, 100, 1000]);
  assert.deepEqual(logTicks(2, 50), [2, 5, 10, 20, 50]);
});

test("niceLinear rounds the domain out to a 1, 2 or 5 step", () => {
  assert.deepEqual(niceLinear(46.27, 100), {domain: [40, 100], ticks: [40, 60, 80, 100]});
  assert.deepEqual(niceLinear(0, 1), {domain: [0, 1], ticks: [0, 0.2, 0.4, 0.6, 0.8, 1]});
});

test("niceLinear never returns an empty domain", () => {
  const {domain: [lo, hi], ticks} = niceLinear(5, 5);
  assert.ok(lo <= 5 && hi >= 5 && lo < hi);
  assert.ok(ticks.length >= 2);
});

test("labelStep labels every nth tick so labels do not touch", () => {
  assert.equal(labelStep(84, 35), 1);
  assert.equal(labelStep(30, 35), 2);
  assert.equal(labelStep(10, 35), 4);
  assert.equal(labelStep(0, 35), 1);
});

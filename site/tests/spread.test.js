import test from "node:test";
import assert from "node:assert/strict";
import { spreadLabels } from "../js/lib/spread.js";

test("labels that do not overlap stay where they are", () => {
  assert.deepEqual(spreadLabels([10, 50, 90], 12, [0, 100]), [10, 50, 90]);
});

test("overlapping labels move apart and keep their order, returned in input order", () => {
  assert.deepEqual(spreadLabels([50, 52, 51], 10, [0, 100]), [50, 70, 60]);
});

test("labels pushed past the bottom move back up", () => {
  assert.deepEqual(spreadLabels([95, 98], 10, [0, 100]), [90, 100]);
});

test("every gap holds and every label stays in bounds when there is room", () => {
  const out = spreadLabels([60, 61, 62, 63, 64], 12, [10, 70]);
  const sorted = [...out].sort((a, b) => a - b);
  for (let i = 1; i < sorted.length; i++) assert.ok(sorted[i] - sorted[i - 1] >= 12 - 1e-9);
  assert.ok(sorted[0] >= 10 && sorted.at(-1) <= 70);
});

test("an empty list stays empty", () => {
  assert.deepEqual(spreadLabels([], 10, [0, 100]), []);
});

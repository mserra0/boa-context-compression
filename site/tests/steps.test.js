import test from "node:test";
import assert from "node:assert/strict";
import { stepAt } from "../js/lib/steps.js";

const STARTS = [100, 400, 700, 1000, 1300];

test("stepAt counts the steps whose start has been reached", () => {
  assert.equal(stepAt(STARTS, 450), 2);
  assert.equal(stepAt(STARTS, 1299), 4);
  assert.equal(stepAt(STARTS, 5000), 5);
});

test("stepAt makes a step active exactly at its start", () => {
  assert.equal(stepAt(STARTS, 700), 3);
  assert.equal(stepAt(STARTS, 699.5), 2);
});

test("stepAt keeps the first step active before any start is reached", () => {
  assert.equal(stepAt(STARTS, 0), 1);
  assert.equal(stepAt(STARTS, -50), 1);
});

test("stepAt does not depend on the order of the starts", () => {
  assert.equal(stepAt([700, 100, 400], 450), 2);
});

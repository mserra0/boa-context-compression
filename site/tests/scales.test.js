import test from "node:test";
import assert from "node:assert/strict";
import { linearScale, logScale } from "../js/lib/scales.js";

test("linearScale maps domain to range", () => {
  assert.equal(linearScale([0, 10], [0, 100])(5), 50);
  assert.equal(linearScale([0, 10], [100, 0])(2.5), 75);
});

test("logScale is base 10", () => {
  assert.ok(Math.abs(logScale([1, 100], [0, 200])(10) - 100) < 1e-9);
  assert.equal(logScale([1, 1000], [0, 300])(1), 0);
});

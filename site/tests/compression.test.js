import test from "node:test";
import assert from "node:assert/strict";
import { compressedLength, keptIndices } from "../js/lib/compression.js";

test("compressedLength rounds up", () => {
  assert.equal(compressedLength(16, 4), 4);
  assert.equal(compressedLength(17, 4), 5);
  assert.equal(compressedLength(3, 16), 1);
});

test("keptIndices is 1-based, ascending and ends at length", () => {
  assert.deepEqual(keptIndices(16, 4), [4, 8, 12, 16]);
  assert.deepEqual(keptIndices(10, 4), [2, 6, 10]);
  assert.deepEqual(keptIndices(3, 16), [3]);
  assert.equal(keptIndices(24, 8).length, compressedLength(24, 8));
});

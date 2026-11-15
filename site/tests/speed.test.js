import test from "node:test";
import assert from "node:assert/strict";
import { speedSentence } from "../js/lib/speed.js";

const SAME = "At this length Boa and the full context take about the same time.";

test("a clear speed-up says how much faster Boa answers", () => {
  assert.equal(speedSentence(3.53), "Boa answers 3.5× faster than reading the full context.");
  assert.equal(speedSentence(1.083), "Boa answers 1.1× faster than reading the full context.");
  assert.equal(speedSentence(1.05), "Boa answers 1.1× faster than reading the full context.");
});

test("a factor close to 1 on either side reads as about the same", () => {
  assert.equal(speedSentence(1.049), SAME);
  assert.equal(speedSentence(1), SAME);
  assert.equal(speedSentence(0.958), SAME);
});

test("a slowdown beyond that band is said plainly", () => {
  assert.equal(speedSentence(0.94), "At this length Boa is 1.1× slower than reading the full context.");
  assert.equal(speedSentence(0.5), "At this length Boa is 2.0× slower than reading the full context.");
});

test("a missing or invalid factor gives no sentence", () => {
  for (const factor of [null, undefined, NaN, Infinity, 0, -1]) assert.equal(speedSentence(factor), null);
});

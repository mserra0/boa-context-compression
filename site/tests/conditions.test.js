import test from "node:test";
import assert from "node:assert/strict";
import { conditionsText } from "../js/lib/conditions.js";

const latency = {
  ratio: 8, generatedTokens: 64, dtype: "bf16",
  decoders: { a: { label: "A", gpu: "GPU X" }, b: { label: "B", note: "A later run." }, c: { label: "C" } },
};

test("conditions come from the data, GPU included when stated", () => {
  assert.equal(conditionsText(latency, "a"), "Measured at 8× compression, 64 generated tokens, bf16, GPU X.");
});

test("an unstated GPU is left out and the decoder note follows", () => {
  assert.equal(conditionsText(latency, "b"), "Measured at 8× compression, 64 generated tokens, bf16. A later run.");
});

test("missing fields degrade to what is known", () => {
  assert.equal(conditionsText({ decoders: { c: {} } }, "c"), "");
});

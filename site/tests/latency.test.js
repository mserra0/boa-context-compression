import test from "node:test";
import assert from "node:assert/strict";
import { measurement, speedup } from "../js/lib/latency.js";

const fx = {
  lengths: [512, 1024],
  decoders: {
    a: { endToEndS: { boa: [1, 2], full: [1, 6] }, memoryGB: { boa: [1, 1], full: [2, 2] } },
    b: { endToEndS: { boa: [1, 2], full: [1, 6] } },
  },
};

test("measurement returns the aligned value or null", () => {
  assert.equal(measurement(fx, { decoder: "a", metric: "endToEndS", method: "boa", length: 1024 }), 2);
  assert.equal(measurement(fx, { decoder: "b", metric: "memoryGB", method: "boa", length: 512 }), null);
  assert.equal(measurement(fx, { decoder: "a", metric: "endToEndS", method: "boa", length: 700 }), null);
  assert.equal(measurement(fx, { decoder: "zzz", metric: "endToEndS", method: "boa", length: 512 }), null);
});

test("speedup is baseline over method", () => {
  assert.equal(speedup(fx, { decoder: "a", metric: "endToEndS", length: 1024, method: "boa", baseline: "full" }), 3);
  assert.equal(speedup(fx, { decoder: "b", metric: "memoryGB", length: 512, method: "boa", baseline: "full" }), null);
});

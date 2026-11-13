import test from "node:test";
import assert from "node:assert/strict";
import { mountFigures } from "../js/lib/mount.js";

test("mountFigures isolates a throwing figure and skips a missing root", () => {
  const calls = [];
  const errors = [];
  const figures = [
    {id: "a", mount: () => calls.push("a")},
    {id: "b", mount: () => { throw new Error("boom"); }},
    {id: "c", mount: () => calls.push("c")},
    {id: "missing", mount: () => calls.push("missing")},
  ];
  const mounted = mountFigures(figures, {}, {
    getRoot: (id) => (id === "missing" ? null : {id}),
    onError: (id, error) => errors.push([id, error.message]),
  });
  assert.deepEqual(calls, ["a", "c"]);
  assert.deepEqual(errors, [["b", "boom"]]);
  assert.deepEqual(mounted, ["a", "c"]);
});

test("mountFigures passes the root and the data to each mount", () => {
  const seen = [];
  const data = {results: {}, latency: {}};
  mountFigures([{id: "a", mount: (root, d) => seen.push([root, d])}], data, {
    getRoot: (id) => ({id}),
    onError: () => {},
  });
  assert.deepEqual(seen, [[{id: "a"}, data]]);
});

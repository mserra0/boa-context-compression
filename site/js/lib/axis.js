// Axis helpers for the SVG charts: nice domains, tick values and label thinning.

const MANTISSAS = [1, 2, 5];

// Removes floating-point noise such as 0.30000000000000004.
function clean(x) {
  return Number(x.toPrecision(12));
}

function niceValues(lo, hi) {
  const values = [];
  for (let k = Math.floor(Math.log10(lo)) - 1; k <= Math.ceil(Math.log10(hi)) + 1; k++) {
    for (const m of MANTISSAS) values.push(clean(m * 10 ** k));
  }
  return values;
}

// Widens [min, max] (both > 0) to the nearest values of the form 1, 2 or 5 times a power of ten.
export function niceLogDomain(min, max) {
  const values = niceValues(min, max);
  const lo = values.filter((v) => v <= min).at(-1);
  let hi = values.find((v) => v >= max);
  if (hi <= lo) hi = values.find((v) => v > lo);
  return [lo, hi];
}

// Ticks at 1, 2 and 5 times each power of ten inside [lo, hi]; only the powers of ten
// when that would give more than `maxCount` ticks.
export function logTicks(lo, hi, maxCount = 9) {
  const inside = niceValues(lo, hi).filter((v) => v >= lo && v <= hi);
  if (inside.length <= maxCount) return inside;
  return inside.filter((v) => clean(10 ** Math.round(Math.log10(v))) === v);
}

// Rounds [min, max] out to a step of 1, 2 or 5 times a power of ten, with about `count` steps.
export function niceLinear(min, max, count = 5) {
  const span = max - min || Math.abs(max) || 1;
  const raw = span / count;
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const step = clean(magnitude * ([1, 2, 5].find((m) => raw / magnitude <= m) ?? 10));
  let lo = clean(Math.floor(min / step) * step);
  let hi = clean(Math.ceil(max / step) * step);
  if (lo === hi) { lo = clean(lo - step); hi = clean(hi + step); }
  const ticks = [];
  for (let i = 0; clean(lo + i * step) <= hi; i++) ticks.push(clean(lo + i * step));
  return {domain: [lo, hi], ticks};
}

// Labels every nth tick, where ticks are `spacing` apart and a label needs `labelWidth`.
export function labelStep(spacing, labelWidth) {
  if (!(spacing > 0)) return 1;
  return Math.max(1, Math.ceil(labelWidth / spacing));
}

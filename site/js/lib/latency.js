export function measurement(data, { decoder, metric, method, length }) {
  const index = data.lengths.indexOf(length);
  if (index === -1) return null;
  const value = data.decoders?.[decoder]?.[metric]?.[method]?.[index];
  return Number.isFinite(value) ? value : null;
}

export function speedup(data, { decoder, metric, length, method, baseline }) {
  const base = measurement(data, { decoder, metric, method: baseline, length });
  const value = measurement(data, { decoder, metric, method, length });
  if (base === null || value === null || value === 0) return null;
  return base / value;
}

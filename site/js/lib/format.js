export function formatSeconds(seconds) {
  if (seconds < 1) return `${Math.round(seconds * 1000)} ms`;
  if (seconds < 10) return `${seconds.toFixed(1)} s`;
  return `${Math.round(seconds)} s`;
}

export function formatGigabytes(gb) {
  return gb < 10 ? `${gb.toFixed(1)} GB` : `${Math.round(gb)} GB`;
}

export function formatTokens(n) {
  if (n < 1000) return String(n);
  if (n % 1024 === 0) return `${n / 1024}k`;
  return `${Math.round(n / 1000)}k`;
}

export function formatFactor(x) {
  return x < 10 ? `${x.toFixed(1)}×` : `${Math.round(x)}×`;
}

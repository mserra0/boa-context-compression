// Each formatter rounds first and then picks the unit, so a value just below a boundary
// never prints as the boundary in the smaller unit ("1000 ms", "10.0 s").
export function formatSeconds(seconds) {
  if (seconds < 1) {
    const ms = Math.round(seconds * 1000);
    if (ms < 1000) return `${ms} ms`;
  }
  if (seconds < 10) {
    const rounded = seconds.toFixed(1);
    if (rounded !== "10.0") return `${rounded} s`;
  }
  return `${Math.round(seconds)} s`;
}

export function formatGigabytes(gb) {
  if (gb < 10 && gb.toFixed(1) !== "10.0") return `${gb.toFixed(1)} GB`;
  return `${Math.round(gb)} GB`;
}

// Powers of two read as binary thousands ("128k" for 131,072); anything else rounds to decimal ("246k" for 245,760).
export function formatTokens(n) {
  if (n < 1000) return String(n);
  if (Number.isInteger(Math.log2(n))) return `${n / 1024}k`;
  return `${Math.round(n / 1000)}k`;
}

export function formatTokensExact(n) {
  return n.toLocaleString("en-US");
}

export function formatFactor(x) {
  if (x < 10 && x.toFixed(1) !== "10.0") return `${x.toFixed(1)}×`;
  return `${Math.round(x)}×`;
}

export function ordinal(n) {
  const lastTwo = n % 100;
  if (lastTwo >= 11 && lastTwo <= 13) return `${n}th`;
  return `${n}${{ 1: "st", 2: "nd", 3: "rd" }[n % 10] ?? "th"}`;
}

// Moves label positions apart so that neighbours are at least `gap` apart, keeping their order
// and staying inside [min, max] where there is room. Returns positions in the input order.
export function spreadLabels(positions, gap, [min, max]) {
  const order = positions.map((p, i) => i).sort((a, b) => positions[a] - positions[b]);
  const placed = order.map((i) => positions[i]);
  for (let k = 0; k < placed.length; k++) {
    if (k === 0) placed[k] = Math.max(placed[k], min);
    else placed[k] = Math.max(placed[k], placed[k - 1] + gap);
  }
  for (let k = placed.length - 1; k >= 0; k--) {
    const limit = k === placed.length - 1 ? max : placed[k + 1] - gap;
    placed[k] = Math.min(placed[k], limit);
  }
  const out = new Array(positions.length);
  order.forEach((i, k) => { out[i] = placed[k]; });
  return out;
}

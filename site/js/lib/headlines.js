import { speedup } from "./latency.js";

const SPEEDUP_TOKENS = 240000;
const SPEEDUP_METHOD = "boa";
const SPEEDUP_BASELINE = "beacon128";

// Rounds to cents first so 1.35 (stored as 1.3499...) becomes 1.4, not 1.3.
const oneDecimal = (x) => Math.round(Math.round(x * 100) / 10) / 10;

// The three headline numbers, with the condition each one holds under.
export function headlines(results, latency) {
  const ratios = results.ratios;
  const low = ratios[0];
  const high = ratios.at(-1);
  const inDomain = results.inDomain.byRatio;

  const speedups = [];
  for (const [id, decoder] of Object.entries(latency.decoders)) {
    // While the latency data is read from a plot, the paper's own annotated factors are used,
    // so the page never contradicts the figure it comes from.
    const annotated = latency.annotatedSpeedup?.[id];
    const plotted = speedup(latency, { decoder: id, metric: "endToEndS", length: SPEEDUP_TOKENS, method: SPEEDUP_METHOD, baseline: SPEEDUP_BASELINE });
    const factor = Number.isFinite(annotated) ? annotated : plotted === null ? null : Math.round(plotted);
    if (factor !== null) speedups.push({ decoder: decoder.label, factor, tokens: SPEEDUP_TOKENS });
  }

  return {
    retained: { ratio: high, percent: Math.round(inDomain[high].boa.normF1) },
    emDrop: { from: low, to: high, points: oneDecimal(inDomain[low].boa.em - inDomain[high].boa.em) },
    speedups,
  };
}

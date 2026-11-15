import { speedup } from "./latency.js";

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

  // The longest measured length, whatever the data holds.
  const tokens = latency.lengths.at(-1);
  const speedups = [];
  for (const [id, decoder] of Object.entries(latency.decoders)) {
    // The paper's own factors win whenever the data carries them; the page labels them as reported.
    const annotated = latency.annotatedSpeedup?.[id];
    const measured = speedup(latency, { decoder: id, metric: "endToEndS", length: tokens, method: SPEEDUP_METHOD, baseline: SPEEDUP_BASELINE });
    if (Number.isFinite(annotated)) speedups.push({ decoder: decoder.label, factor: annotated, tokens, reported: true });
    else if (measured !== null) speedups.push({ decoder: decoder.label, factor: Math.round(measured), tokens, reported: false });
  }

  return {
    retained: { ratio: high, percent: Math.round(inDomain[high].boa.normF1) },
    emDrop: { from: low, to: high, points: oneDecimal(inDomain[low].boa.em - inDomain[high].boa.em) },
    speedups,
  };
}

import { formatFactor } from "./format.js";

// Within this factor either way, Boa and the full context take about the same time.
const SAME_BAND = 1.05;

// The read-out sentence for a speed-up of Boa over the full context (full time / Boa time).
export function speedSentence(factor) {
  if (!Number.isFinite(factor) || factor <= 0) return null;
  if (factor >= SAME_BAND) return `Boa answers ${formatFactor(factor)} faster than reading the full context.`;
  if (factor > 1 / SAME_BAND) return "At this length Boa and the full context take about the same time.";
  return `At this length Boa is ${formatFactor(1 / factor)} slower than reading the full context.`;
}

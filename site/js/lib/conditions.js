// "Measured at 8× compression, ..." from the data; a part the data does not state is left out,
// and a note on the decoder (for example, that its run is newer than the paper's figure) follows.
export function conditionsText(latency, decoderId) {
  const decoder = latency.decoders[decoderId];
  const parts = [
    latency.ratio ? `${latency.ratio}× compression` : null,
    latency.generatedTokens ? `${latency.generatedTokens} generated tokens` : null,
    latency.dtype ?? null,
    decoder.gpu ?? null,
  ].filter(Boolean);
  const text = parts.length ? `Measured at ${parts.join(", ")}.` : "";
  return [text, decoder.note].filter(Boolean).join(" ");
}

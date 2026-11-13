export function compressedLength(length, ratio) {
  return Math.ceil(length / ratio);
}

// 1-based positions of the tokens kept after compression; always ends at `length`.
export function keptIndices(length, ratio) {
  const count = compressedLength(length, ratio);
  const indices = [];
  for (let i = count - 1; i >= 0; i--) indices.push(length - i * ratio);
  return indices;
}

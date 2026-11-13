import { el } from "../lib/svg.js";

// Left edge of the token at a 1-based index in a row starting at `x`.
export function tokenLeft(index, {x = 0, size, gap}) {
  return x + (index - 1) * (size + gap);
}

// Total width of a row of `count` tokens.
export function rowWidth(count, {size, gap}) {
  return count * size + (count - 1) * gap;
}

// A row of square tokens. Each rect carries its 1-based position in data-index.
export function tokenRow(count, {x = 0, y = 0, size, gap, className = "token"}) {
  const group = el("g");
  for (let index = 1; index <= count; index++) {
    group.append(el("rect", {
      class: className,
      x: tokenLeft(index, {x, size, gap}),
      y,
      width: size,
      height: size,
      rx: size * 0.18,
      "data-index": index,
    }));
  }
  return group;
}

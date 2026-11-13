const SVG_NS = "http://www.w3.org/2000/svg";

// Creates an SVG element: el("rect", {x: 0, width: 10}, [child, "text"]).
export function el(name, attrs = {}, children = []) {
  const node = document.createElementNS(SVG_NS, name);
  for (const [key, value] of Object.entries(attrs)) {
    if (value !== undefined && value !== null) node.setAttribute(key, String(value));
  }
  for (const child of children) {
    node.append(typeof child === "string" ? document.createTextNode(child) : child);
  }
  return node;
}

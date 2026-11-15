import { el } from "../lib/svg.js";
import { prefersReducedMotion } from "../lib/motion.js";
import { compressedLength, keptIndices } from "../lib/compression.js";
import { ordinal } from "../lib/format.js";
import { rowWidth, tokenLeft, tokenRow } from "./tokens.js";

// Diagram parameters, not measurements.
const CONTEXT_TOKENS = 48;
const RATIOS = [4, 8, 16];
const DEFAULT_RATIO = 8;
const DIMMED = 0.3;

const WIDE_QUERY = "(min-width: 56em)";

// Geometry in viewBox units: the context row on top, the packed soft tokens below.
const LAYOUTS = {
  wide: {width: 960, height: 140, token: {size: 15, gap: 4}, font: 15, contextLabelY: 15, rowY: 26, softLabelY: 108, softY: 120},
  narrow: {width: 344, height: 98, token: {size: 5.5, gap: 1.5}, font: 12, contextLabelY: 12, rowY: 20, softLabelY: 75, softY: 86},
};

function readout(ratio) {
  return `${CONTEXT_TOKENS} context tokens → ${compressedLength(CONTEXT_TOKENS, ratio)} soft tokens`;
}

function describe(ratio) {
  return `Of ${CONTEXT_TOKENS} context tokens, every ${ordinal(ratio)}, counting back from the last, is kept: ${compressedLength(CONTEXT_TOKENS, ratio)} soft tokens.`;
}

function buildScene(layout) {
  const L = layout;
  const row = {x: (L.width - rowWidth(CONTEXT_TOKENS, L.token)) / 2, ...L.token};
  const svg = el("svg", {viewBox: `0 0 ${L.width} ${L.height}`, class: `ratio ratio--${L.name}`, role: "img"});
  const label = (content, y) => el("text", {x: row.x, y, class: "diagram-label diagram-label--muted", "font-size": L.font}, [content]);
  const context = tokenRow(CONTEXT_TOKENS, {...row, y: L.rowY, className: "token token--context"});
  svg.append(label("Context tokens", L.contextLabelY), label("Soft tokens", L.softLabelY), context);
  return {svg, layout: L, row, tokens: [...context.children], state: null};
}

// Links and soft tokens for one ratio, at their final positions.
function buildState(scene, ratio) {
  const {layout: L, row} = scene;
  const kept = keptIndices(CONTEXT_TOKENS, ratio);
  const softX = (L.width - rowWidth(kept.length, row)) / 2;
  const links = el("g");
  const soft = el("g");
  const moves = kept.map((index, j) => {
    const from = tokenLeft(index, row);
    const to = tokenLeft(j + 1, {...row, x: softX});
    const half = row.size / 2;
    const link = el("line", {x1: from + half, y1: L.rowY + row.size, x2: to + half, y2: L.softY, class: "ratio__link"});
    const token = el("rect", {x: to, y: L.softY, width: row.size, height: row.size, rx: row.size * 0.18, class: "token token--soft"});
    links.append(link);
    soft.append(token);
    return {token, link, dx: from - to, dy: L.rowY - L.softY};
  });
  return {group: el("g", {}, [links, soft]), kept: new Set(kept), moves};
}

function cssColour(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function showState(scene, ratio, gsap) {
  const previous = scene.state;
  const next = buildState(scene, ratio);
  scene.state = next;
  scene.svg.setAttribute("aria-label", describe(ratio));
  scene.svg.append(next.group);
  for (const stale of [...scene.svg.querySelectorAll(".ratio__leaving")]) stale.remove();

  if (!gsap) {
    previous?.group.remove();
    for (const token of scene.tokens) {
      token.style.opacity = next.kept.has(Number(token.dataset.index)) ? "1" : String(DIMMED);
    }
    return;
  }

  scene.tl?.kill();
  const tl = gsap.timeline();
  scene.tl = tl;
  if (previous) {
    previous.group.classList.add("ratio__leaving");
    tl.to(previous.group, {opacity: 0, duration: 0.2, ease: "none", onComplete: () => previous.group.remove()}, 0);
  }
  for (const token of scene.tokens) {
    tl.to(token, {opacity: next.kept.has(Number(token.dataset.index)) ? 1 : DIMMED, duration: 0.3, ease: "power1.out"}, 0);
  }
  next.moves.forEach(({token, link, dx, dy}, j) => {
    tl.fromTo(token,
      {x: dx, y: dy, fill: cssColour("--context"), stroke: cssColour("--ink")},
      {x: 0, y: 0, fill: cssColour("--kv"), stroke: cssColour("--kv-stroke"), duration: 0.55, ease: "power2.inOut"},
      0.1 + j * 0.02);
    tl.fromTo(link, {opacity: 0}, {opacity: 1, duration: 0.25, ease: "none"}, 0.5 + j * 0.02);
  });
}

function radioGroup(name, onChange) {
  const fieldset = document.createElement("fieldset");
  fieldset.className = "choice-group";
  const legend = document.createElement("legend");
  legend.textContent = "Compression ratio";
  const options = document.createElement("div");
  options.className = "choice-group__options";
  for (const ratio of RATIOS) {
    const label = document.createElement("label");
    label.className = "choice";
    const input = document.createElement("input");
    input.type = "radio";
    input.name = name;
    input.value = String(ratio);
    input.checked = ratio === DEFAULT_RATIO;
    label.append(input, ` ${ratio}×`);
    options.append(label);
  }
  fieldset.append(legend, options);
  fieldset.addEventListener("change", (event) => onChange(Number(event.target.value)));
  return fieldset;
}

export function mount(root) {
  const stage = root.querySelector(".figure__stage");
  const gsap = window.gsap;
  const animate = Boolean(gsap) && !prefersReducedMotion();
  const wide = window.matchMedia(WIDE_QUERY);
  let ratio = DEFAULT_RATIO;
  let scene = null;

  const status = document.createElement("p");
  status.className = "figure__readout";
  status.setAttribute("aria-live", "polite");
  status.textContent = readout(ratio);

  const controls = document.createElement("div");
  controls.className = "figure__controls";
  controls.append(radioGroup("compression-ratio", (value) => {
    ratio = value;
    status.textContent = readout(ratio);
    showState(scene, ratio, animate ? gsap : null);
  }), status);

  const render = () => {
    scene?.tl?.kill();
    const name = wide.matches ? "wide" : "narrow";
    const next = buildScene({name, ...LAYOUTS[name]});
    if (scene) scene.svg.replaceWith(next.svg);
    else stage.append(next.svg);
    scene = next;
    showState(scene, ratio, null);
  };

  stage.append(controls);
  render();
  wide.addEventListener("change", render);
  root.classList.add("is-ready");
}

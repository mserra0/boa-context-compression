import { el } from "../lib/svg.js";
import { prefersReducedMotion } from "../lib/motion.js";
import { stepAt } from "../lib/steps.js";
import { rowWidth, tokenRow } from "./tokens.js";

// Diagram parameters, not measurements: the counts of the paper's Figure 1.
const CONTEXT_TOKENS = 8;
const COMPRESSED_TOKENS = 3;
const OUTPUT_TOKENS = 5;
const STEP_COUNT = 5;

const WIDE_QUERY = "(min-width: 56em)";

// A step becomes active when the top of its list item crosses this line: half-way down the
// viewport (wide), or half-way down the space left below the sticky diagram (narrow). Each
// caption is centred in its item, so the active caption stays within one item's height of
// the line, in view.
const WIDE_LINE = 0.5;
const NARROW_LINE = 0.5;

const DESCRIPTION = "Boa's method, after Figure 1 of the paper. The context tokens go into Mamba-2, "
  + "inside the trained Boa model, marked with a flame. A stride operation keeps three compressed tokens. "
  + "K and V projections turn them into a key-value cache, placed at the start of a frozen Transformer LLM, "
  + "marked with a snowflake, which generates the student's output tokens with distribution p S given x before t and z. "
  + "The same frozen LLM reading the full context is the teacher, with distribution p T given x before t and c. "
  + "A KL divergence links the two distributions.";

// Geometry in viewBox units. The pipeline runs from the bottom up, as in the paper.
// Every y is a top edge unless it says otherwise; "cy" is a centre.
const LAYOUTS = {
  wide: {
    width: 640, height: 600, rise: 8, head: 8, corner: 14,
    font: {kl: 26, math: 22, llm: 19, mamba: 20, op: 16},
    llmLines: ["Transformer LLM"],
    teacher: {x0: 12, x1: 262},
    student: {x0: 302, x1: 626, kv: 72},
    llm: {y: 150, height: 120, rx: 10},
    kl: {cy: 22},
    math: {y: 76},
    output: {y: 100, size: 28, gap: 14},
    snowflake: {r: 12, dx: -4, dy: 5},
    hatch: {spacing: 22, width: 2.6},
    group: {x0: 345, x1: 625, y0: 298, y1: 596, rx: 22, dash: "9 6", stroke: 2},
    pipeX: 485,
    join: {cy: 336, r: 14, pathY: 308},
    proj: {y0: 362, y1: 392, offset: 61, top: 35, bottom: 51},
    branchY: 412,
    compressed: {y: 430, size: 21, gap: 11},
    stride: {y: 462, height: 24, half: 88},
    mamba: {y: 512, height: 68, x0: 359, x1: 610},
    context: {size: 21, gap: 10, arrowFrom: 12, arrowTo: 14},
    flame: {x: 594, y: 330, size: 30},
  },
  narrow: {
    width: 344, height: 356, rise: 5, head: 6, corner: 10,
    font: {kl: 17, math: 16, llm: 13, mamba: 14, op: 12},
    llmLines: ["Transformer", "LLM"],
    teacher: {x0: 4, x1: 158},
    student: {x0: 176, x1: 338, kv: 34},
    llm: {y: 86, height: 54, rx: 7},
    kl: {cy: 14},
    math: {y: 50},
    output: {y: 60, size: 15, gap: 7},
    snowflake: {r: 8, dx: -3, dy: 4},
    hatch: {spacing: 12, width: 1.5},
    group: {x0: 186, x1: 338, y0: 150, y1: 352, rx: 14, dash: "6 4", stroke: 1.5},
    pipeX: 262,
    join: {cy: 180, r: 10, pathY: 160},
    proj: {y0: 200, y1: 222, offset: 36, top: 23, bottom: 32},
    branchY: 236,
    compressed: {y: 246, size: 13, gap: 7},
    stride: {y: 268, height: 20, half: 50},
    mamba: {y: 300, height: 42, x0: 196, x1: 328},
    context: {size: 12, gap: 5.5, arrowFrom: 6, arrowTo: 5},
    flame: {x: 322, y: 172, size: 19},
  },
};

const LEGEND = [
  {id: "kv", label: "KV cache"},
  {id: "compressed", label: "Compressed tokens"},
  {id: "context", label: "Context"},
  {id: "boa", label: "Boa model"},
  {id: "teacher", label: "Teacher output"},
  {id: "student", label: "Student output"},
  {id: "frozen", label: "Frozen"},
  {id: "trained", label: "Trained"},
];

// A flame in a 24 by 24 box: the outer tongue and its core.
const FLAME_OUTER = "M12 1.5C12.7 5 14.7 7.3 16.7 9.2C18.9 11.3 20 13.5 20 15.9C20 20.2 16.4 23 12 23C7.6 23 4 20.2 4 15.9C4 13.1 5.4 10.8 7.4 9.2C7.5 11.4 8.4 12.8 9.8 13.3C9.2 9.2 10.2 4.9 12 1.5Z";
const FLAME_CORE = "M12 12.6C13.2 14.9 15.6 16.3 15.6 18.9C15.6 20.9 14 22.3 12 22.3C10 22.3 8.4 20.9 8.4 18.9C8.4 16.7 10.6 15.3 12 12.6Z";

const round = (n) => Number(n.toFixed(2));

function label(content, x, cy, size, className = "diagram-label") {
  return el("text", {x, y: round(cy + size * 0.35), "font-size": size, "text-anchor": "middle", class: className}, [content]);
}

function lines(contents, x, cy, size) {
  const leading = size * 1.2;
  const top = cy - ((contents.length - 1) * leading) / 2;
  return contents.map((content, i) => label(content, x, top + i * leading, size));
}

// A filled arrowhead with its tip at (x, y), pointing along `angle` degrees (0 is right, -90 is up).
function head(x, y, angle, size) {
  const a = (angle * Math.PI) / 180;
  const bx = x - size * Math.cos(a);
  const by = y - size * Math.sin(a);
  const px = -Math.sin(a) * size * 0.5;
  const py = Math.cos(a) * size * 0.5;
  return el("path", {
    d: `M${round(x)} ${round(y)}L${round(bx + px)} ${round(by + py)}L${round(bx - px)} ${round(by - py)}Z`,
    class: "method__head",
  });
}

function line(d) {
  return el("path", {d, class: "method__line"});
}

// A short straight arrow along one axis.
function arrow(x1, y1, x2, y2, size) {
  const angle = (Math.atan2(y2 - y1, x2 - x1) * 180) / Math.PI;
  const a = (angle * Math.PI) / 180;
  const ex = x2 - Math.cos(a) * size * 0.6;
  const ey = y2 - Math.sin(a) * size * 0.6;
  return [line(`M${round(x1)} ${round(y1)}L${round(ex)} ${round(ey)}`), head(x2, y2, angle, size)];
}

function snowflake(cx, cy, r) {
  const d = [];
  for (let k = 0; k < 6; k++) {
    const a = ((90 + k * 60) * Math.PI) / 180;
    const ux = Math.cos(a);
    const uy = -Math.sin(a);
    d.push(`M${round(cx)} ${round(cy)}L${round(cx + ux * r)} ${round(cy + uy * r)}`);
    const bx = cx + ux * r * 0.55;
    const by = cy + uy * r * 0.55;
    for (const side of [-1, 1]) {
      const b = a + side * (Math.PI / 4);
      d.push(`M${round(bx)} ${round(by)}L${round(bx + Math.cos(b) * r * 0.38)} ${round(by - Math.sin(b) * r * 0.38)}`);
    }
  }
  const path = d.join("");
  return el("g", {}, [
    el("path", {d: path, class: "method__snow-halo", "stroke-width": round(r * 0.5)}),
    el("path", {d: path, class: "method__snow", "stroke-width": round(r * 0.17)}),
  ]);
}

function flame(cx, cy, size) {
  const s = size / 24;
  return el("g", {transform: `translate(${round(cx - 12 * s)} ${round(cy - 12 * s)}) scale(${round(s)})`}, [
    el("path", {d: FLAME_OUTER, class: "method__flame"}),
    el("path", {d: FLAME_CORE, class: "method__flame-core"}),
  ]);
}

// Text with italic symbols and subscripts, written as [content, style] pairs where the style
// holds "it" for italic and "sub" for a subscript.
function math(parts, x, y, size, className = "method__math") {
  let lowered = false;
  const spans = parts.map(([content, style = ""]) => {
    const sub = style.includes("sub");
    const attrs = {};
    if (sub !== lowered) {
      attrs.dy = round((sub ? 1 : -1) * size * 0.28);
      lowered = sub;
    }
    if (sub) attrs["font-size"] = round(size * 0.72);
    if (style.includes("it")) attrs["font-style"] = "italic";
    return el("tspan", attrs, [content]);
  });
  return el("text", {x: round(x), y, "font-size": size, "text-anchor": "middle", class: className}, spans);
}

function distribution(model, condition) {
  return [["p", "it"], [model, "sub"], ["(· | "], ["x", "it"], ["<", "sub"], ["t", "sub it"], [", "], [condition, "it"], [")"]];
}

function trapezoid(cx, {y0, y1, top, bottom}) {
  const points = [[cx - bottom, y1], [cx - top, y0], [cx + top, y0], [cx + bottom, y1]];
  return el("polygon", {points: points.map(([x, y]) => `${round(x)},${round(y)}`).join(" "), class: "method__op"});
}

function llmBox(x0, x1, L, {openLeft = false} = {}) {
  const {y, height, rx} = L.llm;
  const y1 = y + height;
  if (!openLeft) return el("rect", {x: x0, y, width: x1 - x0, height, rx, class: "block block--llm"});
  // Open on the left, where the KV cache closes it.
  return el("path", {
    d: `M${x0} ${y}H${x1 - rx}A${rx} ${rx} 0 0 1 ${x1} ${y + rx}V${y1 - rx}A${rx} ${rx} 0 0 1 ${x1 - rx} ${y1}H${x0}`,
    class: "block block--llm",
  });
}

function kvBlock(x0, x1, L) {
  const {y, height, rx} = L.llm;
  const y1 = y + height;
  return el("path", {
    d: `M${x0 + rx} ${y}H${x1}V${y1}H${x0 + rx}A${rx} ${rx} 0 0 1 ${x0} ${y1 - rx}V${y + rx}A${rx} ${rx} 0 0 1 ${x0 + rx} ${y}Z`,
    class: "method__kv",
  });
}

function centredRow(count, cx, {y, size, gap}, className) {
  return tokenRow(count, {x: round(cx - rowWidth(count, {size, gap}) / 2), y, size, gap, className});
}

// Builds the diagram; returns the SVG and, for each of the five steps, the parts it adds,
// in the order they appear.
function buildScene(L) {
  const svg = el("svg", {viewBox: `0 0 ${L.width} ${L.height}`, class: `method__svg method__svg--${L.name}`, role: "img", "aria-label": DESCRIPTION});
  const teacherX = (L.teacher.x0 + L.teacher.x1) / 2;
  const kvX1 = L.student.x0 + L.student.kv;
  const studentX = (kvX1 + L.student.x1) / 2;
  const kvX = L.student.x0 + L.student.kv / 2;
  const llmCy = L.llm.y + L.llm.height / 2;
  const llmBottom = L.llm.y + L.llm.height;
  const mambaCy = L.mamba.y + L.mamba.height / 2;
  const contextY = mambaCy - L.context.size / 2;
  const contextRight = teacherX + rowWidth(CONTEXT_TOKENS, L.context) / 2;
  const {pipeX, join, proj, corner: k} = L;
  const part = (...children) => el("g", {}, children.flat());

  svg.append(el("defs", {}, [
    el("pattern", {
      id: "method-kv-hatch", patternUnits: "userSpaceOnUse", width: L.hatch.spacing, height: L.hatch.spacing,
      patternTransform: "rotate(55)",
    }, [
      el("rect", {width: L.hatch.spacing, height: L.hatch.spacing, class: "method__hatch-ground"}),
      el("line", {x1: 0, y1: 0, x2: 0, y2: L.hatch.spacing, class: "method__hatch-line", "stroke-width": L.hatch.width}),
    ]),
  ]));

  // 1. Mamba-2 reads the context once.
  const step1 = [
    part(centredRow(CONTEXT_TOKENS, teacherX, {y: contextY, ...L.context}, "token token--context")),
    part(arrow(contextRight + L.context.arrowFrom, mambaCy, L.group.x0 - L.context.arrowTo, mambaCy, L.head)),
    part(
      el("rect", {
        x: L.group.x0, y: L.group.y0, width: L.group.x1 - L.group.x0, height: L.group.y1 - L.group.y0, rx: L.group.rx,
        class: "method__boa", "stroke-dasharray": L.group.dash, "stroke-width": L.group.stroke,
      }),
      flame(L.flame.x, L.flame.y, L.flame.size),
    ),
    part(
      el("rect", {x: L.mamba.x0, y: L.mamba.y, width: L.mamba.x1 - L.mamba.x0, height: L.mamba.height, rx: L.llm.rx, class: "block block--mamba"}),
      label("Mamba-2", pipeX, mambaCy, L.font.mamba),
    ),
  ];

  // 2. A stride keeps every r-th hidden state.
  const strideBottom = L.stride.y + L.stride.height;
  const step2 = [
    part(arrow(pipeX, L.mamba.y - 2, pipeX, strideBottom + 2, L.head)),
    part(
      el("rect", {x: pipeX - L.stride.half, y: L.stride.y, width: 2 * L.stride.half, height: L.stride.height, rx: L.stride.height / 2, class: "method__op"}),
      label("Stride op.", pipeX, L.stride.y + L.stride.height / 2, L.font.op),
    ),
    part(centredRow(COMPRESSED_TOKENS, pipeX, L.compressed, "token token--soft")),
  ];

  // 3. K and V projections turn them into a key-value cache.
  const kX = pipeX - proj.offset;
  const vX = pipeX + proj.offset;
  const tickTip = proj.y1 + 1;
  const projCy = (proj.y0 + proj.y1) / 2;
  const joinLeft = pipeX - join.r - 1;
  const joinRight = pipeX + join.r + 1;
  const cross = join.r * 0.7;
  const kvTip = llmBottom + 2;
  const step3 = [
    part(
      line(`M${pipeX} ${L.compressed.y - 3}V${L.branchY}`),
      line(`M${kX} ${tickTip + L.head * 0.6}V${L.branchY - 4}Q${kX} ${L.branchY} ${kX + 4} ${L.branchY}H${vX - 4}Q${vX} ${L.branchY} ${vX} ${L.branchY - 4}V${tickTip + L.head * 0.6}`),
      head(kX, tickTip, -90, L.head),
      head(vX, tickTip, -90, L.head),
    ),
    part(trapezoid(kX, proj), label("K proj", kX, projCy, L.font.op)),
    part(trapezoid(vX, proj), label("V proj", vX, projCy, L.font.op)),
    part(
      line(`M${kX} ${proj.y0}C${kX} ${join.cy} ${round(joinLeft - (joinLeft - kX) * 0.45)} ${join.cy} ${round(joinLeft - L.head * 0.6)} ${join.cy}`),
      head(joinLeft, join.cy, 0, L.head),
      line(`M${vX} ${proj.y0}C${vX} ${join.cy} ${round(joinRight + (vX - joinRight) * 0.45)} ${join.cy} ${round(joinRight + L.head * 0.6)} ${join.cy}`),
      head(joinRight, join.cy, 180, L.head),
    ),
    part(
      el("circle", {cx: pipeX, cy: join.cy, r: join.r, class: "method__join"}),
      line(`M${round(pipeX - cross)} ${round(join.cy - cross)}L${round(pipeX + cross)} ${round(join.cy + cross)}M${round(pipeX + cross)} ${round(join.cy - cross)}L${round(pipeX - cross)} ${round(join.cy + cross)}`),
    ),
    part(
      line(`M${pipeX} ${join.cy - join.r}V${join.pathY + k}Q${pipeX} ${join.pathY} ${pipeX - k} ${join.pathY}H${kvX + k}Q${kvX} ${join.pathY} ${kvX} ${join.pathY - k}V${round(kvTip + L.head * 0.6)}`),
      head(kvX, kvTip, -90, L.head),
    ),
    part(kvBlock(L.student.x0, kvX1, L)),
  ];

  // 4. The frozen LLM attends to that cache.
  const step4 = [
    part(
      llmBox(kvX1, L.student.x1, L, {openLeft: true}),
      lines(L.llmLines, studentX, llmCy, L.font.llm),
      snowflake(L.student.x1 + L.snowflake.dx, L.llm.y + L.snowflake.dy, L.snowflake.r),
    ),
    part(centredRow(OUTPUT_TOKENS, studentX, L.output, "token token--student")),
  ];

  // 5. Training matches the student to the full-context teacher.
  const klX = (teacherX + studentX) / 2;
  const klHalf = L.font.kl * 0.72;
  const arcTop = L.kl.cy;
  const arcFoot = L.math.y - L.font.math * 0.95;
  const r = arcFoot - arcTop;
  const tipLeft = klX - klHalf - 6;
  const tipRight = klX + klHalf + 6;
  const step5 = [
    part(arrow(teacherX, contextY - L.context.arrowFrom, teacherX, llmBottom + 2, L.head)),
    part(
      llmBox(L.teacher.x0, L.teacher.x1, L),
      lines(L.llmLines, teacherX, llmCy, L.font.llm),
      snowflake(L.teacher.x1 + L.snowflake.dx, L.llm.y + L.snowflake.dy, L.snowflake.r),
    ),
    part(centredRow(OUTPUT_TOKENS, teacherX, L.output, "token token--teacher")),
    part(
      math(distribution("T", "c"), teacherX, L.math.y, L.font.math),
      math(distribution("S", "z"), studentX, L.math.y, L.font.math),
    ),
    part(
      line(`M${teacherX} ${round(arcFoot)}Q${teacherX} ${arcTop} ${round(teacherX + r)} ${arcTop}H${round(tipLeft - L.head * 0.6)}`),
      head(tipLeft, arcTop, 0, L.head),
      line(`M${studentX} ${round(arcFoot)}Q${studentX} ${arcTop} ${round(studentX - r)} ${arcTop}H${round(tipRight + L.head * 0.6)}`),
      head(tipRight, arcTop, 180, L.head),
      label("KL", klX, arcTop, L.font.kl, "method__kl"),
    ),
  ];

  const steps = [step1, step2, step3, step4, step5];
  steps.forEach((parts, i) => svg.append(el("g", {"data-step": i + 1}, parts)));
  return {svg, steps, rise: L.rise};
}

function swatch(id) {
  const svg = el("svg", {viewBox: "0 0 16 16", width: 16, height: 16, class: "method__swatch", "aria-hidden": "true", focusable: "false"});
  const square = (className) => el("rect", {x: 1.5, y: 1.5, width: 13, height: 13, rx: 2.5, class: className});
  const content = {
    kv: () => [square("method__kv method__kv--swatch"), el("path", {d: "M3 10L10 3M4.5 13.5L13.5 4.5", class: "method__hatch-line", "stroke-width": 1.25})],
    compressed: () => [square("token token--soft")],
    context: () => [square("token token--context")],
    boa: () => [el("rect", {x: 1.5, y: 1.5, width: 13, height: 13, rx: 2.5, class: "method__boa", "stroke-dasharray": "3 2", "stroke-width": 1.5})],
    teacher: () => [square("token token--teacher")],
    student: () => [square("token token--student")],
    frozen: () => [snowflake(8, 8, 7)],
    trained: () => [flame(8, 8, 16)],
  };
  svg.append(...content[id]());
  return svg;
}

function buildLegend() {
  const list = document.createElement("ul");
  list.className = "method__legend";
  list.setAttribute("aria-label", "Legend");
  for (const {id, label: text} of LEGEND) {
    const item = document.createElement("li");
    item.append(swatch(id), text);
    list.append(item);
  }
  return list;
}

// Shows the parts of steps 1 to `step` and hides the rest, at once.
function setStep(gsap, scene, step) {
  scene.steps.forEach((parts, i) => {
    const visible = i < step;
    gsap.set(parts, {autoAlpha: visible ? 1 : 0, y: visible ? 0 : scene.rise});
  });
}

// Moving forward, each new step's parts rise into place one after another; moving back,
// the later steps fade out together.
function moveStep(gsap, scene, from, to) {
  if (to > from) {
    for (let step = from + 1; step <= to; step++) {
      gsap.to(scene.steps[step - 1], {
        autoAlpha: 1, y: 0, duration: 0.45, ease: "power2.out", stagger: 0.09,
        delay: (step - from - 1) * 0.3, overwrite: true,
      });
    }
  } else {
    for (let step = to + 1; step <= from; step++) {
      gsap.to(scene.steps[step - 1], {autoAlpha: 0, y: scene.rise, duration: 0.25, ease: "power1.in", overwrite: true});
    }
  }
}

// Moves each caption into a native button, so a step can be chosen without scrolling.
function stepButtons(items) {
  return items.map((item) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "method__step-button";
    // The caption sits in one span so its inline parts (<var>, nowrap) do not become flex items.
    const caption = document.createElement("span");
    caption.append(...item.childNodes);
    button.append(caption);
    item.append(button);
    return button;
  });
}

export function mount(root) {
  const stage = root.querySelector(".figure__stage");
  const items = [...root.querySelectorAll(".method__steps > li")];
  const {gsap, ScrollTrigger} = window;
  const assemble = Boolean(gsap && ScrollTrigger) && !prefersReducedMotion() && items.length === STEP_COUNT;
  const wide = window.matchMedia(WIDE_QUERY);
  let scene = null;
  let active = STEP_COUNT;

  const render = () => {
    const name = wide.matches ? "wide" : "narrow";
    const next = buildScene({name, ...LAYOUTS[name]});
    if (scene) {
      if (assemble) gsap.killTweensOf(scene.steps.flat());
      scene.svg.replaceWith(next.svg);
    } else {
      stage.append(next.svg);
    }
    scene = next;
    if (assemble) setStep(gsap, scene, active);
  };

  stage.after(buildLegend());

  if (!assemble) {
    render();
    wide.addEventListener("change", render);
    root.classList.add("is-ready");
    return;
  }

  root.classList.add("method--assembling");
  active = 1;
  render();

  const buttons = stepButtons(items);
  const markActive = () => {
    items.forEach((item, i) => item.classList.toggle("is-active", i + 1 === active));
    buttons.forEach((button, i) => {
      if (i + 1 === active) button.setAttribute("aria-current", "step");
      else button.removeAttribute("aria-current");
    });
  };
  // The line a caption's top crosses to become the active step, in pixels from the viewport top.
  const activationLine = () => {
    if (wide.matches) return Math.round(window.innerHeight * WIDE_LINE);
    const covered = stage.getBoundingClientRect().height;
    return Math.round(covered + (window.innerHeight - covered) * NARROW_LINE);
  };
  // The scroll position at which each step starts, measured now: content above the figure
  // (web fonts, the results tables, the cost figure) can change height after load.
  const stepStarts = () => {
    const line = activationLine();
    return items.map((item) => window.scrollY + item.getBoundingClientRect().top - line);
  };
  const update = () => {
    const next = stepAt(stepStarts(), window.scrollY);
    if (next === active) return;
    moveStep(gsap, scene, active, next);
    active = next;
    markActive();
  };

  // One trigger over the whole page reports every scroll; it neither pins nor adds length.
  ScrollTrigger.create({start: 0, end: "max", onUpdate: update, onRefresh: update});

  // Choosing a step scrolls its caption to the activation line, the middle of its active range.
  buttons.forEach((button, i) => {
    button.addEventListener("click", () => {
      const top = Math.ceil(stepStarts()[i] + items[i].offsetHeight / 2);
      window.scrollTo({top, behavior: "smooth"});
    });
  });

  wide.addEventListener("change", () => {
    render();
    update();
  });
  markActive();
  update();
  root.classList.add("is-ready");
}

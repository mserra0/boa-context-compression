import { el } from "../lib/svg.js";
import { linearScale, logScale } from "../lib/scales.js";
import { labelStep, logTicks, niceLinear, niceLogDomain } from "../lib/axis.js";
import { spreadLabels } from "../lib/spread.js";
import { prefersReducedMotion } from "../lib/motion.js";

// A chart drawn this narrow (in viewBox units) is meant for phones and uses the smaller text.
const NARROW_WIDTH = 600;
const FONT = {narrow: 12, wide: 14};
const CHAR_EM = 0.6;          // average glyph width in ems, for sizing labels before they render
const DRAW_S = 1.2;           // time for the lines to cross the plot
const POINT_FADE_S = 0.2;
const HIT_RADIUS = 11;        // pointer target around each point
const DOT_RADIUS = {normal: 3.25, emphasis: 4.5};
const RING_RADIUS = 7;        // marker ring around the points at the marked x
const FLIP_AT = 0.7;          // past this share of the plot, marker labels sit left of the marker

let chartCount = 0;

function xPositions(xScale, xValues, x0, x1) {
  const n = xValues.length;
  if (n === 1) return [(x0 + x1) / 2];
  if (xScale === "log") return xValues.map(logScale([xValues[0], xValues[n - 1]], [x0, x1]));
  return xValues.map((v, i) => x0 + (i * (x1 - x0)) / (n - 1));
}

// Splits a series into runs of consecutive usable values (a null leaves a gap),
// with each point's distance along its run.
function splitRuns(values, usable, xs, yAt) {
  const runs = [];
  let run = null;
  values.forEach((value, index) => {
    if (!usable(value)) { run = null; return; }
    if (!run) runs.push(run = []);
    run.push({index, value, x: xs[index], y: yAt(value)});
  });
  for (const points of runs) {
    let along = 0;
    points.forEach((p, k) => {
      if (k > 0) along += Math.hypot(p.x - points[k - 1].x, p.y - points[k - 1].y);
      p.along = along;
    });
  }
  return runs;
}

function classes(...names) {
  return names.filter(Boolean).join(" ");
}

/**
 * Draws an SVG line chart with a legend into `root` and returns controls for it.
 * Lines and points stay hidden until draw() is called. The y axis fits the visible series.
 *
 * options: {width, height, xScale: "log" | "point", yScale: "log" | "linear", xValues,
 *   xFormat, yFormat, yLabel, xLabel, fontSize?,
 *   series: [{id, label, values: (number | null)[], className, emphasis?}]}
 * returns: {setMarker(index), setVisible(ids), draw(animate)}
 */
export function lineChart(root, options) {
  const {width, height, xScale, yScale, xValues, xFormat = String, yFormat = String, yLabel, xLabel, series} = options;
  const font = options.fontSize ?? (width < NARROW_WIDTH ? FONT.narrow : FONT.wide);
  const uid = `line-chart-${++chartCount}`;
  const usable = (v) => Number.isFinite(v) && (yScale !== "log" || v > 0);
  const textWidth = (s) => String(s).length * font * CHAR_EM;

  const svg = el("svg", {
    viewBox: `0 0 ${width} ${height}`,
    class: "line-chart__svg",
    role: "group",
    tabindex: -1,   // the points are the tab stop, not the drawing
    "aria-label": [yLabel, xLabel].filter(Boolean).join(" against "),
  });

  const legend = document.createElement("ul");
  legend.className = "line-chart__legend";
  const legendItems = new Map();
  for (const s of series) {
    const item = document.createElement("li");
    item.className = classes("line-chart__legend-item", s.className, s.emphasis && "is-emphasis");
    item.append(el("svg", {viewBox: "0 0 28 10", width: 28, height: 10, class: "line-chart__swatch", "aria-hidden": "true", focusable: "false"}, [
      el("line", {x1: 1, y1: 5, x2: 27, y2: 5, class: "line-chart__line"}),
      el("circle", {cx: 14, cy: 5, r: s.emphasis ? 3.5 : 2.75, class: "line-chart__dot"}),
    ]), s.label);
    legend.append(item);
    legendItems.set(s.id, item);
  }

  const wrapper = document.createElement("div");
  wrapper.className = "line-chart";
  wrapper.append(svg, legend);
  root.append(wrapper);

  let visible = new Set(series.map((s) => s.id));
  let markerIndex = null;
  let drawn = false;           // whether draw() has run; later rebuilds show the lines at once
  let timeline = null;
  let roving = null;           // the one point in the tab order
  let rovingKey = null;        // {id, index} of that point, kept across rebuilds
  let view = null;             // everything that depends on the visible series

  const pointOf = (target) => (target instanceof Element ? view?.pointByNode.get(target.closest(".line-chart__point")) : undefined);

  // Builds axes, series, marker and tooltip for the visible series.
  const build = () => {
    timeline?.kill();
    timeline = null;
    const shown = series.filter((s) => visible.has(s.id));
    const values = (shown.length ? shown : series).flatMap((s) => s.values).filter(usable);
    if (values.length === 0) throw new Error("lineChart: no values to plot");
    const [min, max] = [Math.min(...values), Math.max(...values)];
    let yDomain;
    let yTicks;
    if (yScale === "log") {
      yDomain = niceLogDomain(min, max);
      yTicks = logTicks(...yDomain);
    } else {
      ({domain: yDomain, ticks: yTicks} = niceLinear(min, max));
    }

    const plot = {
      left: Math.max(...yTicks.map((t) => textWidth(yFormat(t)))) + font * 0.75,
      right: width - font,
      top: yLabel ? font * 2.4 : font,
      bottom: height - font * (xLabel ? 3.8 : 2.2),
    };
    const inset = xScale === "point" ? (plot.right - plot.left) / (2 * xValues.length) : font * 0.75;
    const x0 = plot.left + inset;
    const x1 = plot.right - inset;
    const xs = xPositions(xScale, xValues, x0, x1);
    const yAt = (yScale === "log" ? logScale : linearScale)(yDomain, [plot.bottom, plot.top]);
    const progressAt = (x) => (x1 > x0 ? (x - x0) / (x1 - x0) : 0);

    // Axes: light horizontal rules at the y ticks, x ticks along the bottom.
    const axes = el("g", {class: "line-chart__axes", "aria-hidden": "true"});
    for (const tick of yTicks) {
      const y = yAt(tick);
      axes.append(
        el("line", {x1: plot.left, x2: plot.right, y1: y, y2: y, class: "line-chart__rule"}),
        el("text", {x: plot.left - font * 0.5, y, dy: "0.35em", "text-anchor": "end", class: "line-chart__tick", "font-size": font}, [yFormat(tick)]),
      );
    }
    axes.append(el("line", {x1: plot.left, x2: plot.right, y1: plot.bottom, y2: plot.bottom, class: "line-chart__axis"}));
    const spacing = Math.min(...xs.slice(1).map((x, i) => x - xs[i]));
    const step = labelStep(spacing, Math.max(...xValues.map((v) => textWidth(xFormat(v)))) + font * 0.6);
    xs.forEach((x, i) => {
      axes.append(el("line", {x1: x, x2: x, y1: plot.bottom, y2: plot.bottom + font * 0.35, class: "line-chart__axis"}));
      if (i % step === 0) {
        axes.append(el("text", {x, y: plot.bottom + font * 1.5, "text-anchor": "middle", class: "line-chart__tick", "font-size": font}, [xFormat(xValues[i])]));
      }
    });
    if (yLabel) axes.append(el("text", {x: 0, y: font * 1.1, class: "line-chart__title", "font-size": font}, [yLabel]));
    if (xLabel) {
      axes.append(el("text", {x: (plot.left + plot.right) / 2, y: height - font * 0.4, "text-anchor": "middle", class: "line-chart__title", "font-size": font}, [xLabel]));
    }

    // Series, the emphasised ones drawn last so they sit on top.
    const defs = el("defs");
    const layer = el("g");
    const runs = [];
    const points = [];
    const pointByNode = new Map();
    const drawOrder = [...shown].sort((a, b) => Number(Boolean(a.emphasis)) - Number(Boolean(b.emphasis)));
    drawOrder.forEach((s, n) => {
      const lines = el("g");
      const dots = el("g");
      splitRuns(s.values, usable, xs, yAt).forEach((run, r) => {
        const d = run.map((p, k) => `${k ? "L" : "M"}${p.x.toFixed(2)} ${p.y.toFixed(2)}`).join("");
        const total = run.at(-1).along;
        const start = progressAt(run[0].x);
        const end = progressAt(run.at(-1).x);
        if (total > 0) {
          // The mask reveals the line along its length; the line keeps its own dash pattern.
          const dash = total + 2;
          const id = `${uid}-${n}-${r}`;
          const maskPath = el("path", {d, class: "line-chart__reveal", "stroke-dasharray": `${dash} ${dash}`, "stroke-dashoffset": drawn ? 0 : dash});
          defs.append(el("mask", {id, maskUnits: "userSpaceOnUse", x: 0, y: 0, width, height}, [maskPath]));
          lines.append(el("path", {d, class: "line-chart__line", mask: `url(#${id})`}));
          runs.push({maskPath, dash, start: start * DRAW_S, duration: (end - start) * DRAW_S});
        }
        for (const p of run) {
          const text = `${s.label}, ${xFormat(xValues[p.index])}: ${yFormat(p.value)}`;
          const node = el("g", {class: "line-chart__point", tabindex: -1, role: "img", "aria-label": text, opacity: drawn ? 1 : 0}, [
            el("circle", {cx: p.x, cy: p.y, r: HIT_RADIUS, class: "line-chart__hit"}),
            el("circle", {cx: p.x, cy: p.y, r: s.emphasis ? DOT_RADIUS.emphasis : DOT_RADIUS.normal, class: "line-chart__dot"}),
          ]);
          const at = (start + (total > 0 ? (end - start) * (p.along / total) : 0)) * DRAW_S;
          const point = {...p, node, text, at, series: s};
          points.push(point);
          pointByNode.set(node, point);
          dots.append(node);
        }
      });
      layer.append(el("g", {class: classes("line-chart__series", s.className, s.emphasis && "is-emphasis")}, [lines, dots]));
    });

    const marker = el("g", {class: "line-chart__marker", "aria-hidden": "true"});
    const tipBox = el("rect", {rx: 4, class: "line-chart__tip-box"});
    const tipText = el("text", {class: "line-chart__tip-text", "font-size": font});
    const tip = el("g", {class: "line-chart__tip", "aria-hidden": "true", visibility: "hidden"}, [tipBox, tipText]);
    const markerTop = el("g", {class: "line-chart__marker", "aria-hidden": "true"});
    svg.replaceChildren(defs, axes, marker, layer, markerTop, tip);

    // Points in tab order: series in the order given, then by x.
    const ordered = series.flatMap((s) => points.filter((p) => p.series === s));
    view = {plot, xs, yAt, progressAt, runs, points: ordered, pointByNode, marker, markerTop, tip, tipBox, tipText};
    const keep = rovingKey && ordered.find((p) => p.series.id === rovingKey.id && p.index === rovingKey.index);
    setRoving(keep ?? ordered.find((p) => p.series.emphasis) ?? ordered[0]);
    renderMarker();
  };

  function setRoving(point) {
    roving = point ?? null;
    for (const p of view.points) p.node.setAttribute("tabindex", p === roving ? 0 : -1);
    if (roving) rovingKey = {id: roving.series.id, index: roving.index};
  }

  function renderMarker() {
    const {marker, markerTop, plot, xs, yAt, progressAt} = view;
    marker.replaceChildren();
    markerTop.replaceChildren();
    if (markerIndex === null || !(markerIndex in xs)) return;
    const x = xs[markerIndex];
    const xText = xFormat(xValues[markerIndex]);
    const boxW = textWidth(xText) + font * 0.9;
    marker.append(
      el("line", {x1: x, x2: x, y1: plot.top, y2: plot.bottom, class: "line-chart__marker-line"}),
      el("rect", {x: x - boxW / 2, y: plot.bottom + font * 0.45, width: boxW, height: font * 1.45, class: "line-chart__marker-backing"}),
      el("text", {x, y: plot.bottom + font * 1.5, "text-anchor": "middle", class: "line-chart__marker-x", "font-size": font}, [xText]),
    );
    // A ring around each marked point, its value beside it; labels move apart where they would touch.
    const items = series
      .filter((s) => visible.has(s.id) && usable(s.values[markerIndex]))
      .map((s) => ({s, value: s.values[markerIndex], y: yAt(s.values[markerIndex])}));
    const ys = spreadLabels(items.map((it) => it.y), font * 1.3, [plot.top + font * 0.5, plot.bottom - font * 0.5]);
    const side = progressAt(x) > FLIP_AT ? -1 : 1;
    items.forEach(({s, value, y}, k) => {
      markerTop.append(el("g", {class: classes("line-chart__marker-value", s.className)}, [
        el("circle", {cx: x, cy: y, r: RING_RADIUS, class: "line-chart__ring"}),
        el("text", {
          x: x + side * (RING_RADIUS + font * 0.4), y: ys[k], dy: "0.35em", "text-anchor": side < 0 ? "end" : "start",
          class: "line-chart__marker-text", "font-size": font,
        }, [yFormat(value)]),
      ]));
    });
  }

  // Tooltip: on hover, focus and tap.
  const showTip = (point) => {
    const {tip, tipBox, tipText} = view;
    tipText.textContent = point.text;
    let textW = 0;
    try { textW = tipText.getComputedTextLength(); } catch { textW = 0; }
    if (!textW) textW = textWidth(point.text);
    const pad = font * 0.5;
    const boxW = textW + 2 * pad;
    const boxH = font * 1.7;
    const bx = Math.min(Math.max(point.x - boxW / 2, 0), width - boxW);
    let by = point.y - boxH - font * 0.8;
    if (by < 0) by = point.y + font * 0.8;
    for (const [k, v] of Object.entries({x: bx, y: by, width: boxW, height: boxH})) tipBox.setAttribute(k, v);
    tipText.setAttribute("x", bx + pad);
    tipText.setAttribute("y", by + boxH / 2);
    tipText.setAttribute("dy", "0.35em");
    tip.setAttribute("visibility", "visible");
  };
  const hideTip = () => view.tip.setAttribute("visibility", "hidden");
  const showFocusedOrHide = () => {
    const focused = pointOf(document.activeElement);
    if (focused) showTip(focused);
    else hideTip();
  };

  svg.addEventListener("pointerover", (event) => {
    const point = pointOf(event.target);
    if (point) showTip(point);
  });
  svg.addEventListener("pointerout", (event) => {
    if (pointOf(event.target) && !pointOf(event.relatedTarget)) showFocusedOrHide();
  });
  svg.addEventListener("pointerup", (event) => {
    const point = pointOf(event.target);
    if (point && event.pointerType !== "mouse") point.node.focus({preventScroll: true});
  });
  svg.addEventListener("focusin", (event) => {
    const point = pointOf(event.target);
    if (!point) return;
    setRoving(point);
    showTip(point);
  });
  svg.addEventListener("focusout", hideTip);
  // One tab stop for the whole chart; arrow keys move between points.
  svg.addEventListener("keydown", (event) => {
    const point = pointOf(event.target);
    if (!point) return;
    const own = view.points.filter((p) => p.series === point.series);
    const k = own.indexOf(point);
    let next;
    if (event.key === "ArrowRight") next = own[k + 1];
    else if (event.key === "ArrowLeft") next = own[k - 1];
    else if (event.key === "Home") next = own[0];
    else if (event.key === "End") next = own.at(-1);
    else if (event.key === "ArrowUp" || event.key === "ArrowDown") {
      // Neighbouring series at the same x, ordered from the top of the plot to the bottom.
      const column = view.points.filter((p) => p.index === point.index).sort((a, b) => a.y - b.y);
      next = column[column.indexOf(point) + (event.key === "ArrowUp" ? -1 : 1)];
    } else return;
    event.preventDefault();
    if (next) next.node.focus();
  });

  const draw = (animate) => {
    const gsap = window.gsap;
    timeline?.kill();
    timeline = null;
    drawn = true;
    const {runs, points} = view;
    if (!animate || !gsap || prefersReducedMotion()) {
      for (const run of runs) run.maskPath.setAttribute("stroke-dashoffset", 0);
      for (const point of points) point.node.setAttribute("opacity", 1);
      return;
    }
    timeline = gsap.timeline();
    for (const run of runs) {
      timeline.fromTo(run.maskPath, {attr: {"stroke-dashoffset": run.dash}}, {attr: {"stroke-dashoffset": 0}, duration: run.duration, ease: "none"}, run.start);
    }
    for (const point of points) {
      timeline.fromTo(point.node, {attr: {opacity: 0}}, {attr: {opacity: 1}, duration: POINT_FADE_S, ease: "none"}, point.at);
    }
  };

  const setVisible = (ids) => {
    visible = new Set(ids);
    for (const [id, item] of legendItems) item.hidden = !visible.has(id);
    const hadFocus = Boolean(pointOf(document.activeElement));
    build();
    if (hadFocus) roving?.node.focus();
  };

  const setMarker = (index) => {
    markerIndex = index;
    renderMarker();
  };

  build();
  return {setMarker, setVisible, draw};
}

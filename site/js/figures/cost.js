import { onVisible, prefersReducedMotion } from "../lib/motion.js";
import { linearScale, logScale } from "../lib/scales.js";
import { labelsWithMetric, measurement, speedup } from "../lib/latency.js";
import { formatGigabytes, formatSeconds, formatTokens } from "../lib/format.js";
import { conditionsText } from "../lib/conditions.js";
import { speedSentence } from "../lib/speed.js";
import { lineChart } from "./line-chart.js";

const INITIAL_INDEX = 8;                  // 128k tokens
const MAIN_METHODS = ["full", "boa"];     // always shown; the other compressors are optional
const BAR_TWEEN_S = 0.4;
// The time bars use a log axis that starts this many times below the shortest time,
// so the shortest bar still has some length.
const LOG_BAR_HEADROOM = 3;

const WIDE_QUERY = "(min-width: 56em)";
const CHART_LAYOUTS = {
  wide: {width: 960, height: 380},
  narrow: {width: 344, height: 340},
};

const BLOCKS = [
  {
    metric: "endToEndS", title: "Time to answer", subject: "Time to answer", format: formatSeconds, scale: "log",
  },
  {
    metric: "memoryGB", title: "Peak GPU memory", subject: "Peak GPU memory", format: formatGigabytes, scale: "linear",
    lede: "The most GPU memory in use at any point while answering.",
  },
];

function node(tag, attrs = {}, children = []) {
  const element = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "text") element.textContent = value;
    else element.setAttribute(key, value);
  }
  element.append(...children);
  return element;
}

// "A, B and C" as nodes; each item stays on one line, so "Qwen3-1.7B" never breaks at its hyphen.
function listJoin(items) {
  return items.flatMap((item, i) => {
    const separator = i === 0 ? [] : [i === items.length - 1 ? " and " : ", "];
    return [...separator, node("span", {class: "nowrap", text: item})];
  });
}

// Bar length in percent, fixed per decoder and metric across every method and length,
// so bars grow as the context grows and do not jump when other compressors are shown.
function barScale(latency, decoder, metric, kind) {
  const values = [];
  for (const {id} of latency.methods) {
    for (const length of latency.lengths) {
      const value = measurement(latency, {decoder, metric, method: id, length});
      if (value !== null && value > 0) values.push(value);
    }
  }
  if (values.length === 0) return null;
  const max = Math.max(...values);
  if (kind === "log") return logScale([Math.min(...values) / LOG_BAR_HEADROOM, max], [0, 100]);
  return linearScale([0, max], [0, 100]);
}

// "Peak GPU memory is reported for Qwen3-1.7B.", from whichever decoders have the metric.
function unmeasuredNote(latency, metric, subject) {
  const reported = labelsWithMetric(latency, metric, MAIN_METHODS);
  return reported.length ? [`${subject} is reported for `, ...listJoin(reported), "."] : [`${subject} is not reported.`];
}

function slider(lengths) {
  const id = "cost-length";
  const input = node("input", {
    id, type: "range", min: 0, max: lengths.length - 1, step: 1, value: INITIAL_INDEX, class: "cost-slider__input",
  });
  const value = node("span", {class: "cost-slider__value"});
  const ends = node("div", {class: "cost-slider__ends", "aria-hidden": "true"}, [
    node("span", {text: `${formatTokens(lengths[0])} tokens`}),
    node("span", {text: `${formatTokens(lengths.at(-1))} tokens`}),
  ]);
  const head = node("div", {class: "cost-slider__head"}, [node("label", {for: id, text: "Context length"}), value]);
  return {root: node("div", {class: "cost-slider"}, [head, input, ends]), input, value};
}

function decoderGroup(latency, initial) {
  const options = node("div", {class: "choice-group__options"});
  for (const [id, decoder] of Object.entries(latency.decoders)) {
    const input = node("input", {type: "radio", name: "cost-decoder", value: id});
    input.checked = id === initial;
    options.append(node("label", {class: "choice"}, [input, ` ${decoder.label}`]));
  }
  return node("fieldset", {class: "choice-group"}, [node("legend", {text: "Frozen LLM"}), options]);
}

function barRow(id, label, isOther) {
  const value = node("span", {class: "cost-bar__value"});
  const fill = node("span", {class: "cost-bar__fill"});
  const classes = ["cost-bar", `series--${id}`];
  if (id === "boa") classes.push("is-emphasis");
  const root = node("li", {class: classes.join(" ")}, [
    node("span", {class: "cost-bar__label", text: label}),
    value,
    node("span", {class: "cost-bar__track", "aria-hidden": "true"}, [fill]),
  ]);
  return {id, root, value, fill, isOther};
}

function buildBlock(block, methods) {
  const titleId = `cost-${block.metric}-title`;
  const rows = methods.map(({id, label}) => barRow(id, label, !MAIN_METHODS.includes(id)));
  const list = node("ul", {class: "cost-bars"}, rows.map((row) => row.root));
  const lede = node("p", {class: "cost-block__lede"});
  if (block.lede) lede.textContent = block.lede;
  else lede.setAttribute("aria-live", "polite");
  const note = node("p", {class: "cost-block__note"});
  note.hidden = true;
  const root = node("section", {class: "cost-block", "aria-labelledby": titleId}, [
    node("h3", {id: titleId, text: block.title}), lede, list, note,
  ]);
  return {...block, root, rows, list, lede, note};
}

export function mount(root, data) {
  const {latency} = data;
  const stage = root.querySelector(".figure__stage");
  const gsap = window.gsap;
  const animate = Boolean(gsap) && !prefersReducedMotion();
  const wide = window.matchMedia(WIDE_QUERY);
  const {lengths} = latency;
  const decoderIds = Object.keys(latency.decoders);
  if (lengths.length === 0 || decoderIds.length === 0) throw new Error("latency.json has no lengths or decoders");

  // Full context and Boa first, then the other compressors in data order.
  const methods = [
    ...MAIN_METHODS.map((id) => latency.methods.find((m) => m.id === id)),
    ...latency.methods.filter((m) => !MAIN_METHODS.includes(m.id)),
  ];
  const state = {index: Math.min(INITIAL_INDEX, lengths.length - 1), decoder: decoderIds[0], showOthers: false};

  const length = slider(lengths);
  length.input.value = state.index;
  const decoders = decoderGroup(latency, state.decoder);
  const others = node("input", {type: "checkbox"});
  const controls = node("div", {class: "figure__controls cost__controls"}, [
    length.root,
    decoders,
    node("label", {class: "choice"}, [others, " Show other compressors"]),
  ]);
  stage.append(controls);
  const conditions = node("p", {class: "cost__note"});
  stage.append(conditions);

  const blocks = BLOCKS.map((block) => buildBlock(block, methods));
  stage.append(node("div", {class: "cost__blocks"}, blocks.map((b) => b.root)));

  const chartHost = node("div", {class: "cost__chart-host"});
  stage.append(node("section", {class: "cost__chart", "aria-labelledby": "cost-chart-title"}, [
    node("h3", {id: "cost-chart-title", text: "Time to answer at each measured length"}),
    chartHost,
  ]));

  const scales = new Map();
  const scaleFor = (block) => {
    const key = `${state.decoder}/${block.metric}`;
    if (!scales.has(key)) scales.set(key, barScale(latency, state.decoder, block.metric, block.scale));
    return scales.get(key);
  };

  const setWidth = (fill, percent, tween) => {
    const width = `${percent}%`;
    if (tween && animate) {
      gsap.to(fill, {width, duration: BAR_TWEEN_S, ease: "power2.out", overwrite: true});
    } else {
      gsap?.killTweensOf(fill);
      fill.style.width = width;
    }
  };

  let chart = null;
  let drawn = false;
  const visibleIds = () => methods.map((m) => m.id).filter((id) => state.showOthers || MAIN_METHODS.includes(id));

  const buildChart = () => {
    const name = wide.matches ? "wide" : "narrow";
    chartHost.className = `cost__chart-host cost__chart-host--${name}`;
    chartHost.replaceChildren();
    // Boa first in the legend; the chart draws the emphasised series on top.
    const order = ["boa", ...methods.map((m) => m.id).filter((id) => id !== "boa")];
    chart = lineChart(chartHost, {
      ...CHART_LAYOUTS[name],
      xScale: "log",
      yScale: "log",
      xValues: lengths,
      xFormat: formatTokens,
      yFormat: formatSeconds,
      yLabel: "Time to answer (log scale)",
      xLabel: "Context length in tokens (log scale)",
      series: order.map((id) => ({
        id,
        label: latency.methods.find((m) => m.id === id).label,
        values: lengths.map((len) => measurement(latency, {decoder: state.decoder, metric: "endToEndS", method: id, length: len})),
        className: `series--${id}`,
        emphasis: id === "boa",
      })),
    });
    chart.setVisible(visibleIds());
    chart.setMarker(state.index);
  };

  const update = ({tween}) => {
    const len = lengths[state.index];
    conditions.textContent = conditionsText(latency, state.decoder);
    const tokens = `${formatTokens(len)} tokens`;
    length.value.textContent = tokens;
    length.input.setAttribute("aria-valuetext", tokens);

    for (const block of blocks) {
      const query = {decoder: state.decoder, metric: block.metric, length: len};
      const measured = MAIN_METHODS.some((method) => measurement(latency, {...query, method}) !== null);
      block.list.hidden = !measured;
      block.lede.hidden = !measured;
      block.note.hidden = measured;
      if (!measured) {
        block.note.replaceChildren(...unmeasuredNote(latency, block.metric, block.subject));
        continue;
      }
      const scale = scaleFor(block);
      for (const row of block.rows) {
        row.root.hidden = row.isOther && !state.showOthers;
        const value = measurement(latency, {...query, method: row.id});
        row.value.textContent = value === null ? "Not measured" : block.format(value);
        const percent = value === null || value <= 0 || !scale ? 0 : Math.min(100, Math.max(0, scale(value)));
        setWidth(row.fill, percent, tween);
      }
    }

    const sentence = speedSentence(speedup(latency, {decoder: state.decoder, metric: "endToEndS", length: len, method: "boa", baseline: "full"}));
    const time = blocks.find((b) => b.metric === "endToEndS");
    time.lede.textContent = sentence ?? "";
    chart?.setMarker(state.index);
  };

  length.input.addEventListener("input", () => {
    state.index = Number(length.input.value);
    update({tween: true});
  });
  decoders.addEventListener("change", (event) => {
    state.decoder = event.target.value;
    buildChart();
    if (drawn) chart.draw(animate);
    update({tween: true});
  });
  others.addEventListener("change", () => {
    state.showOthers = others.checked;
    chart.setVisible(visibleIds());
    update({tween: false});
  });
  wide.addEventListener("change", () => {
    buildChart();
    if (drawn) chart.draw(false);
  });

  buildChart();
  update({tween: false});
  onVisible(root, {
    enter: () => {
      if (drawn) return;
      drawn = true;
      chart.draw(animate);
    },
  });
  root.classList.add("is-ready");
}

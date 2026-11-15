import { onVisible, prefersReducedMotion } from "../lib/motion.js";
import { headlines } from "../lib/headlines.js";
import { formatTokens } from "../lib/format.js";
import { lineChart } from "./line-chart.js";

const WIDE_QUERY = "(min-width: 56em)";
const CHART_LAYOUTS = {
  wide: {width: 960, height: 380},
  narrow: {width: 344, height: 340},
};
const COUNT_S = 1.1;
const FULL_CONTEXT = 100;   // normalised F1 of the LLM reading the full context, by definition

// The line colours are keyed by series class; the data calls Activation Beacon "beacon",
// the colour tokens call the same method (w=128) "beacon128".
const SERIES_CLASS = {boa: "series--boa", beacon: "series--beacon128", sac: "series--sac"};

function node(tag, attrs = {}, children = []) {
  const element = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "text") element.textContent = value;
    else element.setAttribute(key, value);
  }
  element.append(...children);
  return element;
}

const oneDecimal = (x) => x.toFixed(1);

function headlineItems(data) {
  const {results, latency} = data;
  const h = headlines(results, latency);
  const conditions = `in-domain MRQA, ${results.decoder}`;
  const items = [
    {
      value: h.retained.percent,
      suffix: "%",
      text: `of the full-context F1 range kept at ${h.retained.ratio}× compression (normalised F1, ${conditions}).`,
    },
  ];
  if (h.speedups.length > 0) {
    const [first, ...rest] = h.speedups;
    const more = rest.map((s) => ` ${s.factor}× with ${s.decoder}.`).join("");
    const approximate = latency.approximate === true ? " Approximate, read from the paper's figure." : "";
    items.push({
      value: first.factor,
      suffix: "×",
      text: `lower end-to-end latency than Activation Beacon (w=128) at ${formatTokens(first.tokens)} tokens, with ${first.decoder}.${more}${approximate}`,
    });
  }
  items.push({
    value: h.emDrop.points,
    decimals: 1,
    text: `exact-match points lost from ${h.emDrop.from}× to ${h.emDrop.to}× compression (${conditions}).`,
  });
  return items;
}

function renderHeadlines(host, items) {
  const counters = [];
  host.replaceChildren(...items.map((item) => {
    const shown = item.decimals ? item.value.toFixed(item.decimals) : String(item.value);
    const figure = node("span", {class: "headline__value", "aria-hidden": "true"});
    figure.textContent = `${shown}${item.suffix ?? ""}`;
    counters.push({figure, item});
    return node("li", {class: "headline"}, [
      node("span", {class: "headline__sr"}, [`${shown}${item.suffix ?? ""} `]),
      figure,
      node("span", {class: "headline__text", text: item.text}),
    ]);
  }));
  return counters;
}

function countUp(counters, gsap) {
  for (const {figure, item} of counters) {
    const decimals = item.decimals ?? 0;
    const state = {n: 0};
    gsap.to(state, {
      n: item.value,
      duration: COUNT_S,
      ease: "power2.out",
      onUpdate: () => { figure.textContent = `${state.n.toFixed(decimals)}${item.suffix ?? ""}`; },
      onComplete: () => { figure.textContent = `${item.value.toFixed(decimals)}${item.suffix ?? ""}`; },
    });
  }
}

export function mount(root, data) {
  const {results} = data;
  const stage = root.querySelector(".figure__stage");
  const gsap = window.gsap;
  const animate = Boolean(gsap) && !prefersReducedMotion();
  const wide = window.matchMedia(WIDE_QUERY);
  const {ratios, methods} = results;
  const byRatio = results.inDomain.byRatio;

  const host = node("div", {class: "results__chart-host"});
  stage.append(host);

  let chart = null;
  let drawn = false;
  const buildChart = () => {
    const name = wide.matches ? "wide" : "narrow";
    host.className = `results__chart-host results__chart-host--${name}`;
    host.replaceChildren();
    chart = lineChart(host, {
      ...CHART_LAYOUTS[name],
      xScale: "point",
      yScale: "linear",
      xValues: ratios,
      xFormat: (r) => `${r}×`,
      yFormat: (v) => (Number.isInteger(v) ? String(v) : oneDecimal(v)),
      yLabel: "Normalised F1",
      xLabel: "Compression ratio",
      series: [
        ...methods.map(({id, label}) => ({
          id,
          label,
          values: ratios.map((r) => byRatio[r][id].normF1),
          className: SERIES_CLASS[id] ?? "",
          emphasis: id === "boa",
        })),
        {id: "full", label: "Full context", values: ratios.map(() => FULL_CONTEXT), className: "series--full"},
      ],
    });
  };
  wide.addEventListener("change", () => {
    buildChart();
    if (drawn) chart.draw(false);
  });
  buildChart();

  // Headline numbers sit above the figure, in the section.
  const headlineHost = root.closest("section")?.querySelector("[data-headlines]");
  const counters = headlineHost ? renderHeadlines(headlineHost, headlineItems(data)) : [];
  if (headlineHost) headlineHost.hidden = false;

  if (animate) for (const {figure, item} of counters) figure.textContent = `${(0).toFixed(item.decimals ?? 0)}${item.suffix ?? ""}`;

  let counted = false;
  onVisible(root, {
    enter: () => {
      if (!drawn) {
        drawn = true;
        chart.draw(animate);
      }
    },
  });
  if (counters.length > 0) {
    onVisible(headlineHost, {
      enter: () => {
        if (counted) return;
        counted = true;
        if (animate) countUp(counters, gsap);
      },
    });
  }
  root.classList.add("is-ready");
}

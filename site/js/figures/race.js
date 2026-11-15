import { el } from "../lib/svg.js";
import { onVisible, prefersReducedMotion } from "../lib/motion.js";
import { keptIndices } from "../lib/compression.js";
import { measurement, speedup } from "../lib/latency.js";
import { formatFactor, formatSeconds, formatTokensExact } from "../lib/format.js";
import { rowWidth, tokenLeft, tokenRow } from "./tokens.js";

// Diagram parameters, not measurements.
const CONTEXT_TOKENS = 32;
const STRIDE = 8;
const ANSWER_TOKENS = 5;

// The measurement the race replays.
const DECODER = "qwen3-1.7b";
const METRIC = "endToEndS";
const LENGTH = 131072;

// Timeline, in seconds of animation.
const SLOW_LANE_MAX_S = 7;   // the slower lane never takes longer than this
const ANSWER_MAX_S = 0.45;   // writing the answer, the same in both lanes
const FADE_S = 0.4;          // live layer fades in at the start and out after the hold
const REST_S = 0.4;          // still frame before both lanes start
const HOLD_S = 2.5;          // final frame held before the loop restarts
const FACTOR_S = 0.3;

const WIDE_QUERY = "(min-width: 56em)";

// Lane geometry in viewBox units. Lanes are stacked; every y is relative to the lane's top.
// Tokens land in a slot row along the top of the LLM box, above its label, so they never cross it.
const LAYOUTS = {
  wide: {
    width: 960, laneHeight: 156, laneGap: 36, fullLift: 20,
    token: {size: 14, gap: 5}, rowX: 12, rowY: 52, slotY: 106,
    font: {lane: 17, box: 15, small: 14, clock: 30, factor: 15},
    laneLabel: {x: 0, y: 15},
    mamba: {x: 0, y: 26, width: 627, height: 48, labelY: 45},
    llm: {x: 0, y: 96, width: 627, height: 52, labelY: 140},
    answer: {x: 690, y: 115, size: 14, gap: 5, label: {x: 690, y: 104, anchor: "start"}},
    arrow: {x1: 638, x2: 676, y: 122},
    clock: {x: 960, y: 133, label: {y: 104, text: "Time to answer"}},
    factor: {x: 960, y: 155},
  },
  narrow: {
    width: 344, laneHeight: 150, laneGap: 26, fullLift: 20,
    token: {size: 8.5, gap: 2}, rowX: 5, rowY: 48, slotY: 90,
    font: {lane: 13.5, box: 12.5, small: 12, clock: 19, factor: 12},
    laneLabel: {x: 0, y: 15},
    mamba: {x: 0, y: 26, width: 344, height: 40, labelY: 41},
    llm: {x: 0, y: 84, width: 344, height: 40, labelY: 118},
    answer: {x: 64, y: 136, size: 10, gap: 3, label: {x: 0, y: 145, anchor: "start"}},
    arrow: null,
    clock: {x: 344, y: 18, label: null, inHeader: true},
    factor: {x: 344, y: 145},
  },
};

const LANES = [
  {id: "full", label: "Full context"},
  {id: "boa", label: "Boa"},
];

// Lane durations keep the measured ratio: boaS / fullS equals 1 / speedup.
function raceModel(latency) {
  const query = {decoder: DECODER, metric: METRIC, length: LENGTH};
  const full = measurement(latency, {...query, method: "full"});
  const boa = measurement(latency, {...query, method: "boa"});
  const factor = speedup(latency, {...query, method: "boa", baseline: "full"});
  if (full === null || boa === null || factor === null) {
    throw new Error(`No ${METRIC} measurement for ${DECODER} at ${LENGTH} tokens`);
  }
  const timeScale = Math.min(1, SLOW_LANE_MAX_S / Math.max(full, boa));
  const fullS = full * timeScale;
  const boaS = fullS / factor;
  return {
    measured: {full, boa},
    duration: {full: fullS, boa: boaS},
    answerS: Math.min(ANSWER_MAX_S, 0.2 * Math.min(fullS, boaS)),
    factor,
    timeScale,
    start: FADE_S + REST_S,
    decoderLabel: latency.decoders[DECODER].label,
    ratio: latency.ratio,
    generatedTokens: latency.generatedTokens,
  };
}

function text(content, attrs) {
  return el("text", attrs, [content]);
}

// While counting, small values keep one decimal instead of switching to milliseconds.
function clockText(value) {
  return value < 1 ? `${value.toFixed(1)} s` : formatSeconds(value);
}

function buildScene(layout, model) {
  const L = layout;
  const row = {x: L.rowX, ...L.token};
  const width = rowWidth(CONTEXT_TOKENS, row);
  const kept = new Set(keptIndices(CONTEXT_TOKENS, STRIDE));
  // The full-context lane has no encoder, so its body moves up into the space Mamba-2 takes below.
  const laneTops = [0, L.laneHeight - L.fullLift + L.laneGap];
  const height = laneTops[1] + L.laneHeight;
  const svg = el("svg", {
    viewBox: `0 0 ${L.width} ${height}`,
    class: `race race--${L.name}`,
    role: "img",
    "aria-label": `The same frozen LLM answers in ${formatSeconds(model.measured.full)} when it reads the full context and in ${formatSeconds(model.measured.boa)} when it reads the context compressed by Boa.`,
  });
  const scene = {svg, live: [], dropY: L.slotY - L.rowY, lanes: {}};

  LANES.forEach(({id, label}, i) => {
    const isBoa = id === "boa";
    const lane = el("g", {transform: `translate(0 ${laneTops[i]})`});
    const body = el("g", {transform: `translate(0 ${isBoa ? 0 : -L.fullLift})`});
    lane.append(text(label, {x: L.laneLabel.x, y: L.laneLabel.y, class: `race__lane race__lane--${id}`, "font-size": L.font.lane}), body);

    // Static layer: boxes and labels.
    if (isBoa) {
      const {x, y, width: w, height: h} = L.mamba;
      body.append(
        el("defs", {}, [el("clipPath", {id: "race-mamba-clip"}, [el("rect", {x, y, width: w, height: h, rx: 6})])]),
        el("rect", {x, y, width: w, height: h, rx: 6, class: "block block--mamba"}),
        text("Mamba-2", {x: L.rowX, y: L.mamba.labelY, class: "diagram-label", "font-size": L.font.box}),
      );
    }
    body.append(
      el("rect", {x: L.llm.x, y: L.llm.y, width: L.llm.width, height: L.llm.height, rx: 8, class: "block block--llm"}),
      text("Transformer LLM (frozen)", {x: L.rowX, y: L.llm.labelY, class: "diagram-label", "font-size": L.font.box}),
      text("Answer", {x: L.answer.label.x, y: L.answer.label.y, class: "diagram-label diagram-label--muted", "font-size": L.font.small}),
    );
    if (L.arrow) {
      const {x1, x2, y} = L.arrow;
      body.append(el("path", {d: `M${x1} ${y}H${x2}M${x2 - 6} ${y - 4}L${x2} ${y}L${x2 - 6} ${y + 4}`, class: "race__arrow"}));
    }
    if (L.clock.label) {
      body.append(text(L.clock.label.text, {x: L.clock.x, y: L.clock.label.y, "text-anchor": "end", class: "diagram-label diagram-label--muted", "font-size": L.font.small}));
    }

    // Live layer: everything the timeline moves.
    const live = el("g", {class: "race__live"});
    let band = null;
    if (isBoa) {
      const bandWidth = 3 * (L.token.size + L.token.gap);
      band = el("rect", {
        x: L.rowX - bandWidth, y: L.mamba.y, width: bandWidth, height: L.mamba.height,
        class: "race__sweep", opacity: 0,
      });
      live.append(el("g", {"clip-path": "url(#race-mamba-clip)"}, [band]));
      scene.sweep = {band, travel: width + bandWidth, width};
    }
    live.append(tokenRow(CONTEXT_TOKENS, {...row, y: L.rowY, className: "token token--ghost"}));
    const tokenGroup = tokenRow(CONTEXT_TOKENS, {...row, y: L.rowY, className: "token token--context"});
    const answerGroup = tokenRow(ANSWER_TOKENS, {...L.answer, className: "token token--answer"});
    for (const answer of answerGroup.children) answer.setAttribute("opacity", 0);
    const clock = text("0.0 s", {x: L.clock.x, y: L.clock.y, "text-anchor": "end", class: `race__clock race__clock--${id}`, "font-size": L.font.clock});
    live.append(tokenGroup, answerGroup);
    // A clock in the lane header stays put; one beside the answer moves with the body.
    const clockLayer = L.clock.inHeader ? el("g", {class: "race__live"}, [clock]) : live;
    if (clockLayer === live) live.append(clock);

    let factor = null;
    if (isBoa) {
      factor = text(`${formatFactor(model.factor)} faster`, {
        x: L.factor.x, y: L.factor.y, "text-anchor": "end", class: "race__factor", "font-size": L.font.factor, opacity: 0,
      });
      live.append(factor);
    }

    body.append(live);
    svg.append(lane);
    scene.live.push(live);
    if (clockLayer !== live) {
      lane.append(clockLayer);
      scene.live.push(clockLayer);
    }

    const tokens = [...tokenGroup.children];
    scene.lanes[id] = {
      tokens,
      kept: tokens.filter((t) => kept.has(Number(t.dataset.index))),
      dropped: tokens.filter((t) => !kept.has(Number(t.dataset.index))),
      centre: (t) => tokenLeft(Number(t.dataset.index), row) + L.token.size / 2 - L.rowX,
      answers: [...answerGroup.children],
      clock,
      factor,
    };
  });
  return scene;
}

function updateClocks(scene, model, time) {
  for (const id of ["full", "boa"]) {
    const progress = Math.min(1, Math.max(0, (time - model.start) / model.duration[id]));
    const value = progress === 1 ? formatSeconds(model.measured[id]) : clockText(progress * model.measured[id]);
    const {clock} = scene.lanes[id];
    if (clock.textContent !== value) clock.textContent = value;
  }
}

function cssColour(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

// Phases (t0 = model.start, both lanes start together):
//   0 .. FADE_S                 live layer fades in
//   t0 .. t0 + full             full context: tokens enter the LLM one by one, then the answer
//   t0 .. t0 + boa              Boa: sweep (kept turn teal, others fade), kept enter the LLM, then the answer
//   end of slower lane          speedup label fades in; this is the final frame
//   + HOLD_S, then FADE_S       hold, fade out, restart (loop only)
function buildTimeline(gsap, scene, model, {loop}) {
  const tl = gsap.timeline({paused: true, repeat: loop ? -1 : 0});
  const tween = (target, from, to, at) => tl.fromTo(target, from, {...to, immediateRender: false}, at);
  const t0 = model.start;
  const {full, boa} = scene.lanes;

  tween(scene.live, {opacity: 0}, {opacity: 1, duration: FADE_S, ease: "none"}, 0);

  const answer = (tokens, end) => {
    const step = model.answerS / tokens.length;
    tokens.forEach((token, j) => tween(token,
      {opacity: 0, scale: 0.6, transformOrigin: "50% 50%"},
      {opacity: 1, scale: 1, duration: step, ease: "power2.out"},
      end - model.answerS + j * step));
  };

  // Full context: every token enters the LLM, at an even pace across the reading time.
  const fullRead = model.duration.full - model.answerS;
  const fullDrop = Math.min(0.4, fullRead / 4);
  full.tokens.forEach((token, i) => {
    const at = t0 + (fullRead - fullDrop) * (i / (full.tokens.length - 1));
    tween(token, {y: 0}, {y: scene.dropY, duration: fullDrop, ease: "power2.inOut"}, at);
  });
  answer(full.answers, t0 + model.duration.full);

  // Boa: one sweep through Mamba-2; each token is kept or let go as the sweep passes it.
  const boaWork = model.duration.boa - model.answerS;
  const sweepS = boaWork * 0.6;
  const enterS = boaWork * 0.4;
  const {band, travel} = scene.sweep;
  const edge = Math.min(0.12, sweepS / 6);
  tween(band, {opacity: 0}, {opacity: 1, duration: edge, ease: "none"}, t0);
  tween(band, {x: 0}, {x: travel, duration: sweepS, ease: "none"}, t0);
  tween(band, {opacity: 1}, {opacity: 0, duration: edge, ease: "none"}, t0 + sweepS - edge);
  tween(band, {x: travel}, {x: 0, duration: 0}, t0 + sweepS);   // park it again, out of the way
  const passes = (token) => t0 + sweepS * (boa.centre(token) / travel);
  const fate = Math.min(0.2, sweepS / 5);
  const colour = {context: cssColour("--context"), soft: cssColour("--kv"), ink: cssColour("--ink"), softStroke: cssColour("--kv-stroke")};
  for (const token of boa.kept) {
    tween(token, {fill: colour.context, stroke: colour.ink}, {fill: colour.soft, stroke: colour.softStroke, duration: fate, ease: "none"}, passes(token));
  }
  for (const token of boa.dropped) {
    tween(token, {opacity: 1}, {opacity: 0, duration: fate, ease: "none"}, passes(token));
  }
  const stagger = enterS * 0.1;
  boa.kept.forEach((token, k) => {
    tween(token, {y: 0}, {y: scene.dropY, duration: enterS - (boa.kept.length - 1) * stagger, ease: "power2.inOut"}, t0 + sweepS + k * stagger);
  });
  answer(boa.answers, t0 + model.duration.boa);

  const finish = t0 + Math.max(model.duration.full, model.duration.boa);
  tween(boa.factor, {opacity: 0}, {opacity: 1, duration: FACTOR_S, ease: "none"}, finish);
  tl.addLabel("final", finish + FACTOR_S);
  if (loop) tween(scene.live, {opacity: 1}, {opacity: 0, duration: FADE_S, ease: "none"}, finish + FACTOR_S + HOLD_S);

  tl.eventCallback("onUpdate", () => updateClocks(scene, model, tl.time()));
  return tl;
}

// The final frame without GSAP: same end values the timeline reaches.
function showFinalFrame(scene, model) {
  const {full, boa} = scene.lanes;
  const drop = `translate(0 ${scene.dropY})`;
  for (const token of full.tokens) token.setAttribute("transform", drop);
  for (const token of boa.dropped) token.style.opacity = "0";
  for (const token of boa.kept) {
    token.classList.replace("token--context", "token--soft");
    token.setAttribute("transform", drop);
  }
  for (const token of [...full.answers, ...boa.answers]) token.style.opacity = "1";
  boa.factor.style.opacity = "1";
  updateClocks(scene, model, Infinity);
}

// Says where the lane times come from, filled from the data the race replays.
function describeSetting(model) {
  const times = `Boa at ${model.ratio}× compression takes ${formatSeconds(model.measured.boa)}, the same LLM reading the full context ${formatSeconds(model.measured.full)}, ${formatFactor(model.factor)} faster`;
  const speed = model.timeScale < 1 ? `The animation is sped up ${formatFactor(1 / model.timeScale)} in proportion, and` : "The animation runs in real time, and";
  return [
    `Lane times are the measured end-to-end latencies (context processing plus ${model.generatedTokens} generated tokens) for ${model.decoderLabel} at ${formatTokensExact(LENGTH)} context tokens: ${times}.`,
    `${speed} its ${CONTEXT_TOKENS} squares stand for those ${formatTokensExact(LENGTH)} tokens, so it is a schematic, not one square per token.`,
  ].join(" ");
}

function button(label) {
  const node = document.createElement("button");
  node.type = "button";
  node.className = "button button--small";
  node.textContent = label;
  return node;
}

export function mount(root, data) {
  const model = raceModel(data.latency);
  const stage = root.querySelector(".figure__stage");
  const setting = root.querySelector("[data-race-setting]");
  const gsap = window.gsap;
  const reduced = prefersReducedMotion();
  const loop = Boolean(gsap) && !reduced;
  const wide = window.matchMedia(WIDE_QUERY);

  const replay = button("Replay");
  const pause = button("Pause");
  const controls = document.createElement("div");
  controls.className = "figure__controls";
  controls.append(replay, pause);
  replay.hidden = !gsap;
  pause.hidden = !loop;

  let scene = null;
  let tl = null;
  let visible = false;
  let userPaused = false;
  let wantsPlay = loop;   // reduced motion plays only after Replay, once

  const sync = () => {
    if (!tl) return;
    if (visible && wantsPlay) tl.play();
    else tl.pause();
  };

  const render = () => {
    tl?.kill();
    const layout = {name: wide.matches ? "wide" : "narrow", ...LAYOUTS[wide.matches ? "wide" : "narrow"]};
    const next = buildScene(layout, model);
    if (scene) scene.svg.replaceWith(next.svg);
    else stage.prepend(next.svg);
    scene = next;
    if (!gsap) {
      tl = null;
      showFinalFrame(scene, model);
      return;
    }
    tl = buildTimeline(gsap, scene, model, {loop});
    if (loop) {
      // The first frame of the loop, so the fade-in does not start from a flash of content.
      gsap.set(scene.live, {opacity: 0});
      updateClocks(scene, model, 0);
    } else {
      tl.progress(1);
      updateClocks(scene, model, Infinity);
      tl.eventCallback("onComplete", () => { wantsPlay = false; });
    }
    sync();
  };

  replay.addEventListener("click", () => {
    if (!tl) return;
    userPaused = false;
    wantsPlay = true;
    pause.textContent = "Pause";
    tl.restart();
    sync();
  });
  pause.addEventListener("click", () => {
    userPaused = !userPaused;
    wantsPlay = !userPaused;
    pause.textContent = userPaused ? "Play" : "Pause";
    sync();
  });

  stage.append(controls);
  render();
  wide.addEventListener("change", render);
  if (setting) setting.textContent = describeSetting(model);
  onVisible(root, {
    enter: () => { visible = true; sync(); },
    leave: () => { visible = false; sync(); },
  });
  root.classList.add("is-ready");
}

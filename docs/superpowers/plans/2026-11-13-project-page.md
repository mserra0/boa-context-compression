# Boa Project Page Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a static, animated and interactive project page for the Boa paper in `site/`, ready to publish on GitHub Pages.

**Architecture:** One hand-written HTML page. Each figure is an ES module under `site/js/figures/` that receives a root element and loaded data and draws SVG directly. DOM-free logic lives in `site/js/lib/` and is unit-tested with Node's test runner. Numbers live in JSON under `site/data/`.

**Tech Stack:** HTML, CSS (custom properties, no preprocessor), JavaScript ES modules, inline SVG, GSAP 3 + ScrollTrigger (vendored), Node >= 20 built-in test runner (`node --test`), GitHub Actions for Pages.

**Spec:** `docs/superpowers/specs/2026-11-13-project-page-design.md`

## Global Constraints

- Everything ships from `site/` as written: no bundler, no transpiler, no `package.json` dependencies at run time.
- No CDN at run time: GSAP in `site/vendor/`, fonts in `site/fonts/` (Source Serif 4 variable for text and headings, Inter variable for figure labels and controls, both woff2, both under the SIL Open Font License; licence files kept next to them).
- Light theme only. Colours are CSS custom properties in `site/styles/tokens.css`, sampled from the figures in `../resources_boa_ssm/`.
- Every number shown comes from `site/data/results.json` or `site/data/latency.json`. No number is typed into HTML or figure code.
- While `latency.json` has `"approximate": true`, the cost section shows the text "Approximate values, read from the paper's figure."
- Copy must not claim Boa wins everywhere. The limitations section is mandatory.
- `prefers-reduced-motion: reduce`: every figure renders its final state, no timeline plays, controls still work.
- Native scrolling is never intercepted. ScrollTrigger only starts, pauses or scrubs figures.
- All controls are native elements (`input type="range"`, `button`, `input type="radio"`) with visible labels and keyboard operation.
- Page and UI copy in English, plain register, no exclamation marks.
- Work on branch `feat/project-page`. Conventional commits, no AI attribution lines. No push.
- Commands use `rg`, `fd`, `bat`, `eza`, `sd` rather than `grep`, `find`, `cat`, `ls`, `sed`.
- Local preview: `python3 -m http.server 8000 --directory site` then `http://localhost:8000/`.

## Review Focus

1. **Page opened without the data files** (JSON request fails, or the page is opened from `file://`): each figure's container shows "This figure needs the page to be served over HTTP." or a short load-failure sentence; the text of the page stays readable. Pinned in Task 1 (`loadData` rejects with a typed error) and Task 2 (shell handles it).
2. **Metric missing for a decoder** (memory with Qwen3-4B): `measurement` returns `null`, the memory block hides with a one-line note, nothing renders `NaN` or `undefined`. Pinned in Task 1 and Task 4.
3. **Narrow screens (360 px wide)**: no horizontal page scroll; figures scale or stack; slider thumb is reachable by touch. Checked in every figure task's browser step.
4. **Reduced motion**: final states render and no timeline runs. Pinned in Task 2 (`prefersReducedMotion` helper) and checked per figure.
5. **JavaScript disabled or a figure module throwing**: headings, text, table values and BibTeX remain in the HTML; one figure failing does not stop the others. Pinned in Task 2 (`mountFigures` isolates failures; test included).

---

### Task 1: Data files and tested logic

**Files:**
- Create: `site/data/results.json`, `site/data/latency.json`
- Create: `site/js/lib/scales.js`, `site/js/lib/compression.js`, `site/js/lib/latency.js`, `site/js/lib/format.js`, `site/js/lib/data.js`
- Create: `site/tests/scales.test.js`, `site/tests/compression.test.js`, `site/tests/latency.test.js`, `site/tests/format.test.js`, `site/tests/data-files.test.js`
- Create: `site/package.json` (only `"type": "module"`, `"private": true` and `"scripts": {"test": "node --test tests/"}`)

**Interfaces (produced):**

```js
// scales.js
linearScale([d0, d1], [r0, r1]) -> (x: number) => number
logScale([d0, d1], [r0, r1])    -> (x: number) => number   // base 10, d0 > 0

// compression.js
compressedLength(length: number, ratio: number) -> number        // ceil(length / ratio)
keptIndices(length: number, ratio: number) -> number[]           // 1-based, ascending, always includes `length`

// latency.js
measurement(data, {decoder, metric, method, length}) -> number | null
speedup(data, {decoder, metric, length, method, baseline}) -> number | null  // baseline / method

// format.js
formatSeconds(seconds) -> string     // 0.082 -> "82 ms", 1.234 -> "1.2 s", 12.6 -> "13 s", 125 -> "125 s"
formatGigabytes(gb) -> string        // 6.04 -> "6.0 GB", 47 -> "47 GB"
formatTokens(n) -> string            // 512 -> "512", 1024 -> "1k", 131072 -> "128k", 240000 -> "240k"
formatFactor(x) -> string            // 19.3 -> "19×", 1.84 -> "1.8×"

// data.js
class DataLoadError extends Error {}            // .url
loadData(baseUrl = "data/") -> Promise<{results, latency}>   // rejects with DataLoadError
```

`latency.json` shape:

```json
{
  "approximate": true,
  "ratio": 8,
  "generatedTokens": 64,
  "lengths": [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 240000],
  "methods": [
    {"id": "boa", "label": "Boa"},
    {"id": "sac", "label": "SAC"},
    {"id": "beacon128", "label": "Activation Beacon (w=128)"},
    {"id": "beacon2048", "label": "Activation Beacon (w=2048)"},
    {"id": "full", "label": "Full context"}
  ],
  "decoders": {
    "qwen3-1.7b": {
      "label": "Qwen3-1.7B",
      "endToEndS": {"boa": [], "sac": [], "beacon128": [], "beacon2048": [], "full": []},
      "contextS":  {"boa": [], "sac": [], "beacon128": [], "beacon2048": [], "full": []},
      "memoryGB":  {"boa": [], "sac": [], "beacon128": [], "beacon2048": [], "full": []}
    },
    "qwen3-4b": {
      "label": "Qwen3-4B",
      "endToEndS": {"boa": [], "sac": [], "beacon128": [], "beacon2048": [], "full": []}
    }
  }
}
```

Each array has ten values aligned with `lengths`. `metric` is one of `endToEndS`, `contextS`, `memoryGB`.

`results.json` shape:

```json
{
  "decoder": "Qwen3-1.7B",
  "ratios": [4, 8, 16],
  "methods": [{"id": "boa", "label": "Boa"}, {"id": "beacon", "label": "Activation Beacon"}, {"id": "sac", "label": "SAC"}],
  "inDomain":    {"full": {"em": 73.61, "f1": 83.74}, "none": {"em": 9.11, "f1": 13.46},
                  "byRatio": {"4": {"boa": {"em": 70.14, "f1": 81.29, "normEm": 94.62, "normF1": 96.51}}}},
  "outOfDomain": {"full": {"em": 60.27, "f1": 71.59}, "none": {}, "byRatio": {}}
}
```

- [ ] **Step 1: Branch.** `git switch -c feat/project-page`. Confirm `node --version` is 20 or newer.
- [ ] **Step 2: Fill `results.json`** from Tables 1 and 2 in `../Linear_Time_Context_Compression_with_State_Space_Encoding/` (`main.tex`, `Sections/`): average EM, F1, normalised EM and F1 for Boa, Activation Beacon and SAC at 4x, 8x, 16x, plus the full-context and no-context rows, in-domain and out-of-domain. Copy values exactly; where the source has no value, use `null`.
- [ ] **Step 3: Fill `latency.json`** by reading the ten points per curve from `../resources_boa_ssm/end_to_end_pair_vertical(2).png` (end-to-end, both decoders) and `../resources_boa_ssm/latency_memory_strip.png` (context processing and memory, Qwen3-1.7B). Axes are logarithmic for time and linear for memory. Keep `"approximate": true`. Sanity anchors that must hold: at 240000 tokens `beacon128 / boa` end-to-end is between 18 and 20 for `qwen3-1.7b` and between 38 and 40 for `qwen3-4b`; full-context memory at 240000 is between 45 and 49 GB.
- [ ] **Step 4: Write the failing tests.**

```js
// scales.test.js
assert.equal(linearScale([0, 10], [0, 100])(5), 50);
assert.equal(linearScale([0, 10], [100, 0])(2.5), 75);
assert.ok(Math.abs(logScale([1, 100], [0, 200])(10) - 100) < 1e-9);
assert.equal(logScale([1, 1000], [0, 300])(1), 0);

// compression.test.js
assert.equal(compressedLength(16, 4), 4);
assert.equal(compressedLength(17, 4), 5);
assert.equal(compressedLength(3, 16), 1);
assert.deepEqual(keptIndices(16, 4), [4, 8, 12, 16]);
assert.deepEqual(keptIndices(10, 4), [2, 6, 10]);
assert.deepEqual(keptIndices(3, 16), [3]);
assert.equal(keptIndices(24, 8).length, compressedLength(24, 8));

// latency.test.js (fixture: two lengths [512, 1024], boa [1, 2], full [1, 6], no memoryGB on "b")
assert.equal(measurement(fx, {decoder: "a", metric: "endToEndS", method: "boa", length: 1024}), 2);
assert.equal(measurement(fx, {decoder: "b", metric: "memoryGB", method: "boa", length: 512}), null);
assert.equal(measurement(fx, {decoder: "a", metric: "endToEndS", method: "boa", length: 700}), null);
assert.equal(measurement(fx, {decoder: "zzz", metric: "endToEndS", method: "boa", length: 512}), null);
assert.equal(speedup(fx, {decoder: "a", metric: "endToEndS", length: 1024, method: "boa", baseline: "full"}), 3);
assert.equal(speedup(fx, {decoder: "b", metric: "memoryGB", length: 512, method: "boa", baseline: "full"}), null);

// format.test.js
assert.equal(formatSeconds(0.082), "82 ms");
assert.equal(formatSeconds(1.234), "1.2 s");
assert.equal(formatSeconds(12.6), "13 s");
assert.equal(formatSeconds(125), "125 s");
assert.equal(formatGigabytes(6.04), "6.0 GB");
assert.equal(formatGigabytes(47), "47 GB");
assert.equal(formatTokens(512), "512");
assert.equal(formatTokens(1024), "1k");
assert.equal(formatTokens(131072), "128k");
assert.equal(formatTokens(240000), "240k");
assert.equal(formatFactor(19.3), "19×");
assert.equal(formatFactor(1.84), "1.8×");

// data-files.test.js (reads the real JSON with fs)
// - every array under decoders.*.{endToEndS,contextS,memoryGB}.* has lengths.length entries, all finite and > 0
// - every method id in `methods` is a key of every metric object present
// - the three sanity anchors from Step 3
// - results.json: inDomain.byRatio["16"].boa.normF1 === 94.91, inDomain.full.f1 === 83.74,
//   outOfDomain.byRatio["16"].boa.f1 === 61.91, every ratio in `ratios` has all three methods
// - loadData with a fetch stub returning {ok: false} rejects with DataLoadError carrying the url
```

- [ ] **Step 5: Run `cd site && node --test tests/`.** Expected: failures for missing modules.
- [ ] **Step 6: Implement the five lib modules** to the signatures above. `loadData` uses `fetch` for `results.json` and `latency.json` relative to `baseUrl`.
- [ ] **Step 7: Run `cd site && node --test tests/`.** Expected: all pass.
- [ ] **Step 8: Commit.** `git add site docs/superpowers && git commit -m "feat(site): add page data and tested figure logic"`

---

### Task 2: Page shell, typography and static content

**Files:**
- Create: `site/index.html`, `site/styles/tokens.css`, `site/styles/base.css`, `site/styles/layout.css`, `site/styles/figures.css`
- Create: `site/js/main.js`, `site/js/lib/motion.js`, `site/js/lib/mount.js`, `site/js/lib/svg.js`
- Create: `site/tests/mount.test.js`
- Create: `site/vendor/gsap.min.js`, `site/vendor/ScrollTrigger.min.js`, `site/vendor/GSAP-LICENSE.txt`, `site/fonts/*`, `site/favicon.svg`

**Interfaces:**
- Consumes: `loadData`, `DataLoadError` from Task 1.
- Produces:

```js
// mount.js
mountFigures(figures: Array<{id: string, mount: (root: Element, data) => void}>, data, {getRoot, onError})
//   calls each mount with getRoot(id); a throwing mount calls onError(id, error) and the rest still mount;
//   a missing root is skipped silently. Returns the ids that mounted.

// motion.js
prefersReducedMotion() -> boolean
onVisible(root: Element, {enter, leave}) -> void     // IntersectionObserver wrapper used to play and pause timelines

// svg.js
el(name: string, attrs?: object, children?: Array<Element|string>) -> SVGElement
```

- Figure roots in `index.html`: `<figure class="figure" id="figure-race">`, `figure-ratio`, `figure-cost`, `figure-method`, `figure-results`. Each contains a `<div class="figure__stage">` (drawn into by its module), a `<figcaption>`, and a `<p class="figure__fallback">` that is visible until the module adds `is-ready` to the figure.
- Section ids: `top`, `concept`, `cost`, `method`, `results`, `limitations`, `citation`.

- [ ] **Step 1: Vendor assets.** Download GSAP 3 core and ScrollTrigger minified builds and the GSAP standard licence into `site/vendor/`; download Source Serif 4 and Inter variable woff2 files and their OFL texts into `site/fonts/`. Record the exact versions in `site/vendor/README.md`.
- [ ] **Step 2: Write the failing test** `mount.test.js`:

```js
const calls = [];
const errors = [];
const figures = [
  {id: "a", mount: () => calls.push("a")},
  {id: "b", mount: () => { throw new Error("boom"); }},
  {id: "c", mount: () => calls.push("c")},
  {id: "missing", mount: () => calls.push("missing")},
];
const mounted = mountFigures(figures, {}, {
  getRoot: (id) => (id === "missing" ? null : {id}),
  onError: (id, error) => errors.push([id, error.message]),
});
assert.deepEqual(calls, ["a", "c"]);
assert.deepEqual(errors, [["b", "boom"]]);
assert.deepEqual(mounted, ["a", "c"]);
```

- [ ] **Step 3: Run `cd site && node --test tests/`.** Expected: `mount.test.js` fails.
- [ ] **Step 4: Implement `mount.js`, `motion.js`, `svg.js`.**
- [ ] **Step 5: Write `tokens.css`.** Sample hex values from `../resources_boa_ssm/Method.svg` fills and the plot PNGs for: `--boa`, `--kv` and `--kv-stroke` (teal), `--mamba` (lilac), `--context` (grey), `--teacher` (light blue), `--student` (light lilac), `--beacon-128`, `--beacon-2048`, `--sac`, `--full` (dashed grey). Add ink, muted ink, rule and paper colours, a type scale, a spacing scale and one easing token.
- [ ] **Step 6: Write `index.html` and the remaining CSS.** Content, in order:
  - Header: title "Linear-Time Context Compression with State-Space Encoding"; the name "Boa" as the project mark; authors Marc Serra Ortega, Andrey Barsky, Carlos Boned Riera, Tomás Ockier, Dimosthenis Karatzas; "Computer Vision Center, Universitat Autònoma de Barcelona"; links Paper (`https://arxiv.org/abs/2610.33006`), Code (`https://github.com/mserra0/boa-context-compression`), BibTeX (`#citation`).
  - The five figure sections with headings and short text per the spec. The results section contains the EM / F1 table as two `<table>` elements (in-domain and out-of-domain). Their bodies are rendered by `main.js` from `results.json`; a `<noscript>` sentence points to the paper for the values.
  - Limitations: two statements from the spec, using the paper's wording as the basis (`Sections/` in the paper source).
  - Citation: BibTeX in a `<pre>` built from `CITATION.cff` fields, plus a "Copy" button.
  - Footer: affiliation, licence, link to the repository.
  - Single text column of about 68 characters; figures may extend to about 1100 px.
- [ ] **Step 7: Write `main.js`.** Loads data, then `mountFigures` with whatever figure modules exist (none yet; the array is filled by later tasks). On `DataLoadError` it writes the fallback sentence from Review Focus 1 into every `.figure__fallback`. Renders the results tables. Wires the copy button (falls back to selecting the text when the Clipboard API is unavailable).
- [ ] **Step 8: Run tests, then check in a browser** at 1280 px and 360 px widths: no horizontal scroll, fonts load from `site/fonts/`, fallbacks visible in each figure, tables populated, copy button works, and with JavaScript disabled all text is readable.
- [ ] **Step 9: Commit.** `git commit -m "feat(site): add page shell, typography and static content"`

---

### Task 3: Race hero and ratio explainer

**Files:**
- Create: `site/js/figures/race.js`, `site/js/figures/ratio.js`, `site/js/figures/tokens.js` (shared token-row drawing)
- Modify: `site/js/main.js` (register both), `site/styles/figures.css`

**Interfaces:**
- Consumes: `el`, `prefersReducedMotion`, `onVisible`, `keptIndices`, `compressedLength`, `measurement`, `speedup`, `formatSeconds`, `formatFactor`, global `gsap`.
- Produces: `export function mount(root, data)` in each figure module; `tokens.js` exports `tokenRow(count, {x, y, size, gap, className}) -> SVGGElement`.

- [ ] **Step 1: `race.js`.** One SVG, two lanes, 32 context tokens each, stride 8.
  - Top lane "Full context": all 32 grey tokens slide into a box labelled "Transformer LLM (frozen)"; a lane clock counts; the answer tokens appear.
  - Bottom lane "Boa": the 32 tokens pass through a lilac box labelled "Mamba-2"; a highlight sweeps once left to right; the four tokens at `keptIndices(32, 8)` turn teal and the others fade; the four teal tokens enter the same LLM box; the answer appears.
  - Lane durations are proportional to real numbers: the ratio of the two lane times equals `speedup(latency, {decoder: "qwen3-1.7b", metric: "endToEndS", length: 131072, method: "boa", baseline: "full"})`, capped so the slow lane takes at most 7 s. Each lane's clock ends on the measured value via `formatSeconds`; the caption states the length and decoder.
  - Loops with a 2.5 s hold. A "Replay" button restarts it. Plays only while visible. Reduced motion: final frame, no loop.
- [ ] **Step 2: `ratio.js`.** A row of 48 context tokens above a row of soft tokens. A radio group "Compression ratio" with 4x, 8x, 16x (8x selected). On change, tokens at `keptIndices(48, r)` drop to the lower row and turn teal, the rest dim; a line reads "48 context tokens → {m} soft tokens". Thin connectors link each kept token to its soft token.
- [ ] **Step 3: Register both in `main.js`; add `is-ready` on success.**
- [ ] **Step 4: Browser check** at 1280 px and 360 px, with reduced motion on and off, keyboard-only operation of the radio group and replay button. `cd site && node --test tests/` still passes.
- [ ] **Step 5: Commit.** `git commit -m "feat(site): add race hero and compression ratio figures"`

---

### Task 4: Time and compute section

**Files:**
- Create: `site/js/figures/cost.js`, `site/js/figures/line-chart.js` (shared SVG line chart)
- Modify: `site/js/main.js`, `site/styles/figures.css`

**Interfaces:**
- Consumes: Task 1 libs, `el`, `prefersReducedMotion`, `onVisible`, `gsap`.
- Produces:

```js
// line-chart.js
lineChart(root, {
  width, height, xScale: "log" | "point", yScale: "log" | "linear",
  xValues: number[], xFormat, yFormat, yLabel, xLabel,
  series: Array<{id, label, values: Array<number|null>, className, emphasis?: boolean}>,
}) -> {setMarker(index: number): void, setVisible(ids: string[]): void, draw(animate: boolean): void}
```

- [ ] **Step 1: `line-chart.js`.** Axes with ticks and light horizontal rules, one `<path>` per series (gaps at `null`), points as circles, a vertical marker with value labels at `setMarker`, a legend built from `series`. `draw(true)` animates stroke length; `draw(false)` renders at once. Points are focusable and show their value on hover, focus and tap.
- [ ] **Step 2: `cost.js`.**
  - Controls: range input "Context length" with ten steps (index 0 to 9, labelled with `formatTokens`, initial index 8 = 128k); radio group "Frozen LLM" (Qwen3-1.7B, Qwen3-4B); checkbox "Show other compressors".
  - Latency block: horizontal bars for full context and Boa (plus SAC and both Activation Beacon variants when the checkbox is on), bar length on a log scale, value via `formatSeconds`, and one sentence: "Boa answers {factor} faster than reading the full context." using `speedup` and `formatFactor`. When the factor is below 1.05, the sentence reads "At this length Boa and the full context take about the same time."
  - Memory block: the same for `memoryGB` with `formatGigabytes`. When `measurement` returns `null` for the selected decoder, the block is replaced by "Peak memory was measured with Qwen3-1.7B only."
  - Below: `lineChart` of end-to-end latency against length for the selected decoder, marker following the slider.
  - If `latency.approximate` is true, show "Approximate values, read from the paper's figure." under the controls.
  - Bars tween between values (0.4 s); reduced motion: values switch at once.
- [ ] **Step 3: Register in `main.js`.**
- [ ] **Step 4: Browser check** at 1280 px and 360 px: slider by mouse, touch emulation and arrow keys; decoder switch hides memory with the note; no `NaN` or `undefined` anywhere at any slider position; reduced motion. Tests still pass.
- [ ] **Step 5: Commit.** `git commit -m "feat(site): add interactive latency and memory comparison"`

---

### Task 5: Method figure

**Files:**
- Create: `site/js/figures/method.js`
- Modify: `site/js/main.js`, `site/styles/figures.css`, `site/index.html` (step captions)

**Interfaces:**
- Consumes: `el`, `tokenRow`, `prefersReducedMotion`, `gsap`, `ScrollTrigger`.
- Produces: `mount(root, data)`.

- [ ] **Step 1: Draw the diagram** as SVG following `../resources_boa_ssm/Method(1).png`: teacher column (context tokens → frozen "Transformer LLM" → teacher output tokens → `p_T(· | x<t, c)`), student column (context tokens → trainable group with "Mamba 2", "Stride op.", three compressed tokens, "K proj" and "V proj" → hatched KV-cache block at the left of the frozen "Transformer LLM" → student output tokens → `p_S(· | x<t, z)`), and the "KL" link between the two distributions. Snowflake and flame as vector icons. Labels are real SVG text. Mathematical labels use `<tspan>` with subscripts, not images.
- [ ] **Step 2: Scroll assembly.** Five steps, each with a caption in an ordered list beside (desktop) or below (phone) the sticky diagram; the active step's parts are at full opacity, earlier parts stay, later parts are hidden:
  1. "Mamba-2 reads the context once, in linear time."
  2. "A temporal stride keeps every r-th hidden state."
  3. "Per-layer K and V projections turn them into a key-value cache."
  4. "The frozen LLM attends to that cache instead of the text."
  5. "Training matches its next-token distribution to the full-context teacher (KL)."
  ScrollTrigger sets the active step from scroll position; the diagram uses `position: sticky`, with no pinning that changes scroll length. Reduced motion or no ScrollTrigger: the full diagram is shown with all five captions.
- [ ] **Step 3: Register; browser check** at 1280 px and 360 px, scrolling both directions, reduced motion. Tests still pass.
- [ ] **Step 4: Commit.** `git commit -m "feat(site): add scroll-assembled method figure"`

---

### Task 6: Results figure, headline numbers and deployment

**Files:**
- Create: `site/js/figures/results.js`, `.github/workflows/pages.yml`
- Modify: `site/js/main.js`, `site/index.html`, `site/styles/figures.css`, `README.md` (one line linking the project page under the paper link)

**Interfaces:**
- Consumes: `lineChart`, `results.json`, `latency.json`, Task 1 formatters.
- Produces: `mount(root, data)`.

- [ ] **Step 1: `results.js`.** `lineChart` with `xScale: "point"` over ratios 4x, 8x, 16x; series Boa (emphasised), Activation Beacon, SAC from `inDomain.byRatio[*][*].normF1`, plus a dashed reference at 100 labelled "Full context"; y axis "Normalised F1 (0 = no context, 100 = full context)". Lines draw when the figure enters view.
- [ ] **Step 2: Headline numbers.** Three blocks rendered from data, each with its condition as a caption:
  - `inDomain.byRatio["16"].boa.normF1` rounded → "95% of the full-context F1 range kept at 16x compression (in-domain MRQA, Qwen3-1.7B)".
  - `speedup` of `beacon128` over `boa`, end-to-end, 240000 tokens, both decoders → "19× lower end-to-end latency than Activation Beacon (w=128) at 240k tokens; 39× with Qwen3-4B". While `latency.approximate` is true the factors are the paper's annotated 19 and 39, stored as `"annotatedSpeedup": {"qwen3-1.7b": 19, "qwen3-4b": 39}` in `latency.json` (add the field and extend `data-files.test.js` to assert it).
  - `inDomain.byRatio["4"].boa.em - inDomain.byRatio["16"].boa.em` → "1.4 EM points lost from 4x to 16x compression (in-domain)".
  Numbers count up once on entering view; reduced motion: shown at once.
- [ ] **Step 3: Tabs for the tables.** Two buttons with `role="tab"` switching the in-domain and out-of-domain tables rendered in Task 2; arrow keys move between tabs.
- [ ] **Step 4: `pages.yml`.** Triggers: push to `main` with paths `site/**` and the workflow file, plus `workflow_dispatch`. Jobs: run `node --test tests/` in `site/`, then `actions/configure-pages`, `actions/upload-pages-artifact` with `path: site`, `actions/deploy-pages`. Permissions: `contents: read`, `pages: write`, `id-token: write`. The whole `site/` folder is uploaded, tests included.
- [ ] **Step 5: Full pass.** `cd site && node --test tests/` passes. In a browser at 1280 px, 768 px and 360 px: whole page top to bottom, reduced motion on and off, keyboard-only pass through every control, page served from a sub-path (`python3 -m http.server 8000` from the repository root, open `http://localhost:8000/site/`) to prove all URLs are relative.
- [ ] **Step 6: Commit.** `git commit -m "feat(site): add results figure, headline numbers and Pages workflow"`

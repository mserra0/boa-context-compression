# Boa project page: design

Date: 2026-11-13. Status: awaiting approval.

## Intent

A public project page for "Linear-Time Context Compression with State-Space Encoding" (arXiv:2610.33006), served from GitHub Pages.

A visitor who leaves after 30 seconds must have understood two things:

1. What context compression into soft prompts is: a long context becomes a short sequence of continuous representations that a frozen LLM reads instead of the text.
2. What it buys compared with the same LLM reading the full context, in time and in compute (GPU memory).

Everything else on the page (method detail, tables, citation) serves visitors who stay longer.

## Goals

- Teach the concept with motion, not with paragraphs.
- Let the visitor drive the time and compute comparison themselves.
- Look clean and academic: quiet typography, generous whitespace, the figure palette of the paper.
- Stay honest: every number on the page is a number from the paper or from the benchmark results.

## Non-goals

- No dark theme in the first version.
- No live model demo, no backend.
- No blog, no multi-page site.
- No reuse of the Figma `Method.svg` as an animation source (outlined text, unnamed groups, embedded rasters). It is the visual reference only.

## Anti-goals

- Overclaiming. The page must not say Boa wins everywhere; the paper reports a weakness on verbatim span extraction (Relation Extraction) and slightly lower long-context F1 than Activation Beacon.
- Numbers presented as measurements that were read off a plot.
- Scroll-jacking, autoplaying video, or motion that ignores `prefers-reduced-motion`.
- A framework or build pipeline that the author must maintain to change one sentence.

## Constraints

- Hosting: GitHub Pages for `mserra0/boa-context-compression`. `docs/` already holds the markdown documentation, so the page cannot live there.
- Static files only.
- Must work on phones; the interactive pieces need touch input.
- Raw latency and memory results are not in the repository (`benchmarks/latency/outputs/` is not committed).

## Approaches

1. **Scroll story with a live race (chosen).** Hand-written HTML, CSS, JavaScript modules and SVG, animated with GSAP. Most hand-crafted work; the only option where the visitor operates the comparison.
2. **Academic template with pre-rendered clips** (Manim or Remotion video in a Nerfies-style layout). Clips are reusable in talks, but nothing is interactive and video is heavy and soft on high-density screens.
3. **Static page with animated accents** over the existing PNG figures. Cheapest; decorates the paper without teaching the concept.

The author chose approach 1 and delegated the remaining design calls ("go fully with your intuition"). The calls below are therefore the designer's, open to correction.

## Design

### Location and deployment (designer's call)

- Page source in `site/` at the repository root.
- A GitHub Actions workflow publishes `site/` to GitHub Pages on pushes to `main` that touch `site/`.
- Resulting URL: `https://mserra0.github.io/boa-context-compression/`.
- No build step: the folder is served as written. GSAP and ScrollTrigger are vendored in `site/vendor/` so the page has no runtime dependency on a CDN.

### Page structure, top to bottom

1. **Header block.** Title, authors (Marc Serra Ortega, Andrey Barsky, Carlos Boned Riera, Tomás Ockier, Dimosthenis Karatzas), affiliation (Computer Vision Center, UAB), links: Paper (arXiv), Code (GitHub), BibTeX (jumps to the citation).
2. **Hero animation: the race.** Two horizontal lanes fed the same row of grey context tokens.
   - Top lane, "Full context": every token enters the Transformer LLM; a clock runs; the answer appears late.
   - Bottom lane, "Boa": tokens pass through the Mamba-2 encoder in one sweep, every r-th state is kept and becomes a teal soft token, the short teal row enters the same LLM; the answer appears early.
   - Loops with a pause; a replay control; one sentence under it states the takeaway.
3. **What is context compression.** A short explainer with one interactive figure: a row of context tokens and a ratio control (4x, 8x, 16x). Changing the ratio changes how many soft tokens remain (`m = ceil(L / r)`), anchored at the final token as in the paper. Two or three sentences of text, no more.
4. **Time and compute, driven by the visitor.** A context-length slider (512 to 240k tokens, logarithmic steps matching the measured lengths).
   - Two bars or clocks: end-to-end latency for full context and for Boa at 8x, with the speedup factor between them.
   - A second pair for peak GPU memory.
   - A toggle reveals the other compressors (SAC, Activation Beacon with both window sizes) so the "order of magnitude faster than Transformer-based compressors" claim is visible.
   - Decoder switch: Qwen3-1.7B and Qwen3-4B.
   - Below it, the full latency curve as an SVG line chart with a marker that follows the slider.
5. **Method.** The overview figure rebuilt as hand-written SVG in the paper palette. It assembles in steps as the visitor scrolls, each step with a one-line caption: Mamba-2 reads the context in one linear pass; temporal stride keeps every r-th hidden state; per-layer K and V projections turn them into a KV cache; the frozen LLM attends to it; training matches the student's next-token distribution to the full-context teacher (KL). Snowflake and flame markers are redrawn as vector icons.
6. **Results.**
   - Animated line chart of normalised F1 against compression ratio for Boa, Activation Beacon and SAC (in-domain MRQA, values from Table 1). Lines draw in when the chart enters view; hovering or tapping a point shows its value.
   - Three headline figures taken from the paper, each with its condition stated next to it.
   - A compact table with in-domain and out-of-domain tabs (average EM / F1 at 4x, 8x, 16x).
7. **Limitations.** Two short, plain statements: weaker verbatim span extraction; long-context results slightly below Activation Beacon, with the paper's explanation.
8. **Citation.** BibTeX with a copy button, built from `CITATION.cff`. Footer with affiliation and licence.

### Motion rules

- GSAP timelines; ScrollTrigger only to start or scrub figures when they enter view. Native scrolling is never taken over.
- With `prefers-reduced-motion`, every figure renders its final state and the interactive controls still work.
- Animations pause when off screen.

### Visual language (designer's call)

- Light theme. Colours come from the paper figures: Boa blue, teal for compressed tokens and KV cache, lilac for the Mamba encoder, grey for context, the baseline reds and purple only inside comparison charts.
- A serif face for headings and body in the manner of a paper, a sans face for figure labels and controls. Fonts self-hosted.
- Single centred column for text; figures may run wider.

### Data

- All numbers live in JSON files under `site/data/`, separate from the drawing code:
  - `results.json`: the table values from the paper.
  - `latency.json`: sequence length, method, decoder, context-processing time, end-to-end time, peak memory.
- **Latency data source (designer's call, needs the author before publishing).** The first version fills `latency.json` with values read from the paper's plots, and the file carries an `approximate: true` flag that makes the page show "approximate, read from the paper's figure" next to that section. Before the page is published, the author replaces the file with the output of `benchmarks/latency` and the flag is removed. The page must not go public with the flag set.

### Sources and coverage (settled after the builder check)

- **Paper text and tables:** LaTeX source at `../Linear_Time_Context_Compression_with_State_Space_Encoding/` (`main.tex`, `supplementary.tex`, `Sections/`). `results.json` is filled from Tables 1 and 2 there.
- **Figures to read values and colours from:** `../resources_boa_ssm/` (`end_to_end_pair_vertical(2).png`, `latency_memory_strip.png`, `gracefull_degradation(1).png`, `Method(1).png`, `Method.svg`). Palette hex values are sampled from these files.
- **Length grid:** 512, 1k, 2k, 4k, 8k, 16k, 32k, 64k, 128k, 240k tokens. The slider snaps to these ten lengths; there is no interpolation between them.
- **Methods:** Boa, SAC, Activation Beacon (w=128), Activation Beacon (w=2048), full context. All five have a point at every length in the figures.
- **Coverage by decoder:** end-to-end latency exists for Qwen3-1.7B and Qwen3-4B. Context-processing time and peak GPU memory exist for Qwen3-1.7B only, so the memory comparison is shown for Qwen3-1.7B and is hidden (with a one-line note) when the visitor switches to Qwen3-4B.
- **Definitions (supplementary, section C):** context-processing time is the elapsed time from raw context tokens to the first generated token, compression included. End-to-end latency adds the decode phase of 64 generated tokens. Boa's figures therefore include its encoding pass. Compression ratio in the latency figures is 8x.
- **Headline figures (designer's call):**
  1. About 95% of the full-context F1 range retained at 16x compression (in-domain MRQA, Qwen3-1.7B; normalised F1 94.91).
  2. 19x lower end-to-end latency than Activation Beacon (w=128) at 240k tokens with Qwen3-1.7B, 39x with Qwen3-4B, as annotated in the paper's figure.
  3. Under 2 points of in-domain EM lost going from 4x to 16x (70.14 to 68.79).
- **Citation:** a hand-written BibTeX string in the page, copied from the fields of `CITATION.cff`; nothing is parsed at run time.
- **GitHub Pages:** not enabled on the repository today. The author enables it with source "GitHub Actions" (or approves doing so) when the page is ready to go public; until then the page is checked locally.

### Code structure

- `site/index.html`, `site/styles/`, `site/js/` with one module per figure (`race`, `ratio`, `cost`, `method`, `results`) and small shared modules for scales, number formatting and data loading.
- Charts are drawn directly as SVG; no charting library.
- Pure logic (linear and logarithmic scales, lookup of a measurement by length, method and decoder, speedup factors, `m = ceil(L / r)`, formatting of seconds and gigabytes) is kept free of DOM access and covered by unit tests run with Node's built-in test runner, written before the logic. Figures themselves are verified in a browser at desktop and phone widths.

## Left to the builder

- Exact typefaces, spacing scale and easing curves.
- Whether the cost section shows bars or clocks, after trying both.
- Wording of captions, within the honesty constraints above.
- Layout of the method figure at phone width (stacked or scaled).

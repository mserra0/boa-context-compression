"""Plot latency vs sequence length from a benchmark results.jsonl.

One curve per adapter: latency (y) against sequence length (x, log2 axis so the
512 -> 64k sweep is evenly spaced). OOM / NaN cells are dropped from a curve.

Two modes:

    # one metric, one figure (single-column, for a specific claim)
    uv run python -m benchmarks.latency.plot \
        --results benchmarks/latency/outputs/results.jsonl \
        --out benchmarks/latency/outputs/latency_curve.png

    # every metric in one figure (the summary panel for a whole sweep)
    uv run python -m benchmarks.latency.plot --panels \
        --results benchmarks/latency/outputs/results.jsonl \
        --out benchmarks/latency/outputs/latency_panels.png

An adapter keeps the same colour, marker and line style in every panel *and in
every figure*, because those are keyed by adapter name rather than by position
(see `_STYLE_BY_ADAPTER`). A sweep that drops one method therefore does not
repaint the rest, and separate figures agree on which curve is which.
"""

import argparse
import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FuncFormatter
import scienceplots  # noqa: F401  registers the 'science' styles (required import)

# Publication style (SciencePlots), applied via a context manager rather than a
# global plt.style.use so importing this module never mutates matplotlib state.
# Order matters — later styles override earlier ones. 'nature' applies Nature
# journal formatting; 'no-latex' drops the system LaTeX dependency.
_STYLE = ["science", "nature", "no-latex"]

# Preferred left-to-right legend order. Ours first, then the compression
# baselines, then the uncompressed reference. Anything unlisted is appended.
_ADAPTER_ORDER = ["ssm", "ssm_kv", "icae", "sac", "beacon", "beacon_long", "full_ctx"]

# --------------------------------------------------------------------------- #
# Visual identity, keyed by adapter NAME rather than by position.              #
#                                                                              #
# Position-keyed colours (`palette[i % len(palette)]`) would let a sweep that   #
# omits one method repaint every method after it, so figures from different     #
# sweeps would disagree about which curve is `sac`. Colour follows the entity.  #
#                                                                              #
# The five compression methods are the first five slots of a categorical        #
# palette validated for the light chart surface (#fcfcfb) in this legend order: #
# lightness band, chroma floor, colour-vision-deficiency separation (worst       #
# adjacent ΔE 22.7 protan), normal-vision separation (worst adjacent ΔE 29.0)    #
# and >= 3:1 contrast against the surface all pass with no warnings.            #
#                                                                              #
# `full_ctx` is deliberately NOT one of those slots: it is the uncompressed     #
# reference the others are measured against, not a competing method, so it is   #
# neutral grey and dashed. Marker shape and line style repeat that identity, so #
# the figure survives greyscale printing and CVD without relying on hue.        #
# --------------------------------------------------------------------------- #
_STYLE_BY_ADAPTER = {
    "ssm":      {"color": "#eb6834", "marker": "h", "linestyle": "-"},
    "ssm_kv":   {"color": "#2a78d6", "marker": "o", "linestyle": "-",
                 "linewidth": 1.5, "zorder": 3},   # ours: the curve the reader must find first
    "icae":     {"color": "#008300", "marker": "^", "linestyle": "-"},
    "sac":      {"color": "#4a3aa7", "marker": "s", "linestyle": "-"},
    "beacon":   {"color": "#e34948", "marker": "D", "linestyle": "-"},
    # Same method, longer compression window -- so it stays in the red family and
    # reads as Beacon's sibling rather than a sixth unrelated method. Separated on
    # three further channels (lightness, marker, dash) because hue alone would not
    # carry it: the pair is the closest one in the palette.
    # Validated as a 6th categorical slot against the #fcfcfb surface: lightness
    # band, chroma floor, adjacent CVD (protan dE 16.9 vs `beacon`), normal-vision
    # floor (19.6) and contrast all pass.
    "beacon_long": {"color": "#8f2321", "marker": "d", "linestyle": "-."},
    "full_ctx": {"color": "#6b6a66", "marker": "X", "linestyle": "--"},
}

# Slots for an adapter with no entry above (a newly added method). Assigned in
# first-appearance order and still stable across the panels of one figure.
_FALLBACK = [
    {"color": "#eda100", "marker": "v", "linestyle": "-"},
    {"color": "#e87ba4", "marker": "P", "linestyle": "-"},
    {"color": "#1baf7a", "marker": "*", "linestyle": "-"},
]

# Legend text. Adapter keys are implementation names; figures should say what the
# method is.
_LABELS = {
    "ssm": "Ours (SSM → soft prompt)",
    "ssm_kv": "Ours (SSM → KV)",
    "icae": "ICAE",
    "sac": "SAC",
    # Both Beacon entries name their window. Two checkpoints of the same method
    # differ only by it, so an unqualified "Activation Beacon" stopped being an
    # identification the moment the second one was measured.
    # Same method twice, so both entries carry the method's full name and differ
    # only in the window. "Beacon-long" read as a separate method, which is the one
    # thing the pair must not suggest: the adapter key stays `beacon_long`, the
    # label does not.
    "beacon": "Activation Beacon (w=128)",
    "beacon_long": "Activation Beacon (w=2048)",
    "full_ctx": "Full context (no compression)",
}

# Recessive grid: present enough to read a value off, never competing with the
# curves. Drawn under the data (`set_axisbelow`), majors darker than minors.
_GRID_MAJOR = {"linewidth": 0.4, "color": "#d7d6d1"}
_GRID_MINOR = {"linewidth": 0.25, "color": "#ebe9e4"}

# Panels of the combined figure, in reading order: the cost of turning context
# into something the decoder can read, what the user waits for, the end-to-end
# total, and what it all fits in.
#
# `t_context_ms` rather than `t_compress_ms`: methods split that cost
# differently. `sac` and `icae` expose a separable `compress()` and report it;
# `ssm_kv`, `beacon` and `full_ctx` fuse compression into the first forward pass
# and report `t_compress = 0`. Plotting the raw field would show three flat zero
# lines and credit those methods with free compression. The sum is the same
# quantity for every method: time from raw tokens to first generated token.
_PANEL_METRICS = ["t_context_ms", "t_ttft_ms", "t_total_ms", "peak_mem_mb"]

# Fields computed from a row rather than read off it.
_DERIVED = {
    "t_context_ms": lambda r: (r.get("t_compress_ms") or 0) + (r.get("t_ttft_ms") or 0),
}

# Panels where a log y axis is the readable choice: latency spans three orders
# of magnitude over this sweep, so a linear axis collapses everything below 64k
# onto the x axis. Memory spans one order and reads better linear.
_LOGY_METRICS = {"t_context_ms", "t_ttft_ms", "t_total_ms", "t_compress_ms", "t_decode_ms"}


def _load(path):
    meta, rows = {}, []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if "meta" in rec:
                meta = rec["meta"]
            else:
                rows.append(rec)
    return meta, rows


def _series(rows, metric):
    """adapter -> (seq_lens, values), sorted by seq_len, skipping NaN/OOM."""
    by_adapter = defaultdict(list)
    derive = _DERIVED.get(metric)
    for r in rows:
        v = derive(r) if derive else r.get(metric)
        # Drop rows that failed the t_gen_full >= t_ttft invariant: their stages
        # were not measured under comparable conditions. The invariant is
        # re-derived here rather than trusted from the `suspect` field so that
        # result files written before that flag existed are covered too.
        suspect = r.get("suspect") or (r.get("t_decode_ms") or 0) < 0
        if r.get("oom") or suspect or v is None or v != v:  # v != v filters NaN
            continue
        by_adapter[r["adapter"]].append((r["seq_len"], v))
    return {a: tuple(zip(*sorted(pts))) for a, pts in by_adapter.items() if pts}


def _oom_starts(rows):
    """adapter -> the shortest sequence length that ran out of memory.

    A curve that just stops is ambiguous: it could mean the method OOMed, or that
    the cell was never scheduled, or that the run was killed. Only OOM is a
    property of the *method* on that device, so only OOM gets drawn (see
    `_draw`); the others are not drawn.
    """
    first = {}
    for r in rows:
        if r.get("oom"):
            L = r["seq_len"]
            if L < first.get(r["adapter"], float("inf")):
                first[r["adapter"]] = L
    return first


def _styles(rows):
    """adapter -> plot kwargs, stable across every panel of a figure.

    Looked up by adapter name, so a method that OOMs on one metric (or is absent
    from a whole sweep) never shifts another method's colour. Only adapters with
    no registered identity fall back to a positional slot.
    """
    present = {r["adapter"] for r in rows}
    ordered = [a for a in _ADAPTER_ORDER if a in present]
    ordered += sorted(present - set(ordered))
    out, spare = {}, 0
    for a in ordered:
        if a in _STYLE_BY_ADAPTER:
            out[a] = dict(_STYLE_BY_ADAPTER[a])
        else:
            out[a] = dict(_FALLBACK[spare % len(_FALLBACK)])
            spare += 1
    return out


def _fmt_tokens(x, _pos=None):
    """262144 -> '256k'. Raw token counts make an axis unreadable at this range."""
    if x >= 1024 ** 2:
        return f"{x / 1024 ** 2:g}M"
    if x >= 1024:
        return f"{x / 1024:g}k"
    return f"{x:g}"


def _fmt_ms(v, _pos=None):
    """Latency spans 10 ms to 100 s in one sweep; label each decade in the unit a
    reader actually thinks in rather than printing '100000' milliseconds."""
    if v >= 1000:
        return f"{v / 1000:g} s"
    return f"{v:g} ms"


def _draw(ax, series, ylabel, logy, styles=None, legend=True, is_time=True,
          oom_starts=None):
    """Draw one panel: every adapter as a curve over sequence length.

    `oom_starts` (adapter -> first OOM length) terminates a curve with a dotted
    stub and a hollow cross. **The cross carries no y measurement** -- there is no
    value to plot, since the cell never completed -- so it is drawn at the last
    measured height purely to keep it on its own curve. The dotted stub and the
    open marker are what say "this method stops here"; the legend names the symbol.
    """
    styles = styles or {}
    # Follow the legend order, not dict insertion order, so the drawing order
    # (and therefore which curve sits on top where they cross) is deterministic.
    order = [a for a in _ADAPTER_ORDER if a in series]
    order += [a for a in series if a not in order]

    for i, adapter in enumerate(order):
        xs, ys = series[adapter]
        kw = dict(styles.get(adapter) or _FALLBACK[i % len(_FALLBACK)])
        ax.plot(
            xs, ys,
            markersize=3.6,
            # A surface-coloured ring around each marker keeps points readable
            # where curves cross, which they do repeatedly on a log-log sweep.
            markeredgecolor="white", markeredgewidth=0.5,
            label=_LABELS.get(adapter, adapter), **kw,
        )

        # Terminate the curve at the length where this method ran out of memory.
        oom_at = (oom_starts or {}).get(adapter)
        if oom_at is not None and xs and oom_at > xs[-1]:
            ax.plot([xs[-1], oom_at], [ys[-1], ys[-1]], linestyle=":", linewidth=1.0,
                    color=kw["color"], zorder=kw.get("zorder", 2))
            ax.plot([oom_at], [ys[-1]], marker="x", markersize=6, markeredgewidth=1.4,
                    color=kw["color"], linestyle="none", zorder=kw.get("zorder", 2) + 1)

    ax.set_xscale("log", base=2)
    if logy:
        ax.set_yscale("log")
        if is_time:
            ax.yaxis.set_major_formatter(FuncFormatter(_fmt_ms))

    # Tick every measured length, but only label those far enough apart to be
    # legible: adjacent sweep points can sit within a few percent of each other
    # (245760 vs 262144), and on a log axis their labels would overlap.
    # OOM lengths join the tick set: if every method OOMed at some length, no
    # measured series would carry it and the marker would land outside the axes.
    all_x = sorted({x for xs, _ in series.values() for x in xs}
                   | {L for a, L in (oom_starts or {}).items() if a in series})
    labels, last = [], None
    for x in all_x:
        if last is None or x / last >= 1.5:
            labels.append(_fmt_tokens(x))
            last = x
        else:
            labels.append("")
    ax.set_xticks(all_x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Sequence length (tokens)")
    ax.set_ylabel(ylabel)
    # `top`/`right` off explicitly: the 'science' style puts ticks on all four
    # sides, and with those two spines hidden they render as marks floating off
    # the edge of the panel.
    ax.tick_params(which="both", direction="out", length=2.5, width=0.5,
                   top=False, right=False)

    # Grid last: it has to be configured after the scales and ticks, which reset
    # locators. Horizontal rules only -- the x positions are the sweep's sample
    # points, already marked by the ticks and by the data markers themselves, and
    # a vertical rule at each of eleven irregularly spaced lengths (two of which
    # nearly coincide) is noise. Values are read off the y axis, so that is where
    # the grid goes. Minor rules only on a linear axis: on a log axis the majors
    # are already decades and the minors would be eight more rules per decade.
    ax.set_axisbelow(True)
    ax.grid(True, which="major", axis="y", **_GRID_MAJOR)
    if not logy:
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(True, which="minor", axis="y", **_GRID_MINOR)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.6)
        ax.spines[side].set_color("#9a9993")
    if legend:
        handles, _ = ax.get_legend_handles_labels()
        handles += _oom_legend_handle(oom_starts, series)
        ax.legend(handles=handles, frameon=False, handlelength=1.8, borderaxespad=0.4)


def _oom_legend_handle(oom_starts, series):
    """One neutral-grey entry naming the OOM symbol, only if a curve carries it.

    Grey rather than a series colour: the symbol means the same thing for every
    method, so tying it to one method's hue would read as belonging to that method.
    """
    if not any(a in series for a in (oom_starts or {})):
        return []
    return [plt.Line2D([], [], color="#52514e", marker="x", markersize=6,
                       markeredgewidth=1.4, linestyle=":", linewidth=1.0,
                       label="out of memory")]


# Per-metric axis label, value scaling and panel title. Anything not listed is
# treated as a latency field in milliseconds (scale 1.0). Latency axis labels
# carry no unit: the tick labels do (see `_fmt_ms`), which is what lets one axis
# span 10 ms to 100 s without an unreadable label.
_METRIC_INFO = {
    "peak_mem_mb":   ("Peak GPU memory (GB)", 1 / 1024, "Peak GPU memory"),
    "t_context_ms":  ("Context processing time", 1.0, "Context processing (compress + TTFT)"),
    "t_ttft_ms":     ("Time to first token", 1.0, "Time to first token"),
    "t_total_ms":    ("End-to-end latency", 1.0, "End-to-end total"),
    "t_compress_ms": ("Compression time", 1.0, "Compression only"),
    "t_decode_ms":   ("Decode time", 1.0, "Decode"),
}


def _metric_info(metric):
    return _METRIC_INFO.get(metric, (f"Latency — {metric}", 1.0, metric))


def _scaled(rows, metric):
    """Series for one metric, already converted to its display units."""
    series = _series(rows, metric)
    _, scale, _title = _metric_info(metric)
    if scale != 1.0:
        series = {a: (xs, tuple(v * scale for v in ys)) for a, (xs, ys) in series.items()}
    return series


def _plot_panels(rows, subtitle, out, logy, title):
    """One figure, every metric: the summary view of a whole sweep."""
    styles = _styles(rows)
    ooms = _oom_starts(rows)
    panels = [(m, _scaled(rows, m)) for m in _PANEL_METRICS]
    panels = [(m, s) for m, s in panels if s]
    if not panels:
        raise SystemExit("No plottable data in any panel metric.")

    ncols = 2
    nrows = (len(panels) + ncols - 1) // ncols
    with plt.style.context(_STYLE):
        # Double-column width: this figure is a summary, not a single claim.
        fig, axes = plt.subplots(nrows, ncols, figsize=(7.2, 3.0 * nrows), squeeze=False)
        flat = [ax for row in axes for ax in row]
        for i, (ax, (metric, series)) in enumerate(zip(flat, panels)):
            ylabel, _scale, panel_title = _metric_info(metric)
            # Legend goes on the figure, once -- repeating it in four panels
            # wastes the plot area the curves need. Log y per metric (see
            # _LOGY_METRICS) unless --logy forces it everywhere.
            _draw(ax, series, ylabel, logy or metric in _LOGY_METRICS,
                  styles=styles, legend=False, is_time=metric != "peak_mem_mb",
                  oom_starts=ooms)
            # Panel letters so the prose can point at one panel by name.
            ax.set_title(f"({chr(ord('a') + i)}) {panel_title}",
                         fontsize="medium", loc="left", pad=6)
        for ax in flat[len(panels):]:
            ax.axis("off")

        handles = [
            plt.Line2D([], [], markersize=3.6, markeredgecolor="white",
                       markeredgewidth=0.5, label=_LABELS.get(a, a), **styles[a])
            for a in styles
            if any(a in s for _, s in panels)
        ]
        handles += _oom_legend_handle(ooms, {a: None for _, s in panels for a in s})
        # One row if it fits, two otherwise -- six full method names on one line
        # would shrink the type below the figure's own font size.
        fig.legend(handles=handles, loc="lower center",
                   ncol=min(len(handles), 3 if len(handles) > 4 else len(handles)),
                   bbox_to_anchor=(0.5, -0.06), frameon=False,
                   handlelength=1.8, columnspacing=1.6)
        fig.suptitle(f"{title}{subtitle}")
        fig.tight_layout()
        fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Wrote {out}  ({len(panels)} panels, {len(handles)} methods)")


def main():
    parser = argparse.ArgumentParser(description="Plot benchmark metrics vs sequence length")
    parser.add_argument("--results", type=str, default="benchmarks/latency/outputs/results.jsonl")
    parser.add_argument("--out", type=str, default="benchmarks/latency/outputs/latency_curve.png")
    parser.add_argument("--metric", type=str, default="t_total_ms",
                        help="Field to plot: t_total_ms, t_compress_ms, t_ttft_ms, t_decode_ms, peak_mem_mb.")
    parser.add_argument("--panels", action="store_true",
                        help=f"One combined figure with every metric ({', '.join(_PANEL_METRICS)}) "
                             "instead of a single-metric plot. Colours and markers are shared "
                             "across panels.")
    parser.add_argument("--title", type=str, default="End-to-end latency",
                        help="Figure title, --panels only.")
    parser.add_argument("--logy", action="store_true", help="Log-scale the y axis.")
    args = parser.parse_args()

    meta, rows = _load(args.results)
    bits = [meta.get("dtype")]
    if meta.get("beacon_checkpoint"):
        bits.append(f"beacon ratio {meta.get('beacon_ratio')}")
    if meta.get("sac_ratio") and any(r["adapter"] == "sac" for r in rows):
        bits.append(f"sac ratio {meta['sac_ratio']}")
    if meta.get("icae_ratio") and any(r["adapter"] == "icae" for r in rows):
        bits.append(f"icae ratio {meta['icae_ratio']}")
    subtitle = f"  ({', '.join(b for b in bits if b)})" if meta else ""

    if args.panels:
        _plot_panels(rows, subtitle, args.out, args.logy, args.title)
        return

    series = _scaled(rows, args.metric)
    if not series:
        raise SystemExit(f"No plottable data for metric '{args.metric}' in {args.results}")
    ylabel, _scale, panel_title = _metric_info(args.metric)

    with plt.style.context(_STYLE):
        fig, ax = plt.subplots(figsize=(3.5, 2.8))
        _draw(ax, series, ylabel, args.logy or args.metric in _LOGY_METRICS,
              styles=_styles(rows), is_time=args.metric != "peak_mem_mb",
              oom_starts=_oom_starts(rows))
        ax.set_title(f"{panel_title}{subtitle}", loc="left", pad=6)
        fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

"""Paper figures: the four sweep metrics, rendered once per decoder scale.

Eight narrow figures into `outputs/paper/` -- four metrics
(`t_context_ms`, `t_ttft_ms`, `t_total_ms`, `peak_mem_mb`) at Qwen3-1.7B and
Qwen3-4B. These are the four panels of `plot.py --panels` as standalone
figures: same metrics, same order, but each one sized narrow and stripped of the
on-canvas title and run metadata that a panel grid carries.

Why one script for both scales rather than a `_1-7b` / `_4b` pair like
`paper_figure_single.py`: a metric's y range has to be **shared** between the two
decoders or only orderings within a figure can be compared, never the
figures against each other. Deriving that range means seeing both files, so both
are loaded in one process and the limits come from the union. There is no
hand-maintained `_YLIM` constant here for the same reason -- a constant is a
number that silently stops matching its data.

Both Activation Beacon windows are drawn (w=128 from the main sweep, w=2048
merged in from a second file). `icae` is dropped -- `--drop` with no values
brings it back. Styling, legend text and the OOM terminator are imported from the
sibling figures so none of them can drift apart.

Usage:
    uv run python -m benchmarks.latency.paper_panels --pdf
    uv run python -m benchmarks.latency.paper_panels --only 1-7b   # partial data
"""

import argparse
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import scienceplots  # noqa: F401  registers the 'science' styles (required import)

from benchmarks.latency.paper_figure import _HERO, _hero_weight
from benchmarks.latency.paper_figure_single import _DROP, _LEGEND, _beacon_long_rows
from benchmarks.latency.plot import (
    _ADAPTER_ORDER, _FALLBACK, _GRID_MAJOR, _LOGY_METRICS, _METRIC_INFO, _STYLE,
    _STYLE_BY_ADAPTER, _fmt_ms, _fmt_tokens, _load, _metric_info, _oom_starts,
    _scaled,
)

# Reading order, and the order the files are written in: the cost of turning
# context into something the decoder can read, what the user waits for, the
# end-to-end total, and what it all fits in. Kept as its own list rather than
# imported from `plot._PANEL_METRICS` because the second element is the output
# filename stem, which is this module's concern alone.
_METRICS = [
    ("t_context_ms", "context"),
    ("t_ttft_ms", "ttft"),
    ("t_total_ms", "total"),
    ("peak_mem_mb", "memory"),
]

# `--strip`: the three main metrics, in one row.
#
# `t_ttft_ms` is the one left out, and deliberately. SAC and ICAE expose a
# separable `compress()` and report it in `t_compress`; Boa, Beacon and full_ctx
# fuse compression into the first forward pass and report `t_compress = 0`. So a
# TTFT panel puts SAC's 90 ms next to Boa's 4.7 s at 240k while SAC's real 18 s of
# compression sits outside the frame -- true per field, misleading as a
# comparison. `t_context_ms` is the same quantity for every method (raw tokens ->
# first generated token), which is what makes it the one to show.
_STRIP_METRICS = ["t_context_ms", "t_total_ms", "peak_mem_mb"]

# Height of the strip reserved under the axes for the legend and the shared x
# label, as a fraction of figure height. `tight_layout` packs the panels into
# what is left. Reserving it is what keeps the legend attached to the figure:
# anchoring below the canvas and letting `bbox_inches="tight"` grow into the gap
# leaves whitespace proportional to how far the anchor overshot. Sized to the
# legend text, so the two move together.
_BAND = 0.20

# The two decoder scales, keyed by the slug that ends every output filename.
# Same sweep, same host, same accelerator model at both
# scales -- which is the precondition for sharing a y range between them.
#
# The 4B `beacon_long` file is the window-128 checkpoint *run* at window 2048; no
# 4B window-2048 checkpoint has been trained. Sound for latency, nothing else.
# `_beacon_long_rows` returns that override flag and `_load_model` prints it, but
# it must be reported wherever the figure is used.
_MODELS = {
    "1-7b": {
        "results": "benchmarks/latency/outputs/qwen3_1-7b/results_sota_1-7b_beacon_w128.jsonl",
        "beacon_long": "benchmarks/latency/outputs/qwen3_1-7b/results_sota_1-7b_beacon_w2048.jsonl",
    },
    "4b": {
        "results": "benchmarks/latency/outputs/qwen3_4b/results_sota_4b_beacon_w128.jsonl",
        "beacon_long": "benchmarks/latency/outputs/qwen3_4b/results_beacon_4b_w2048.jsonl",
    },
}

_OUT_DIR = "benchmarks/latency/outputs/paper"


def _load_model(slug, paths, drop):
    """Every row for one decoder scale: the main sweep plus the second Beacon window."""
    _, rows = _load(paths["results"])
    if not rows:
        raise SystemExit(f"No rows in {paths['results']}.")
    print(f"{slug}: {len(rows)} rows from {paths['results']}")

    if paths.get("beacon_long"):
        extra, window, overridden = _beacon_long_rows(paths["beacon_long"])
        rows = rows + extra
        print(f"  + beacon_long from {paths['beacon_long']} (window {window})")
        if overridden:
            print(f"    NOTE: window {window} was set at load time on a checkpoint trained at a "
                  f"different window.\n"
                  f"    Latency is valid (no beacon tensor's shape depends on the window); "
                  f"generation quality is not.\n"
                  f"    Note this wherever the figure is used.")

    if drop:
        kept = [r for r in rows if r["adapter"] not in set(drop)]
        if len(kept) == len(rows):
            print(f"  ! --drop {' '.join(drop)} matched no rows")
        rows = kept
        print(f"  - dropped: {', '.join(drop)}")
    return rows


def _shared_ylim(metric, per_model_series):
    """One y range for this metric across every decoder scale being drawn.

    Log metrics snap out to whole decades so the lowest and highest labelled
    ticks bracket the data instead of cutting through it. Memory is linear and
    anchored at zero, because a bar-like quantity read off a floating baseline
    invites comparing the wrong distances.
    """
    values = [v for series in per_model_series for _, ys in series.values() for v in ys]
    if not values:
        return None
    lo, hi = min(values), max(values)
    if metric in _LOGY_METRICS:
        return 10 ** math.floor(math.log10(lo)), 10 ** math.ceil(math.log10(hi))
    return 0.0, hi * 1.08


def _tick_labels(all_x, anchor="left", min_ratio=1.5):
    """Label the measured lengths far enough apart to stay legible.

    Every length gets a tick; this only decides which get text. A label is kept
    when it is at least `min_ratio` from the last one kept, and blanked otherwise.

    Both knobs exist because the strip needs different answers from the
    standalone figures, at a third of their width:

    * `anchor` -- walking left to right keeps 240k and drops 256k, right at single
      figure width. Narrower, 128k and 240k also collide, and dropping 240k would
      blank the tail, where the curves separate and where every quoted ratio is
      measured. Walking right to left anchors on the largest length and thins away
      from it, so the tail stays labelled at any width.
    * `min_ratio` -- one doubling apart is legible for `64k 128k` and not for
      `128k 256k`, since the second pair is two four-character labels. Rather than
      special-case the widest pair, the strip labels every second doubling
      (`min_ratio=2.5` on a x2 sweep), which is the density a third-width panel
      has room for anyway.
    """
    seq = all_x if anchor == "left" else list(reversed(all_x))
    keep, last = set(), None
    for x in seq:
        if last is None or (x / last if anchor == "left" else last / x) >= min_ratio:
            keep.add(x)
            last = x
    return [_fmt_tokens(x) if x in keep else "" for x in all_x]


def _draw(ax, metric, series, ooms, ylim, xlabel=True, tick_anchor="left",
          tick_ratio=1.5):
    """One panel: every method as a curve over sequence length.

    `xlabel` is off for a panel in the strip, where the axis title is set once on
    the figure -- repeating it under all three panels is the clearest tell of a
    figure assembled panel-by-panel rather than designed as one. The y label stays
    per-panel either way: the three panels are three different quantities in three
    different units, so there is nothing to hoist.
    """
    order = [a for a in _ADAPTER_ORDER if a in series]
    order += [a for a in series if a not in order]
    is_time = metric != "peak_mem_mb"

    for i, adapter in enumerate(order):
        xs, ys = series[adapter]
        kw = _hero_weight(dict(_STYLE_BY_ADAPTER.get(adapter)
                               or _FALLBACK[i % len(_FALLBACK)]), adapter)
        ax.plot(xs, ys, markersize=3.7 if adapter == _HERO else 3.0,
                markeredgecolor="white", markeredgewidth=0.5,
                label=_LEGEND.get(adapter, adapter), **kw)
        # A curve that stops is ambiguous; only OOM is a property of the method,
        # so only OOM is drawn. The cross sits at the last measured height purely
        # to keep it on its own curve -- it carries no y measurement.
        oom_at = ooms.get(adapter)
        if oom_at is not None and xs and oom_at > xs[-1]:
            ax.plot([xs[-1], oom_at], [ys[-1], ys[-1]], linestyle=":", linewidth=1.0,
                    color=kw["color"], zorder=kw.get("zorder", 2))
            ax.plot([oom_at], [ys[-1]], marker="x", markersize=5.5, markeredgewidth=1.3,
                    color=kw["color"], linestyle="none", zorder=kw.get("zorder", 2) + 1)

    ax.set_xscale("log", base=2)
    if is_time:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(_fmt_ms))
    if ylim is not None:
        ax.set_ylim(*ylim)

    all_x = sorted({x for xs, _ in series.values() for x in xs}
                   | {L for a, L in ooms.items() if a in series})
    ax.set_xticks(all_x)
    ax.set_xticklabels(_tick_labels(all_x, tick_anchor, tick_ratio))
    if xlabel:
        ax.set_xlabel("Sequence length (tokens)")
    ax.set_ylabel(_metric_info(metric)[0])

    ax.tick_params(which="both", direction="out", length=2.5, width=0.5,
                   top=False, right=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.6)
        ax.spines[side].set_color("#9a9993")
    ax.set_axisbelow(True)
    ax.grid(True, which="major", axis="y", **_GRID_MAJOR)


def _legend(ax, series, ooms):
    """Inside the axes, top left, two columns.

    These curves rise from bottom-left to top-right on every one of the four
    metrics, so the top-left quadrant is dead space; the legend there costs
    nothing a figure-level legend would not cost more of. Framed and opaque
    because sitting over the plotting area it needs a boundary to read as an
    overlay rather than as stray marks among the data.
    """
    handles = [h for h in ax.get_lines() if h.get_label() in _LEGEND.values()]
    if any(a in series for a in ooms):
        handles.append(plt.Line2D([], [], color="#52514e", marker="x", markersize=5.5,
                                  markeredgewidth=1.3, linestyle=":", linewidth=1.0,
                                  label="out of memory"))
    leg = ax.legend(handles=handles, loc="upper left", ncol=2, frameon=True,
                    handlelength=1.6, labelspacing=0.32, columnspacing=1.1,
                    borderaxespad=0.4, borderpad=0.5, handletextpad=0.5,
                    fontsize="small")
    frame = leg.get_frame()
    frame.set_linewidth(0.5)
    frame.set_edgecolor("#c9c8c3")
    frame.set_facecolor("white")
    frame.set_alpha(1.0)
    leg.set_zorder(10)


def _handles(seen):
    """Figure-level legend handles, in `_ADAPTER_ORDER`, for the methods drawn."""
    out = [
        plt.Line2D([], [], markersize=3.7 if a == _HERO else 3.0,
                   markeredgecolor="white", markeredgewidth=0.5,
                   label=_LEGEND.get(a, a),
                   **_hero_weight(dict(_STYLE_BY_ADAPTER[a]), a))
        for a in _ADAPTER_ORDER if a in seen
    ]
    if "__oom__" in seen:
        out.append(plt.Line2D([], [], color="#52514e", marker="x", markersize=5.5,
                              markeredgewidth=1.3, linestyle=":", linewidth=1.0,
                              label="out of memory"))
    return out


def _render_strip(slug, rows, ooms, ylims, out, pdf):
    """`_STRIP_METRICS` as one 1x3 row.

    Wide figure. Unlike the two-panel figures in
    `paper_figure_pair.py` there is no `sharey` to apply -- three different
    quantities in three different units share no ruler -- so each panel keeps its
    own axis, and the per-metric range still comes from `ylims`, shared with the
    other decoder scale.
    """
    with plt.style.context(_STYLE):
        fig, axes = plt.subplots(1, len(_STRIP_METRICS), figsize=(7.2, 2.7))
        seen = {}
        for i, (ax, metric) in enumerate(zip(axes, _STRIP_METRICS)):
            series = _scaled(rows, metric)
            if not series:
                ax.axis("off")
                continue
            _draw(ax, metric, series, ooms, ylims.get(metric), xlabel=False,
                  tick_anchor="right", tick_ratio=2.5)
            # Panel letters so the prose can point at one panel by name.
            ax.set_title(f"({chr(ord('a') + i)}) {_metric_info(metric)[2]}",
                         fontsize="medium", loc="left", pad=6)
            seen.update({a: None for a in series})
            if any(a in series for a in ooms):
                seen["__oom__"] = None

        # tight_layout BEFORE the legend and the figure-level x label: it re-lays
        # the axes and ignores both, so placing them first strands them relative
        # to the final frame.
        fig.tight_layout(rect=(0, _BAND, 1, 1), pad=0.3, w_pad=1.4)
        fig.supxlabel("Sequence length (tokens)", fontsize="medium", y=_BAND - 0.035)
        fig.legend(handles=_handles(seen), loc="upper center", ncol=len(_handles(seen)),
                   frameon=False, bbox_to_anchor=(0.5, _BAND - 0.075),
                   handlelength=1.8, columnspacing=1.2, fontsize="medium")

        fig.savefig(out, dpi=300, bbox_inches="tight")
        if pdf:
            fig.savefig(os.path.splitext(out)[0] + ".pdf", bbox_inches="tight")
        plt.close(fig)
    print(f"  Wrote {out}" + ("  (+ .pdf)" if pdf else ""))


def _ratios(metric, series):
    """Print each baseline against ours, and the two Beacon windows against each other.

    Printed rather than drawn: these figures are bare by design, so the numbers are
    reported separately. Reported at the longest length both methods actually
    reached, so both endpoints are measured points rather than an interpolation.
    """
    def at_common(fast, slow):
        if fast not in series or slow not in series:
            return None
        x_fast, y_fast = series[fast]
        x_slow, y_slow = series[slow]
        common = sorted(set(x_fast) & set(x_slow))
        if not common:
            return None
        x = common[-1]
        return x, dict(zip(x_slow, y_slow))[x] / dict(zip(x_fast, y_fast))[x]

    for adapter in (a for a in _ADAPTER_ORDER if a in series and a != _HERO):
        got = at_common(_HERO, adapter)
        if got is not None:
            x, factor = got
            print(f"    {metric}: {_LEGEND.get(adapter, adapter)} / {_LEGEND[_HERO]}"
                  f" at {_fmt_tokens(x)} = {factor:.1f}x")
    got = at_common("beacon_long", "beacon")
    if got is not None:
        x, factor = got
        print(f"    {metric}: beacon w=128 / w=2048 at {_fmt_tokens(x)} = {factor:.2f}x")


def render(args):
    os.makedirs(args.out_dir, exist_ok=True)

    slugs = [args.only] if args.only else list(_MODELS)
    missing = [p for s in slugs for p in _MODELS[s].values() if not os.path.exists(p)]
    if missing:
        raise SystemExit("Missing results files:\n  " + "\n  ".join(missing) +
                         "\n\nRun the sweeps (`benchmarks/latency/bench.py`) or pass --only "
                         "to render the scale you have. With one scale the y ranges come "
                         "from it alone, so the figures are NOT comparable against the other.")
    if args.only:
        print(f"! --only {args.only}: y ranges derived from this scale alone; the figures "
              f"are not comparable against the other decoder.\n")

    rows = {s: _load_model(s, _MODELS[s], args.drop) for s in slugs}
    ooms = {s: _oom_starts(r) for s, r in rows.items()}

    # Every y range up front, from the union of the scales being drawn, so the
    # standalone figures and the strip put the same metric on the same axis.
    # Display units (GB for memory) are applied by `_scaled`, so limits and drawn
    # values are in the same units by construction.
    series = {m: {s: _scaled(r, m) for s, r in rows.items()} for m, _ in _METRICS}
    series = {m: {s: v for s, v in per.items() if v} for m, per in series.items()}
    ylims = {m: _shared_ylim(m, per.values()) for m, per in series.items() if per}

    for metric, stem in _METRICS:
        if not series[metric]:
            print(f"! no plottable data for {metric}; skipped")
            continue
        for slug, s in series[metric].items():
            out = os.path.join(args.out_dir, f"{stem}_{slug}.png")
            with plt.style.context(_STYLE):
                fig, ax = plt.subplots(figsize=(3.4, 2.9))
                _draw(ax, metric, s, ooms[slug], ylims[metric])
                _legend(ax, s, ooms[slug])
                fig.tight_layout()
                fig.savefig(out, dpi=300, bbox_inches="tight")
                if args.pdf:
                    fig.savefig(os.path.splitext(out)[0] + ".pdf", bbox_inches="tight")
                plt.close(fig)
            print(f"  Wrote {out}" + ("  (+ .pdf)" if args.pdf else ""))
            _ratios(metric, s)

    if not args.no_strip:
        for slug in slugs:
            _render_strip(slug, rows[slug], ooms[slug], ylims,
                          os.path.join(args.out_dir, f"strip_{slug}.png"), args.pdf)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", choices=sorted(_MODELS),
                        help="Render one decoder scale instead of both. Its y ranges are then "
                             "derived from that scale alone, breaking cross-figure reading.")
    parser.add_argument("--drop", nargs="*", default=list(_DROP), metavar="ADAPTER",
                        help=f"Adapters to leave out (default: {' '.join(_DROP)}). "
                             "Pass with no values to plot every method in the files.")
    parser.add_argument("--no-strip", action="store_true",
                        help=f"Skip the 1x3 combined row ({', '.join(_STRIP_METRICS)}), "
                             "leaving only the standalone per-metric figures.")
    parser.add_argument("--out-dir", default=_OUT_DIR)
    parser.add_argument("--pdf", action="store_true",
                        help="Also write a vector PDF beside each PNG.")
    render(parser.parse_args())


if __name__ == "__main__":
    main()

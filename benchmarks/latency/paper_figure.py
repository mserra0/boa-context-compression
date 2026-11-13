"""Paper figure: end-to-end latency at two decoder scales, one stacked figure.

Deliberately bare: no suptitle, no metric title, no run metadata in the frame; model,
hardware and sweep settings are not drawn.

Layout is 2 rows x 1 column in a narrow figure. The two runs share the
sequence-length axis exactly, so stacking aligns it: a vertical line at any
length reads both scales at once. Side by side would repeat that axis
and need a wide figure for no gain.

The y axes are deliberately NOT shared: this figure is for runs that may come from
different accelerators, so absolute heights are not comparable across panels; what
it shows is the *ordering and the gaps within* each panel, which is what
independent axes direct the eye to. For runs on one accelerator model use
`paper_figure_pair.py` / `paper_figure_single.py`, which share a y range.

Styling (colour, marker, line style per method, and the OOM terminator) is
imported from `plot.py` so this figure and the diagnostic ones cannot drift apart.

Usage:
    uv run python -m benchmarks.latency.paper_figure
"""

import argparse
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import scienceplots  # noqa: F401  registers the 'science' styles (required import)

from benchmarks.latency.plot import (
    _ADAPTER_ORDER, _FALLBACK, _GRID_MAJOR, _LABELS, _STYLE, _STYLE_BY_ADAPTER,
    _fmt_ms, _fmt_tokens, _load, _oom_starts, _series,
)

# (results file, panel label). Order is the order of the rows.
_RUNS = [
    ("benchmarks/latency/outputs/qwen3_1-7b/results_sota_1-7b_beacon_w128.jsonl", "Qwen3-1.7B"),
    ("benchmarks/latency/outputs/qwen3_4b/results_sota_4b_beacon_w128.jsonl", "Qwen3-4B"),
]

_METRIC = "t_total_ms"

# Methods whose gap gets a speedup arrow. On a log y axis the vertical distance
# between two curves IS their ratio, so a plain double-headed arrow annotated with
# the factor is an honest reading of the geometry rather than a decoration.
_ARROW_PAIR = ("ssm_kv", "beacon")

# The featured method (`ssm_kv`). It gets a heavier stroke and sits on top of every
# crossing; the baselines use a lighter weight, so the featured curve stands out
# among five otherwise equally-weighted curves.
_HERO = "ssm_kv"
_HERO_LW = 1.9
_BASELINE_LW = 1.1


def _hero_weight(kw, adapter):
    """Apply the hero/baseline stroke split to one method's plot kwargs."""
    kw["linewidth"] = _HERO_LW if adapter == _HERO else _BASELINE_LW
    if adapter == _HERO:
        kw["zorder"] = 5
    return kw


def _speedup_arrow(ax, series, fast, slow):
    """Double-headed arrow spanning `slow` down to `fast`, labelled with the factor.

    Drawn at the longest length both methods actually reached, so the two endpoints
    are measured points rather than an interpolation. Returns the factor, or None
    if the pair never overlaps (e.g. one of them OOMed out of the shared range).
    """
    if fast not in series or slow not in series:
        return None
    x_fast, y_fast = series[fast]
    x_slow, y_slow = series[slow]
    common = sorted(set(x_fast) & set(x_slow))
    if not common:
        return None
    x = common[-1]
    lo = dict(zip(x_fast, y_fast))[x]
    hi = dict(zip(x_slow, y_slow))[x]
    if hi <= lo:
        return None

    # Caliper (`|-|`), not a double-headed arrow. The annotation marks an *extent*
    # -- the distance between two curves -- and arrowheads read as direction or
    # flow, which is not what a ratio is. Serif caps are the dimension-line idiom
    # from technical drawing and sit quieter on the page at the same weight.
    ax.annotate("", xy=(x, hi), xytext=(x, lo),
                arrowprops=dict(arrowstyle="|-|,widthA=0.45,widthB=0.45",
                                color="#3d3c3a", linewidth=0.8, mutation_scale=8,
                                shrinkA=3, shrinkB=3), zorder=6)
    # Midpoint of a log axis is the geometric mean. The label sits OUTSIDE the
    # data, to the right of the arrow: `clip_on=False` lets it render past the
    # axes and `bbox_inches="tight"` at save time grows the canvas to include it.
    # Outside the curves it needs no masking patch, so it reads cleaner than the
    # boxed-glyph-on-top-of-the-lines alternative.
    ax.text(x * 1.10, math.sqrt(lo * hi), f"{hi / lo:.0f}×",
            ha="left", va="center", fontsize="medium", fontweight="bold",
            color="#3d3c3a", zorder=7, clip_on=False)
    return hi / lo


def _panel(ax, rows, label, show_xlabel, arrow_pair=_ARROW_PAIR):
    series = _series(rows, _METRIC)
    ooms = _oom_starts(rows)
    order = [a for a in _ADAPTER_ORDER if a in series]
    order += [a for a in series if a not in order]

    for i, adapter in enumerate(order):
        xs, ys = series[adapter]
        kw = _hero_weight(dict(_STYLE_BY_ADAPTER.get(adapter)
                               or _FALLBACK[i % len(_FALLBACK)]), adapter)
        # The hero's markers scale with its stroke, or the heavier line swallows
        # them and the series loses its shape encoding.
        ax.plot(xs, ys, markersize=3.7 if adapter == _HERO else 3.0,
                markeredgecolor="white", markeredgewidth=0.5,
                label=_LABELS.get(adapter, adapter), **kw)
        oom_at = ooms.get(adapter)
        if oom_at is not None and xs and oom_at > xs[-1]:
            ax.plot([xs[-1], oom_at], [ys[-1], ys[-1]], linestyle=":", linewidth=1.0,
                    color=kw["color"], zorder=kw.get("zorder", 2))
            ax.plot([oom_at], [ys[-1]], marker="x", markersize=5.5, markeredgewidth=1.3,
                    color=kw["color"], linestyle="none", zorder=kw.get("zorder", 2) + 1)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_ms))

    # Drop the floor to the decade below the data. Without it a panel whose
    # minimum sits just above a decade boundary (4B starts at 1.44 s) renders its
    # lowest labelled tick above most of the curves, leaving the bottom of the
    # panel with no gridline to read against. Snapping per panel keeps the axes
    # independent -- forcing a shared range would reinstate the cross-panel
    # height comparison that different accelerators would make invalid.
    lo = min(min(ys) for _, ys in series.values())
    ax.set_ylim(bottom=10 ** math.floor(math.log10(lo)))

    all_x = sorted({x for xs, _ in series.values() for x in xs}
                   | {L for a, L in ooms.items() if a in series})
    labels, last = [], None
    for x in all_x:
        if last is None or x / last >= 1.5:
            labels.append(_fmt_tokens(x))
            last = x
        else:
            labels.append("")
    ax.set_xticks(all_x)
    ax.set_xticklabels(labels if show_xlabel else [""] * len(all_x))
    if show_xlabel:
        ax.set_xlabel("Sequence length (tokens)")
    # No per-axes y label: both panels measure the same quantity, so it is set
    # once on the figure (`supylabel`). Repeating it is the clearest tell of a
    # figure assembled panel-by-panel rather than designed as one.

    # The decoder name sits inside the frame: at this size an axes title would
    # cost a line of vertical space the curves need.
    # Sans against the style's serif axis text: the panel label names the panel,
    # it does not describe the data, and giving it a different voice keeps the two
    # roles apart. Set explicitly rather than inherited, so this survives a change
    # of SciencePlots style. (`cmss10` is the other sans available but ships no
    # bold face, so `fontweight` is silently ignored there.)
    ax.text(0.03, 0.94, label, transform=ax.transAxes, va="top", ha="left",
            fontsize="large", fontweight="bold", fontfamily="DejaVu Sans")

    ax.tick_params(which="both", direction="out", length=2.5, width=0.5,
                   top=False, right=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.6)
        ax.spines[side].set_color("#9a9993")
    ax.set_axisbelow(True)
    ax.grid(True, which="major", axis="y", **_GRID_MAJOR)

    factor = _speedup_arrow(ax, series, *arrow_pair)
    return series, ooms, factor


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="benchmarks/latency/outputs/paper/end_to_end.png")
    parser.add_argument("--pdf", action="store_true",
                        help="Also write a vector PDF beside the PNG.")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    loaded = [(_load(path)[1], label) for path, label in _RUNS]
    for rows, label in loaded:
        if not rows:
            raise SystemExit(f"No rows for {label}.")

    with plt.style.context(_STYLE):
        # Narrow figure. Two short panels rather than one tall one: the
        # curves span three decades, and extra height buys no resolution on a log
        # axis.
        fig, axes = plt.subplots(2, 1, figsize=(3.4, 4.4))
        seen = {}
        for i, (ax, (rows, label)) in enumerate(zip(axes, loaded)):
            series, ooms, factor = _panel(ax, rows, label,
                                          show_xlabel=(i == len(loaded) - 1))
            seen.update({a: None for a in series})
            if any(a in series for a in ooms):
                seen["__oom__"] = None
            if factor is not None:
                print(f"  {label}: {_LABELS[_ARROW_PAIR[1]]} / {_LABELS[_ARROW_PAIR[0]]}"
                      f" = {factor:.1f}x  (annotated as {factor:.0f}x)")

        handles = [
            plt.Line2D([], [], markersize=3.7 if a == _HERO else 3.0,
                       markeredgecolor="white",
                       markeredgewidth=0.5, label=_LABELS.get(a, a),
                       **_hero_weight(dict(_STYLE_BY_ADAPTER[a]), a))
            for a in _ADAPTER_ORDER if a in seen
        ]
        if "__oom__" in seen:
            handles.append(plt.Line2D([], [], color="#52514e", marker="x", markersize=5.5,
                                      markeredgewidth=1.3, linestyle=":", linewidth=1.0,
                                      label="out of memory"))
        fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
                   bbox_to_anchor=(0.5, -0.085), handlelength=1.8, columnspacing=1.2,
                   fontsize="small")
        # y offset: supylabel centres on the whole canvas, and the legend strip below
        # the panels would otherwise drag the label below the plotting area.
        fig.supylabel("End-to-end latency", fontsize="medium", x=0.005, y=0.58)
        # Tight vertical coupling: the panels share an x axis, so a wide gutter
        # would read as two separate figures rather than one comparison.
        fig.tight_layout(h_pad=0.6)
        fig.savefig(args.out, dpi=300, bbox_inches="tight")
        if args.pdf:
            fig.savefig(os.path.splitext(args.out)[0] + ".pdf", bbox_inches="tight")
    print(f"Wrote {args.out}" + ("  (+ .pdf)" if args.pdf else ""))


if __name__ == "__main__":
    main()

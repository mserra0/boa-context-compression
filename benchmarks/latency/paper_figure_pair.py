"""Paper figure: end-to-end latency at both decoder scales, in one frame.

The two single-panel figures (`paper_figure_1_7b.py`, `paper_figure_4b.py`)
composed, in either orientation:

    --orientation horizontal   1x2, `sharey`, wide figure
    --orientation vertical     2x1, `sharex`, narrow figure

Horizontal shares the *vertical* axis, so the panels sit on one ruler and the eye
travels sideways along it -- the reading for "how much does the bigger decoder
cost". Vertical shares the *sequence-length* axis, so a vertical line at any
length reads both decoders off it -- the reading for "at this length, what happens
at each scale". Both show the same data. Either way the y range is the same fixed
one, so heights are comparable between panels.

Sharing a y range across panels is only legitimate because every sweep behind them
ran on the same accelerator model. The other two-panel
figure, `paper_figure.py`, keeps its axes independent precisely because its runs
did not.

Each panel is titled with its decoder, above the frame; nothing else is drawn on
the canvas. The panels are what differ, so the decoder name is the one label kept.

Panel bodies, styling and the shared y range come from `paper_figure_single.py`,
so this figure cannot drift from the two it composes.

Usage:
    uv run python -m benchmarks.latency.paper_figure_pair --pdf
    uv run python -m benchmarks.latency.paper_figure_pair --orientation vertical --pdf
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401  registers the 'science' styles (required import)

from benchmarks.latency import paper_figure_1_7b as fig_1_7b
from benchmarks.latency import paper_figure_4b as fig_4b
from benchmarks.latency.paper_figure import _HERO, _hero_weight
from benchmarks.latency.paper_figure_single import (
    _ADAPTER_ORDER, _DROP, _LEGEND, _STYLE, _STYLE_BY_ADAPTER, _YLIM,
    _beacon_long_rows, _draw, _fmt_tokens, _load, _ratio,
)

# (five-way sweep, longer-window Beacon sweep, panel title). Left to right, or top
# to bottom. Paths are imported from the two single-panel scripts rather than
# repeated, so repointing one of those figures repoints this one too.
_PANELS = [
    (fig_1_7b._RESULTS, fig_1_7b._BEACON_LONG, "Qwen3-1.7B"),
    (fig_4b._RESULTS, fig_4b._BEACON_LONG, "Qwen3-4B"),
]

# Per-orientation layout. `band` is the strip reserved under the axes for the
# legend (and, when horizontal, the shared x label), as a fraction of figure
# height: `tight_layout` packs the panels into what is left. Reserving the strip
# is what keeps the legend attached -- anchoring it below the canvas and letting
# `bbox_inches="tight"` grow into the gap leaves whitespace proportional to how far
# the anchor overshot.
#
# `legend_cols` follows the width available: five entries fit on one line across a
# wide figure and would run off a narrow one.
#
# `band` is sized to the legend text: enlarging the font without widening the strip
# would push the legend into the panels above it, so the two move together.
_LAYOUTS = {
    "horizontal": {
        "shape": (1, 2), "figsize": (6.8, 2.9), "sharey": True, "sharex": False,
        "band": 0.20, "legend_cols": None, "out": "end_to_end_pair.png",
    },
    "vertical": {
        "shape": (2, 1), "figsize": (3.4, 4.7), "sharey": False, "sharex": True,
        "band": 0.14, "legend_cols": 2, "out": "end_to_end_pair_vertical.png",
    },
}

# Legend text one step up from the axis labels' `small`. The legend is the only key
# to which curve is which, and the figure is rendered at a small size: naming
# the size here rather than inline is what keeps both orientations in step.
_LEGEND_FONTSIZE = "medium"

_OUT_DIR = "benchmarks/latency/outputs/paper"


def _rows(results, beacon_long, drop):
    _, rows = _load(results)
    if not rows:
        raise SystemExit(f"No rows in {results}.")
    if beacon_long:
        extra, window, overridden = _beacon_long_rows(beacon_long)
        rows = rows + extra
        note = "  (window set at load time -- latency only)" if overridden else ""
        print(f"  + beacon_long from {beacon_long} (window {window}){note}")
    return [r for r in rows if r["adapter"] not in set(drop)]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--orientation", choices=sorted(_LAYOUTS), default="horizontal",
                        help="horizontal: side by side, sharing the latency axis. "
                             "vertical: stacked, sharing the sequence-length axis.")
    parser.add_argument("--drop", nargs="*", default=list(_DROP), metavar="ADAPTER",
                        help=f"Adapters to leave out (default: {' '.join(_DROP)}). "
                             "Pass with no values to plot every method in the files.")
    parser.add_argument("--out", help="Defaults to a name derived from the orientation.")
    parser.add_argument("--pdf", action="store_true",
                        help="Also write a vector PDF beside the PNG.")
    args = parser.parse_args()

    layout = _LAYOUTS[args.orientation]
    out = args.out or os.path.join(_OUT_DIR, layout["out"])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    vertical = args.orientation == "vertical"

    loaded = []
    for results, beacon_long, title in _PANELS:
        print(f"{title}:")
        loaded.append((_rows(results, beacon_long, args.drop), title))
    if args.drop:
        print(f"  - dropped: {', '.join(args.drop)}")

    with plt.style.context(_STYLE):
        rows_n, cols_n = layout["shape"]
        fig, axes = plt.subplots(rows_n, cols_n, figsize=layout["figsize"],
                                 sharex=layout["sharex"], sharey=layout["sharey"])
        seen = {}
        for i, (ax, (rows, title)) in enumerate(zip(axes, loaded)):
            last = i == len(loaded) - 1
            # Axis titles are set once per figure, never per panel -- repeating
            # them is the clearest tell of a figure assembled panel-by-panel.
            # Vertical: the shared x axis means only the bottom panel is labelled,
            # and the y label goes on the figure (`supylabel`) since both panels
            # carry ticks. Horizontal: the reverse -- x on the figure, y on the
            # left panel, which is the only one `sharey` leaves ticks on.
            series, ooms = _draw(ax, rows, _YLIM,
                                 xlabel=(vertical and last),
                                 ylabel=(not vertical and i == 0))
            # Sans against the style's serif axis text: the title names the panel,
            # it does not describe the data, and a different voice keeps the two
            # roles apart. Set explicitly so it survives a change of SciencePlots
            # style. (`cmss10`, the other sans available, ships no bold face.)
            ax.set_title(title, fontsize="large", fontweight="bold",
                         fontfamily="DejaVu Sans", pad=4)
            seen.update({a: None for a in series})
            if any(a in series for a in ooms):
                seen["__oom__"] = None
            for adapter in (a for a in _ADAPTER_ORDER if a in series and a != _HERO):
                got = _ratio(series, _HERO, adapter)
                if got is not None:
                    at, factor = got
                    print(f"  {title}: {_LEGEND.get(adapter, adapter)} / {_LEGEND[_HERO]}"
                          f" at {_fmt_tokens(at)} = {factor:.1f}x")

        handles = [
            plt.Line2D([], [], markersize=3.7 if a == _HERO else 3.0,
                       markeredgecolor="white", markeredgewidth=0.5,
                       label=_LEGEND.get(a, a),
                       **_hero_weight(dict(_STYLE_BY_ADAPTER[a]), a))
            for a in _ADAPTER_ORDER if a in seen
        ]
        if "__oom__" in seen:
            handles.append(plt.Line2D([], [], color="#52514e", marker="x", markersize=5.5,
                                      markeredgewidth=1.3, linestyle=":", linewidth=1.0,
                                      label="out of memory"))

        # tight_layout BEFORE the legend and any figure-level axis label: it
        # re-lays the axes and ignores both, so placing them first strands them
        # relative to the final frame. Padding between panels is tight in both
        # orientations -- they share an axis, and a wide gutter would read as two
        # separate figures rather than one comparison.
        band = layout["band"]
        fig.tight_layout(rect=(0, band, 1, 1), pad=0.3,
                         **({"h_pad": 0.9} if vertical else {"w_pad": 0.8}))
        if vertical:
            # y offset: `supylabel` centres on the whole canvas, so the reserved
            # legend band would otherwise drag it below the panels.
            fig.supylabel("End-to-end latency", fontsize="medium",
                          x=0.005, y=0.5 + band / 2)
            legend_y = band + 0.005
        else:
            fig.supxlabel("Sequence length (tokens)", fontsize="medium", y=band - 0.035)
            legend_y = band - 0.075
        fig.legend(handles=handles, loc="upper center",
                   ncol=layout["legend_cols"] or len(handles), frameon=False,
                   bbox_to_anchor=(0.5, legend_y), handlelength=1.8,
                   columnspacing=1.2, fontsize=_LEGEND_FONTSIZE)

        fig.savefig(out, dpi=300, bbox_inches="tight")
        if args.pdf:
            fig.savefig(os.path.splitext(out)[0] + ".pdf", bbox_inches="tight")
    print(f"Wrote {out}" + ("  (+ .pdf)" if args.pdf else ""))


if __name__ == "__main__":
    main()

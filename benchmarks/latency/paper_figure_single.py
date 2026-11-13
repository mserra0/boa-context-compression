"""Shared body of the single-panel end-to-end latency figures.

`paper_figure_1_7b.py` and `paper_figure_4b.py` are the entry points; both are
thin configs over this module. Keeping one panel body guarantees that the two
figures differ in nothing but their data.

Both panels are deliberately bare -- no title, no decoder name on the canvas, no
run metadata (model, hardware, sweep settings and decoder scale are not drawn).

Legend sits inside the axes at the top left. On a log-log axis these curves rise
from bottom-left to top-right, so the top-left quadrant is dead space; putting
the legend there recovers the strip a figure-level legend would cost and keeps
the plot square enough for a narrow figure.

Styling (colour, marker, line style per method, the OOM terminator, the hero
stroke split) is imported from the sibling figures so none of them can drift
apart. Only the legend text is overridden here: these figures name the method,
the diagnostic plots name the implementation.
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
from benchmarks.latency.plot import (
    _ADAPTER_ORDER, _FALLBACK, _GRID_MAJOR, _LABELS, _STYLE, _STYLE_BY_ADAPTER,
    _fmt_ms, _fmt_tokens, _load, _oom_starts, _series,
)

_METRIC = "t_total_ms"

# Methods omitted from these end-to-end figures. `icae` is omitted from the figures only: its
# rows are still in every results file and `--drop` with no arguments restores it. It is omitted
# at both scales, because two figures compared across decoders have to carry the same set of
# methods.
_DROP = ("icae",)

# The caliper/ratio is quoted against the FASTER of the two Beacon windows.
# Quoting the slower one is a bigger number obtained by choosing the weaker
# baseline.
_ARROW_PAIR = ("ssm_kv", "beacon_long")

# One y range for both decoder scales, so the two figures can be compared
# directly instead of only reading orderings within each. This is only legitimate
# because every sweep behind these figures ran on the same accelerator model
# (the same GPU model) -- the two-panel
# `paper_figure.py` keeps its axes independent precisely because its runs did not.
# Covers 1.1 s (`full_ctx` at 512, Qwen3-1.7B) to 252.8 s (`beacon` at 262144,
# Qwen3-4B) with a decade floor and headroom for the ratio label.
# `--free-ylim` restores per-figure decade snapping.
_YLIM = (1e3, 4e5)

# Legend text, overriding `plot.py`'s implementation-facing names.
# "Boa" is the method's name; the architecture is described in the text, and a
# parenthetical in the legend only competes with it. "Full context" needs no
# "(no compression)" gloss -- the figure is about compression, so the absence is
# the obvious reading.
_LEGEND = dict(_LABELS)
_LEGEND["ssm_kv"] = "Boa"
_LEGEND["full_ctx"] = "Full context"


def _beacon_long_rows(path):
    """A second Beacon sweep's `beacon` rows, renamed to the `beacon_long` series.

    Window size is the only difference between two Beacon checkpoints, so the
    longer-window run is plotted as its own series rather than as a second file.
    There is no `beacon_long` adapter -- the rename happens here.

    The 1.7B curve comes from a checkpoint actually trained at window 2048. No 4B
    window-2048 checkpoint has been trained, so the 4B curve comes from the
    window-128 checkpoint run at `--beacon-window 2048`, which is sound for
    latency and nothing else (see `beacon/load.py`). The override flag is returned
    rather than swallowed so callers can report it at the scale where it applies.
    """
    meta, rows = _load(path)
    window = meta.get("beacon_window")
    # Guard the point of the series: an unnoticed window-128 file here would draw
    # two differently-styled curves over the same measurement.
    if window in (None, 128):
        raise SystemExit(f"{path}: beacon_window={window!r}; expected a longer-window run.")
    renamed = [dict(r, adapter="beacon_long") for r in rows if r["adapter"] == "beacon"]
    if not renamed:
        raise SystemExit(f"{path}: no `beacon` rows to rename.")
    return renamed, window, bool(meta.get("beacon_window_overridden"))


def _ratio(series, fast, slow):
    """(length, ratio) between two methods at the longest length both reached.

    Reported to stdout only. The stacked figure draws this as a caliper on the
    canvas; here the number is printed and nothing is drawn.
    """
    if fast not in series or slow not in series:
        return None
    x_fast, y_fast = series[fast]
    x_slow, y_slow = series[slow]
    common = sorted(set(x_fast) & set(x_slow))
    if not common:
        return None
    x = common[-1]
    return x, dict(zip(x_slow, y_slow))[x] / dict(zip(x_fast, y_fast))[x]


def _draw(ax, rows, ylim, xlabel=True, ylabel=True):
    series = _series(rows, _METRIC)
    ooms = _oom_starts(rows)
    order = [a for a in _ADAPTER_ORDER if a in series]
    order += [a for a in series if a not in order]

    for i, adapter in enumerate(order):
        xs, ys = series[adapter]
        kw = _hero_weight(dict(_STYLE_BY_ADAPTER.get(adapter)
                               or _FALLBACK[i % len(_FALLBACK)]), adapter)
        ax.plot(xs, ys, markersize=3.7 if adapter == _HERO else 3.0,
                markeredgecolor="white", markeredgewidth=0.5,
                label=_LEGEND.get(adapter, adapter), **kw)
        oom_at = ooms.get(adapter)
        if oom_at is not None and xs and oom_at > xs[-1]:
            ax.plot([xs[-1], oom_at], [ys[-1], ys[-1]], linestyle=":", linewidth=1.0,
                    color=kw["color"], zorder=kw.get("zorder", 2))
            ax.plot([oom_at], [ys[-1]], marker="x", markersize=5.5, markeredgewidth=1.3,
                    color=kw["color"], linestyle="none", zorder=kw.get("zorder", 2) + 1)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_ms))

    if ylim is None:
        # Floor at the decade below the data, so the lowest labelled tick sits
        # under the curves rather than through the middle of them.
        lo = min(min(ys) for _, ys in series.values())
        ax.set_ylim(bottom=10 ** math.floor(math.log10(lo)))
    else:
        ax.set_ylim(*ylim)

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
    ax.set_xticklabels(labels)
    # Off for a panel in a multi-panel figure: there the axis titles are set once
    # on the figure, because repeating them per panel is the clearest tell of a
    # figure assembled panel-by-panel rather than designed as one.
    if xlabel:
        ax.set_xlabel("Sequence length (tokens)")
    if ylabel:
        ax.set_ylabel("End-to-end latency")

    ax.tick_params(which="both", direction="out", length=2.5, width=0.5,
                   top=False, right=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.6)
        ax.spines[side].set_color("#9a9993")
    ax.set_axisbelow(True)
    ax.grid(True, which="major", axis="y", **_GRID_MAJOR)

    return series, ooms


def build_parser(doc, results, beacon_long, out):
    parser = argparse.ArgumentParser(description=doc.splitlines()[0])
    parser.add_argument("--results", default=results,
                        help="The five-way sweep. Supplies every curve except `beacon_long`.")
    parser.add_argument("--beacon-long", metavar="RESULTS", default=beacon_long,
                        help="A longer-window Beacon sweep. Its `beacon` rows are added as a "
                             "separate `beacon_long` curve.")
    parser.add_argument("--no-beacon-long", action="store_true",
                        help="Drop the second Beacon window, leaving the original five curves.")
    parser.add_argument("--drop", nargs="*", default=list(_DROP), metavar="ADAPTER",
                        help=f"Adapters to leave out (default: {' '.join(_DROP)}). "
                             "Pass with no values to plot every method in the file.")
    parser.add_argument("--free-ylim", action="store_true",
                        help="Snap the y axis to this figure's own data instead of the range "
                             "shared with the other decoder scale. Breaks cross-figure reading.")
    parser.add_argument("--out", default=out)
    parser.add_argument("--pdf", action="store_true",
                        help="Also write a vector PDF beside the PNG.")
    return parser


def render(args):
    """Draw and save one panel. `args` comes from `build_parser`."""
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    _, rows = _load(args.results)
    if not rows:
        raise SystemExit(f"No rows in {args.results}.")
    if not args.no_beacon_long and args.beacon_long:
        extra, window, overridden = _beacon_long_rows(args.beacon_long)
        rows = rows + extra
        print(f"  + beacon_long from {args.beacon_long} (window {window})")
        if overridden:
            print(f"    NOTE: window {window} was set at load time on a checkpoint trained at a "
                  f"different window.\n"
                  f"    Latency is valid (no beacon tensor's shape depends on the window); "
                  f"generation quality is not.\n"
                  f"    Note this wherever the figure is used.")

    if args.drop:
        kept = [r for r in rows if r["adapter"] not in set(args.drop)]
        if len(kept) == len(rows):
            print(f"  ! --drop {' '.join(args.drop)} matched no rows")
        rows = kept
        print(f"  - dropped: {', '.join(args.drop)}")

    with plt.style.context(_STYLE):
        fig, ax = plt.subplots(figsize=(3.4, 2.9))
        series, ooms = _draw(ax, rows, None if args.free_ylim else _YLIM)

        handles = [h for h in ax.get_lines() if h.get_label() in _LEGEND.values()]
        if any(a in series for a in ooms):
            handles.append(plt.Line2D([], [], color="#52514e", marker="x", markersize=5.5,
                                      markeredgewidth=1.3, linestyle=":", linewidth=1.0,
                                      label="out of memory"))
        # Inside the axes, top left, two columns. Matplotlib fills column-major,
        # so `_ADAPTER_ORDER` puts the featured method at the head of the left column and
        # the baselines follow. Two columns are half as tall as one, which keeps
        # the legend clear of the rising curves.
        #
        # Framed: sitting over the plotting area it needs a boundary to read as
        # an overlay rather than as stray marks among the data. Thin, light,
        # matching the axis spines, with an opaque fill so the gridlines behind
        # it do not run through the labels.
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

        # Summary numbers, printed rather than drawn: every
        # baseline against ours at the longest length both reached, then the
        # window ratio the two Beacon curves exist to show.
        for adapter in (a for a in _ADAPTER_ORDER if a in series and a != _HERO):
            got = _ratio(series, _HERO, adapter)
            if got is not None:
                at, factor = got
                print(f"  {_LEGEND.get(adapter, adapter)} / {_LEGEND[_HERO]}"
                      f" at {_fmt_tokens(at)} = {factor:.1f}x")
        window_ratio = _ratio(series, "beacon_long", "beacon")
        if window_ratio is not None:
            at, factor = window_ratio
            print(f"  beacon w=128 / w=2048 at {_fmt_tokens(at)} = {factor:.2f}x")

        fig.tight_layout()
        fig.savefig(args.out, dpi=300, bbox_inches="tight")
        if args.pdf:
            fig.savefig(os.path.splitext(args.out)[0] + ".pdf", bbox_inches="tight")
    print(f"Wrote {args.out}" + ("  (+ .pdf)" if args.pdf else ""))

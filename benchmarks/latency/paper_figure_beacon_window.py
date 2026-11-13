"""Paper figure: end-to-end latency on Qwen3-1.7B, with both Beacon windows.

One panel, six curves. The extra curve against the standard five-way sweep is
`beacon_long` -- the same Activation Beacon method at a compression window of
2048 instead of 128. Window size is the only difference between the two
checkpoints, so plotting them as separate series is what makes it readable as a
property of the method rather than a property of the run.

Both curves come from sweeps run together on the same hardware, so
the comparison between them is not confounded by host load. Everything except
`beacon_long` is read from the window-128 file, so the five baselines and the
Beacon they are plotted against all come from one process.

Deliberately bare -- no title, no run metadata in the frame.

Styling is imported from `plot.py` and the panel body from `paper_figure.py`, so
this figure cannot drift from the two-scale one.

Usage:
    uv run python -m benchmarks.latency.paper_figure_beacon_window --pdf
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401  registers the 'science' styles (required import)

from benchmarks.latency.paper_figure import _HERO, _panel, _hero_weight
from benchmarks.latency.plot import _ADAPTER_ORDER, _LABELS, _STYLE, _STYLE_BY_ADAPTER, _load

_OUT_DIR = "benchmarks/latency/outputs/qwen3_1-7b"

# The five baselines plus Beacon at its original window.
_BASE = f"{_OUT_DIR}/results_sota_1-7b_beacon_w128.jsonl"
# Only the `beacon` rows of this file are used, renamed to `beacon_long`.
_LONG = f"{_OUT_DIR}/results_sota_1-7b_beacon_w2048.jsonl"

_PANEL_LABEL = "Qwen3-1.7B"

# The caliper annotates ours against the FASTER of the two Beacon checkpoints.
# Quoting the ratio against the slower one would be a larger number obtained by
# choosing the weaker baseline.
_ARROW_PAIR = ("ssm_kv", "beacon_long")


def _merged_rows():
    """Window-128 sweep, plus the window-2048 sweep's Beacon as `beacon_long`."""
    meta_base, rows = _load(_BASE)
    meta_long, long_rows = _load(_LONG)

    # Guard the whole point of the figure: if either file is not the window it is
    # supposed to be, the two red curves are mislabelled and nothing downstream
    # would notice. `beacon_window` is stamped into meta by bench.py.
    for path, meta, want in ((_BASE, meta_base, 128), (_LONG, meta_long, 2048)):
        got = meta.get("beacon_window")
        if got != want:
            raise SystemExit(f"{path}: expected beacon_window={want}, found {got!r}")

    renamed = [dict(r, adapter="beacon_long") for r in long_rows if r["adapter"] == "beacon"]
    if not renamed:
        raise SystemExit(f"{_LONG}: no `beacon` rows to rename.")
    return rows + renamed


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="benchmarks/latency/outputs/paper/end_to_end_beacon_window.png")
    parser.add_argument("--pdf", action="store_true",
                        help="Also write a vector PDF beside the PNG.")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    rows = _merged_rows()

    with plt.style.context(_STYLE):
        # Narrow figure, one panel. Slightly taller than one row of the
        # two-scale figure: this one carries an extra curve and a six-entry legend.
        fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.7))
        series, ooms, factor = _panel(ax, rows, _PANEL_LABEL, show_xlabel=True,
                                      arrow_pair=_ARROW_PAIR)
        ax.set_ylabel("End-to-end latency", fontsize="medium")

        handles = [
            plt.Line2D([], [], markersize=3.7 if a == _HERO else 3.0,
                       markeredgecolor="white", markeredgewidth=0.5,
                       label=_LABELS.get(a, a),
                       **_hero_weight(dict(_STYLE_BY_ADAPTER[a]), a))
            for a in _ADAPTER_ORDER if a in series
        ]
        if any(a in series for a in ooms):
            handles.append(plt.Line2D([], [], color="#52514e", marker="x", markersize=5.5,
                                      markeredgewidth=1.3, linestyle=":", linewidth=1.0,
                                      label="out of memory"))
        # tight_layout BEFORE the legend: it re-lays the axes and ignores figure
        # legends, so anchoring first strands the legend far below the frame.
        # Reserve the legend strip in the layout rather than anchoring below the
        # canvas and letting `bbox_inches="tight"` grow into the gap -- that route
        # leaves whitespace proportional to how far the anchor overshot. Here
        # tight_layout packs the axes into the top 78% and the legend fills the
        # band underneath, so the gap is the band and nothing else.
        _LEGEND_BAND = 0.20
        fig.tight_layout(rect=(0, _LEGEND_BAND, 1, 1), pad=0.2)
        fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
                   bbox_to_anchor=(0.5, _LEGEND_BAND), handlelength=1.8,
                   columnspacing=1.2, fontsize="small")
        fig.savefig(args.out, dpi=300, bbox_inches="tight")
        if args.pdf:
            fig.savefig(os.path.splitext(args.out)[0] + ".pdf", bbox_inches="tight")

    # Summary numbers, printed rather than drawn.
    at = {a: dict(zip(*series[a])) for a in series}
    common = sorted(set(at["ssm_kv"]) & set(at["beacon"]) & set(at["beacon_long"]))
    L = common[-1]
    print(f"  at L={L}:  ssm_kv {at['ssm_kv'][L] / 1000:.1f}s   "
          f"beacon(w=128) {at['beacon'][L] / 1000:.1f}s   "
          f"beacon-long(w=2048) {at['beacon_long'][L] / 1000:.1f}s")
    print(f"  w=128 / w=2048 = {at['beacon'][L] / at['beacon_long'][L]:.2f}x")
    if factor is not None:
        print(f"  {_LABELS[_ARROW_PAIR[1]]} / {_LABELS[_ARROW_PAIR[0]]}"
              f" = {factor:.1f}x  (annotated as {factor:.0f}x)")
    print(f"Wrote {args.out}" + ("  (+ .pdf)" if args.pdf else ""))


if __name__ == "__main__":
    main()

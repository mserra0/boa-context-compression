"""Paper figure: end-to-end latency at Qwen3-4B, single panel.

Six methods swept, five drawn: `icae` is dropped (`--drop` with no values plots
it). Both Activation Beacon windows are drawn, so the figure shows what widening
the compression window does and does not buy.

The 4B `beacon_long` curve is the window-128 checkpoint *run* at window 2048 --
no 4B window-2048 checkpoint has been trained. Valid for latency only; the render
prints the warning; report it wherever the figure is used. The 1.7B twin
(`paper_figure_1_7b.py`) has a genuinely trained window-2048 checkpoint, so that
caveat belongs to this scale alone.

Everything about the panel itself lives in `paper_figure_single.py`, shared with
the 1.7B figure so the pair cannot drift.

Usage:
    uv run python -m benchmarks.latency.paper_figure_4b --pdf
"""

from benchmarks.latency.paper_figure_single import build_parser, render

# Default results file; pass `--results` to draw another sweep.
_RESULTS = "benchmarks/latency/outputs/qwen3_4b/results_sota_4b_beacon_w128.jsonl"
_BEACON_LONG = "benchmarks/latency/outputs/qwen3_4b/results_beacon_4b_w2048.jsonl"
_OUT = "benchmarks/latency/outputs/paper/end_to_end_4b.png"


def main():
    render(build_parser(__doc__, _RESULTS, _BEACON_LONG, _OUT).parse_args())


if __name__ == "__main__":
    main()

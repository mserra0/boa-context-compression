"""Paper figure: end-to-end latency at Qwen3-1.7B, single panel.

The 4B figure's twin (`paper_figure_4b.py`): same panel body, same styling, same
methods, same axes -- only the decoder changes. That is what makes the two
comparable side by side, and why the y range is fixed in the shared module rather
than snapped to each figure's own data.

Both Activation Beacon windows are drawn. Here both come from checkpoints
actually trained at their window (128 and 2048), so unlike the 4B figure this one
carries no load-time-override caveat. `icae` is dropped (`--drop` with no values
plots it).

`paper_figure_beacon_window.py` plots the same 1.7B data with the legend in a band
below the axes; this file is the one that matches the 4B figure.

Usage:
    uv run python -m benchmarks.latency.paper_figure_1_7b --pdf
"""

from benchmarks.latency.paper_figure_single import build_parser, render

_OUT_DIR = "benchmarks/latency/outputs/qwen3_1-7b"

# Both sweeps were run together on the same hardware, so the comparison
# between the windows is not confounded by host load. Everything except
# `beacon_long` is read from the window-128 file, so the baselines and the Beacon
# they are plotted against all come from one process.
_RESULTS = f"{_OUT_DIR}/results_sota_1-7b_beacon_w128.jsonl"
_BEACON_LONG = f"{_OUT_DIR}/results_sota_1-7b_beacon_w2048.jsonl"
_OUT = "benchmarks/latency/outputs/paper/end_to_end_1-7b.png"


def main():
    render(build_parser(__doc__, _RESULTS, _BEACON_LONG, _OUT).parse_args())


if __name__ == "__main__":
    main()

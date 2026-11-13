"""Activation Beacon baseline for the latency benchmark.

The backend itself is vendored (see `qwen3_official.py`); `load_beacon_model`
is the only entry point the benchmark uses.
"""

from .load import load_beacon_model

__all__ = ["load_beacon_model"]

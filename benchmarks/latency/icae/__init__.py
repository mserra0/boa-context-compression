"""ICAE baseline for the latency benchmark.

The encoder/decoder paths are vendored from the `baseline/icae` branch (see
`modeling.py`); `load_icae_model` is the only entry point the benchmark uses.
"""

from .load import load_icae_model
from .modeling import n_mem_for

__all__ = ["load_icae_model", "n_mem_for"]

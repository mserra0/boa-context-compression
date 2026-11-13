"""SAC baseline for the latency benchmark.

The backend is vendored (see `modeling.py`); `load_sac_model` and
`greedy_generate` are the only entry points the benchmark uses.
"""

from .load import greedy_generate, load_sac_model

__all__ = ["load_sac_model", "greedy_generate"]

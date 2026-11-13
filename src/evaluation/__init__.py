"""Evaluation package: dataset loaders, scorers, and metrics.

Hosts multiple evaluation datasets and swappable scorers (span-overlap
F1/EM metrics and an LLM-as-judge alternative). Inference is delegated to
``src.utils.model_runner.ModelRunner`` so evaluation reproduces the exact
training-aligned generation path (no train/inference gap).
"""

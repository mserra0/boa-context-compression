"""Result caching and console progress utilities for evaluation."""

import json
import os

from rich.console import Console
from rich.progress import (
    Progress, BarColumn, TextColumn, MofNCompleteColumn,
    TimeElapsedColumn, TimeRemainingColumn, SpinnerColumn,
)


def save_results(data: dict, path: str) -> None:
    """Save results dictionary as JSON, creating parent directories if needed."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def results_exist(path: str) -> bool:
    return os.path.exists(path)


def _make_progress(console: Console) -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("eta"),
        TimeRemainingColumn(),
        console=console,
    )

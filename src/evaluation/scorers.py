"""Evaluation scorers for computing per-sample and aggregate metrics."""

from .metrics import (
    exact_match_score,
    f1_score,
    metric_max_over_ground_truths,
)


def _mean(values):
    return sum(values) / len(values) if values else 0.0


class SpanMetricScorer:
    """Official MRQA F1/EM, taking the max over all gold answers per sample."""

    metric_keys = ("em", "f1")

    def score(self, samples: list, preds: list, progress=None, task_id=None):
        per_sample = []
        for sample, pred in zip(samples, preds):
            golds = sample["answers"]
            em = metric_max_over_ground_truths(exact_match_score, pred, golds)
            f1 = metric_max_over_ground_truths(f1_score, pred, golds)
            per_sample.append({"em": float(em), "f1": float(f1)})
            if progress is not None and task_id is not None:
                progress.update(task_id, advance=1)

        aggregate = {
            "em": _mean([s["em"] for s in per_sample]),
            "f1": _mean([s["f1"] for s in per_sample]),
        }
        return per_sample, aggregate

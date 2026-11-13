"""Scorers turn (samples, predictions) into per-sample and aggregate metrics.

Two interchangeable scorers share one interface so the runner is
scorer-agnostic:

- :class:`SpanMetricScorer` — official MRQA F1/EM span overlap (no server).
- :class:`LLMJudgeScorer`   — LLM-as-judge accuracy (alternative; needs a
  running vLLM server). Wraps the existing, tested
  ``src.utils.evaluate_utils.LLMJudge`` so there is no duplicated judge logic.

Interface contract:
    metric_keys -> tuple[str, ...]        # aggregate metric names, e.g. ("em", "f1")
    score(samples, preds) -> (per_sample, aggregate)
        per_sample : list[dict]  one metric dict per prediction
        aggregate  : dict        mean of each metric key over samples

Metrics are fractions in [0, 1]; formatting to percentages is the reporter's
job.
"""

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


class LLMJudgeScorer:
    """LLM-as-judge accuracy — the alternative to span metrics.

    Delegates to the existing ``LLMJudge`` (binary 0/1 per sample) and reports
    accuracy over valid responses. Requires a reachable vLLM judge server.
    """

    metric_keys = ("accuracy",)

    def __init__(self, judge_cfg: dict):
        # Imported lazily so span-metric evaluation has no openai/vllm dependency.
        from ..utils.evaluate_utils import LLMJudge

        self.judge = LLMJudge(judge_cfg)

    def score(self, samples: list, preds: list, progress=None, task_id=None):
        # LLMJudge renders a single reference string; join the gold aliases.
        scores = self.judge.score(
            [(s["question"], s.get("context", ""), "; ".join(s["answers"]), p)
             for s, p in zip(samples, preds)],
            progress=progress, task_id=task_id,
        )
        per_sample = [{"accuracy": float(sc) if sc >= 0 else None} for sc in scores]
        valid = [sc for sc in scores if sc >= 0]
        aggregate = {"accuracy": _mean(valid)}
        return per_sample, aggregate

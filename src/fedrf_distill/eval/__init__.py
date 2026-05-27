"""Evaluation pipeline for FedRF-Distill.

Extends the bare :class:`fedrf_distill.orchestration.ExperimentRunner` with:

* **Per-client metrics** — populates ``RoundResult.per_client_metrics`` with
  local accuracy / F1 / fidelity / bytes uploaded for each client every round.
* **Surrogate fidelity** — KL(teacher‖student) and argmax-agreement, so we can
  diagnose whether DP noise is hurting *the distillation step* or only the
  *final classification head*.
* **Membership inference attack (MIA)** — a Yeom-style loss-threshold attack
  on the final-round global surrogate, producing the empirical privacy curve
  referenced in ``docs/diagrams/10-mia-evaluation.html``.
* **Cross-run aggregation** — load N :class:`EvaluationResult` artifacts and
  emit a privacy-utility frontier table.
* **Markdown report** — paper-ready table grouped by (dataset, DP, partition).

Public API:

>>> from fedrf_distill.eval import EvaluationPipeline, EvaluationResult
>>> from fedrf_distill.eval.report import render_markdown_report
>>> from fedrf_distill.eval.aggregate import load_results, privacy_utility_table

The CLI driver lives at ``experiments/scripts/run_eval.py``.
"""

from __future__ import annotations

from fedrf_distill.eval.aggregate import (
    EvalSummary,
    load_results,
    privacy_utility_table,
)
from fedrf_distill.eval.mia import MIAResult, yeom_membership_inference
from fedrf_distill.eval.pipeline import EvaluationPipeline, EvaluationResult
from fedrf_distill.eval.report import render_markdown_report

__all__ = [
    "EvalSummary",
    "EvaluationPipeline",
    "EvaluationResult",
    "MIAResult",
    "load_results",
    "privacy_utility_table",
    "render_markdown_report",
    "yeom_membership_inference",
]

"""Shared dataclasses and type aliases used throughout the framework.

These types are deliberately framework-agnostic — they carry numpy arrays and
plain Python primitives so that the wire format between client and coordinator
remains identical regardless of which ML framework the client used locally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, NewType

import numpy as np
from numpy.typing import NDArray

# ── Strongly-typed aliases ───────────────────────────────────────────────────
ClientID = NewType("ClientID", int)
"""Stable identifier for a federated client (0-indexed)."""

SoftLabels = NDArray[np.float64]
"""(n_samples, n_classes) probability matrix. Each row must sum to 1.0."""

FeatureMatrix = NDArray[np.float64]
"""(n_samples, n_features) feature matrix."""

LabelVector = NDArray[np.int64]
"""(n_samples,) integer class labels."""


# ── Privacy bookkeeping ──────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class PrivacyBudget:
    """An (epsilon, delta)-DP budget snapshot.

    Frozen so callers cannot mutate a budget once issued. To "spend", construct
    a new `PrivacyBudget` via :py:meth:`spend`.
    """

    epsilon: float
    delta: float

    def __post_init__(self) -> None:
        if self.epsilon < 0:
            raise ValueError(f"epsilon must be >= 0, got {self.epsilon}")
        if not 0 <= self.delta < 1:
            raise ValueError(f"delta must be in [0, 1), got {self.delta}")

    def spend(self, eps: float, delta: float = 0.0) -> PrivacyBudget:
        """Return a new budget reduced by (eps, delta). Raises if it would go negative."""
        from fedrf_distill.core.exceptions import PrivacyBudgetExhausted

        new_eps = self.epsilon - eps
        new_delta = self.delta - delta
        if new_eps < -1e-12:
            raise PrivacyBudgetExhausted(eps, self.epsilon)
        return PrivacyBudget(epsilon=max(new_eps, 0.0), delta=max(new_delta, 0.0))

    def is_exhausted(self) -> bool:
        return self.epsilon <= 1e-12


# ── Federated artifacts ──────────────────────────────────────────────────────
@dataclass(slots=True)
class ClientUpdate:
    """Payload uploaded from a client to the coordinator after local distillation.

    The surrogate model is serialised opaquely (any picklable object) so that
    clients running different frameworks can still ship a compatible artifact.
    The coordinator never inspects the surrogate's internals — it only calls
    ``predict_proba(X_proxy)`` via the wrapping model adapter.
    """

    client_id: ClientID
    framework: str  # e.g. "sklearn", "xgboost", "lightgbm"
    surrogate_blob: bytes  # pickled model
    surrogate_class_path: str  # qualified import path for safe deserialisation
    n_samples: int
    classes: list[int]
    eps_spent_this_round: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GlobalArtifact:
    """Payload broadcast from coordinator to clients after aggregation."""

    round_idx: int
    global_surrogate_blob: bytes
    global_surrogate_class_path: str
    classes: list[int]
    aggregation_strategy: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Round-level results ──────────────────────────────────────────────────────
@dataclass(slots=True)
class ClientRoundMetrics:
    client_id: ClientID
    local_accuracy: float
    local_f1_macro: float
    local_auc: float | None
    surrogate_fidelity: float  # KL/agreement between teacher RF and surrogate
    bytes_uploaded: int
    eps_spent: float
    framework: str


@dataclass(slots=True)
class RoundResult:
    """Aggregated metrics for one federated round."""

    round_idx: int
    global_accuracy: float
    global_f1_macro: float
    global_auc: float | None
    per_client_metrics: list[ClientRoundMetrics]
    total_bytes_communicated: int
    cumulative_epsilon: float
    cumulative_delta: float
    wall_clock_seconds: float
    extra: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"[round {self.round_idx:3d}] "
            f"acc={self.global_accuracy:.4f}  "
            f"F1={self.global_f1_macro:.4f}  "
            f"bytes={self.total_bytes_communicated:>10,}  "
            f"eps={self.cumulative_epsilon:.3f}  "
            f"t={self.wall_clock_seconds:.1f}s"
        )

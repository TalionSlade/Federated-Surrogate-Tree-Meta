"""Stacking meta-learner.

Given a (just-broadcast) global surrogate ``G`` and a client's local data
``(X_local, y_local)``, stacking augments features as

    X_aug = [ X_local || G.predict_proba(X_local) ]

and refits the client's local model ``M`` on ``(X_aug, y_local)``. The
augmented columns inject *cross-client* dark knowledge into ``M`` while
preserving y_local as the ground truth signal — a classic stacking architecture
(Wolpert, 1992) re-cast for federated learning.

Important design choice: we **do not** modify ``y_local``. Some prior work
mixes y_local with G's argmax — that biases ``M`` toward ``G`` and erases the
local advantage. Stacking with original ``y`` keeps the local model *anchored
to its own labels* but *aware of* global structure.

The :class:`StackedAdapter` returned by :meth:`StackingMetaLearner.refine`
auto-augments inputs at predict-time so callers can use it as a drop-in
replacement for the original local model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Protocol

import numpy as np

from fedrf_distill.core.protocols import ModelAdapterProtocol
from fedrf_distill.core.types import FeatureMatrix, LabelVector, SoftLabels
from fedrf_distill.models.base import BaseModelAdapter


class ModelFactory(Protocol):
    """Callable that returns a fresh, unfitted :class:`ModelAdapterProtocol`."""

    def __call__(self) -> ModelAdapterProtocol: ...


@dataclass
class StackingMetaLearner:
    """Refines a local model by appending the global surrogate's soft labels."""

    name: ClassVar[str] = "stacking"

    # If True, normalise the augmented probability columns to zero-mean, unit-
    # variance before fitting — sometimes helps when local features are heavily
    # scaled (e.g. age in years, income in USD). Trees are invariant to monotone
    # transforms, so this is off by default.
    standardise_proba_cols: bool = False

    def refine(
        self,
        X_local: FeatureMatrix,
        y_local: LabelVector,
        global_surrogate: ModelAdapterProtocol,
        local_model_factory: ModelFactory,
    ) -> StackedAdapter:
        """Return a freshly-fitted local model trained on stacked features."""
        X_local = np.asarray(X_local, dtype=np.float64)
        proba = np.asarray(global_surrogate.predict_proba(X_local), dtype=np.float64)
        if self.standardise_proba_cols:
            proba, _stats = _fit_standardise(proba)
        else:
            _stats = None
        X_aug = np.concatenate([X_local, proba], axis=1)

        new_model = local_model_factory()
        new_model.fit(X_aug, y_local)
        return StackedAdapter(
            inner=new_model,
            global_surrogate=global_surrogate,
            standardise_stats=_stats,
        )


# ── Wrapper adapter that auto-augments at predict-time ──────────────────────
class StackedAdapter(BaseModelAdapter):
    """Wraps a refined local model + the global surrogate used during training.

    Predictions on raw X are turned into stacked predictions automatically.
    Serialisation includes the global surrogate so the wire format remains
    self-contained.
    """

    framework: ClassVar[str] = "stacked"

    def __init__(
        self,
        inner: ModelAdapterProtocol,
        global_surrogate: ModelAdapterProtocol,
        standardise_stats: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        super().__init__()
        self._inner = inner
        self._global = global_surrogate
        self._stats = standardise_stats
        self._fitted = True
        self._classes = list(inner.classes())

    # The base class' fit/predict_proba flow is bypassed because the inner
    # model is already fitted. We override the public API directly.
    def fit(self, X: FeatureMatrix, y: LabelVector) -> None:  # pragma: no cover
        raise RuntimeError("StackedAdapter is built pre-fitted by StackingMetaLearner.")

    def _fit_impl(self, X: FeatureMatrix, y: LabelVector) -> None:  # pragma: no cover
        raise RuntimeError("StackedAdapter has no _fit_impl.")

    def predict_proba(self, X: FeatureMatrix) -> SoftLabels:
        X_aug = self._augment(X)
        return self._inner.predict_proba(X_aug)

    def predict(self, X: FeatureMatrix) -> LabelVector:
        X_aug = self._augment(X)
        return self._inner.predict(X_aug)

    def n_estimators(self) -> int:
        return self._inner.n_estimators()

    def _predict_proba_impl(self, X: FeatureMatrix) -> SoftLabels:  # pragma: no cover
        return self.predict_proba(X)

    def _augment(self, X: FeatureMatrix) -> FeatureMatrix:
        X = np.asarray(X, dtype=np.float64)
        proba = np.asarray(self._global.predict_proba(X), dtype=np.float64)
        if self._stats is not None:
            mu, sd = self._stats
            proba = (proba - mu) / sd
        return np.concatenate([X, proba], axis=1)


# ── Internal helpers ─────────────────────────────────────────────────────────
def _fit_standardise(
    p: np.ndarray,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    mu = p.mean(axis=0, keepdims=True)
    sd = p.std(axis=0, keepdims=True)
    sd_safe = np.where(sd < 1e-12, 1.0, sd)
    return (p - mu) / sd_safe, (mu, sd_safe)

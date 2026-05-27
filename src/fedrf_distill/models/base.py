"""Abstract base class for every framework-specific model adapter.

The base class provides:
* Serialization via ``pickle`` (overridable by adapters that have a faster path,
  e.g. XGBoost's native JSON dump).
* A uniform ``predict_proba`` contract: output must be ``(n_samples, n_classes)``
  with rows summing to 1.0 even for binary classification (sklearn returns this
  natively; XGBoost binary returns shape ``(n,)`` and must be reshaped).
* A guard that fitting twice silently re-uses the same instance is rejected:
  each adapter is a one-shot estimator; refit by constructing a new adapter.
"""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np

from fedrf_distill.core.types import FeatureMatrix, LabelVector, SoftLabels


class BaseModelAdapter(ABC):
    """Common machinery for every concrete model adapter."""

    framework: ClassVar[str] = "abstract"

    def __init__(self) -> None:
        self._fitted: bool = False
        self._classes: list[int] | None = None

    # ── Required hooks ──────────────────────────────────────────────────────
    @abstractmethod
    def _fit_impl(self, X: FeatureMatrix, y: LabelVector) -> None:
        """Framework-specific fit logic."""

    @abstractmethod
    def _predict_proba_impl(self, X: FeatureMatrix) -> SoftLabels:
        """Framework-specific predict_proba; MUST return ``(n_samples, n_classes)``."""

    @abstractmethod
    def n_estimators(self) -> int:
        """Number of trees / boosting rounds in the underlying ensemble."""

    # ── Template methods ────────────────────────────────────────────────────
    def fit(self, X: FeatureMatrix, y: LabelVector) -> None:
        if self._fitted:
            raise RuntimeError(
                f"{type(self).__name__} is already fitted. "
                "Construct a new adapter instead of refitting."
            )
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        self._classes = sorted(int(c) for c in np.unique(y).tolist())
        self._fit_impl(X, y)
        self._fitted = True

    def predict_proba(self, X: FeatureMatrix) -> SoftLabels:
        self._require_fitted()
        X = np.asarray(X, dtype=np.float64)
        proba = self._predict_proba_impl(X)
        proba = np.asarray(proba, dtype=np.float64)
        if proba.ndim == 1:
            # Binary case where adapter returned only P(class=1). Reshape to canonical 2-col.
            proba = np.column_stack([1.0 - proba, proba])
        # Renormalise defensively to handle floating-point drift.
        row_sums = proba.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums <= 0, 1.0, row_sums)
        return proba / row_sums  # type: ignore[no-any-return]

    def predict(self, X: FeatureMatrix) -> LabelVector:
        proba = self.predict_proba(X)
        idx = proba.argmax(axis=1)
        classes_arr = np.asarray(self.classes(), dtype=np.int64)
        return classes_arr[idx]

    def classes(self) -> list[int]:
        self._require_fitted()
        assert self._classes is not None
        return list(self._classes)

    # ── Serialization ───────────────────────────────────────────────────────
    def serialize(self) -> bytes:
        """Default: pickle the whole adapter. Adapters may override for efficiency."""
        return pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def deserialize(cls, blob: bytes) -> BaseModelAdapter:
        obj = pickle.loads(blob)  # noqa: S301 – trusted internal channel
        if not isinstance(obj, BaseModelAdapter):
            raise TypeError(
                f"Deserialized object is {type(obj).__name__}, "
                "expected a BaseModelAdapter subclass."
            )
        return obj

    # ── Internals ───────────────────────────────────────────────────────────
    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(f"{type(self).__name__} must be fitted before use.")

    def __repr__(self) -> str:
        fitted = "fitted" if self._fitted else "unfitted"
        n_est = self.n_estimators() if self._fitted else "?"
        return f"<{type(self).__name__} framework={self.framework} n_est={n_est} ({fitted})>"

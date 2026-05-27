"""Surrogate distillation: fit a small student on the teacher's soft labels.

We support two distillation strategies via the ``DistillationMode`` enum:

* ``ARGMAX_WEIGHTED`` (default): take ``argmax(soft_labels)`` as the hard
  target and pass the confidence (``max(soft_labels)``) as ``sample_weight``.
  Works with any classifier-style adapter; loses some dark-knowledge but is
  fast and well-tested.

* ``MULTIOUTPUT_REGRESSION``: fit a multi-output regressor on the probability
  *vector* directly. Preserves dark knowledge but requires the regressor to
  emit a probability vector — currently only sklearn's
  ``DecisionTreeRegressor`` (multi-output) is wired in via a small wrapper.

For both strategies, the resulting student is wrapped in a
:class:`BaseModelAdapter` and uploaded by the client.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from sklearn.tree import DecisionTreeRegressor

from fedrf_distill.core.exceptions import DistillationError
from fedrf_distill.core.types import FeatureMatrix, LabelVector, SoftLabels
from fedrf_distill.models.base import BaseModelAdapter
from fedrf_distill.models.factory import make_model
from fedrf_distill.models.sklearn_adapter import SklearnDecisionTreeAdapter


class DistillationMode(str, Enum):
    """How to translate soft labels into a trainable target."""

    ARGMAX_WEIGHTED = "argmax_weighted"
    """Use argmax(p) as hard label, max(p) as sample weight."""

    MULTIOUTPUT_REGRESSION = "multioutput_regression"
    """Multi-output regression on the probability vector itself."""


@dataclass
class SurrogateDistiller:
    """Fits a compact surrogate model on soft labels.

    Parameters
    ----------
    student_framework:
        Adapter key used by :func:`make_model` (default ``"sklearn_dt"``).
    mode:
        Distillation strategy (:class:`DistillationMode`).
    student_kwargs:
        Hyper-params forwarded to the student adapter (e.g. ``max_depth=6``).
    random_state:
        Seed for the multi-output regressor when ``mode == MULTIOUTPUT_REGRESSION``.
    """

    student_framework: str = "sklearn_dt"
    mode: DistillationMode = DistillationMode.ARGMAX_WEIGHTED
    student_kwargs: dict[str, Any] | None = None
    random_state: int | None = None

    def distill(
        self,
        X: FeatureMatrix,
        soft_labels: SoftLabels,
        classes: list[int],
    ) -> BaseModelAdapter:
        """Train a surrogate to mimic ``soft_labels(X)``."""
        if X.shape[0] != soft_labels.shape[0]:
            raise DistillationError(
                f"Row count mismatch: X has {X.shape[0]}, soft_labels has {soft_labels.shape[0]}."
            )
        if soft_labels.shape[1] != len(classes):
            raise DistillationError(
                f"soft_labels has {soft_labels.shape[1]} columns "
                f"but {len(classes)} classes were declared."
            )

        kw = dict(self.student_kwargs or {})
        if self.random_state is not None and "random_state" not in kw:
            kw["random_state"] = self.random_state

        if self.mode == DistillationMode.ARGMAX_WEIGHTED:
            return self._distill_argmax_weighted(X, soft_labels, classes, kw)
        if self.mode == DistillationMode.MULTIOUTPUT_REGRESSION:
            return self._distill_multioutput(X, soft_labels, classes, kw)
        raise DistillationError(f"Unknown distillation mode: {self.mode}")

    # ── Strategy 1: argmax + weight ──────────────────────────────────────────
    def _distill_argmax_weighted(
        self,
        X: FeatureMatrix,
        soft_labels: SoftLabels,
        classes: list[int],
        student_kwargs: dict[str, Any],
    ) -> BaseModelAdapter:
        classes_arr = np.asarray(classes, dtype=np.int64)
        hard = classes_arr[np.argmax(soft_labels, axis=1)]
        weight = soft_labels.max(axis=1)

        # Degenerate edge case: ``argmax`` may yield only one unique class on
        # very skewed shards. Inject a tiny synthetic counter-example with
        # ε-weight so the tree can still be fit with the canonical class set.
        if len(np.unique(hard)) < 2 and len(classes) >= 2:
            X, hard, weight = _inject_synthetic_minority(
                X, hard, weight, classes_arr
            )

        student = make_model(self.student_framework, **student_kwargs)
        # The base adapter records ``_classes`` from ``y`` during fit. To keep
        # the canonical class set even when ``hard`` is single-class, we pre-set.
        student._classes = list(classes)  # noqa: SLF001
        student._fitted_classes_override = True  # hint flag (unused otherwise)
        # Try to pass sample_weight through if the underlying estimator supports it.
        underlying = getattr(student, "_estimator", None)
        if underlying is not None and "sample_weight" in _supported_fit_kwargs(
            underlying
        ):
            # Bypass the base ``fit`` so we can route the weight through.
            X = np.asarray(X, dtype=np.float64)
            hard = np.asarray(hard, dtype=np.int64)
            underlying.fit(X, hard, sample_weight=weight)
            student._fitted = True  # noqa: SLF001
        else:
            # Fall back to weight-less fit — slightly less faithful.
            student.fit(X, hard)
        return student

    # ── Strategy 2: multi-output regression ──────────────────────────────────
    def _distill_multioutput(
        self,
        X: FeatureMatrix,
        soft_labels: SoftLabels,
        classes: list[int],
        student_kwargs: dict[str, Any],
    ) -> BaseModelAdapter:
        if self.student_framework not in {"sklearn_dt", "sklearn", "dt"}:
            raise DistillationError(
                "MULTIOUTPUT_REGRESSION currently supports only sklearn_dt "
                f"as the student framework; got '{self.student_framework}'."
            )
        return _MultiOutputDTSurrogate(classes, **student_kwargs).fit_soft(
            X, soft_labels
        )


def _supported_fit_kwargs(est: Any) -> set[str]:
    """Best-effort introspection of an estimator's ``fit`` signature."""
    import inspect

    try:
        sig = inspect.signature(est.fit)
    except (TypeError, ValueError):
        return set()
    return set(sig.parameters)


def _inject_synthetic_minority(
    X: FeatureMatrix,
    y: LabelVector,
    weight: np.ndarray,
    classes: np.ndarray,
) -> tuple[FeatureMatrix, LabelVector, np.ndarray]:
    """Append one tiny-weight row per missing class so the tree has full class set."""
    present = set(int(c) for c in np.unique(y).tolist())
    missing = [int(c) for c in classes if int(c) not in present]
    if not missing:
        return X, y, weight

    extra_x = np.tile(X.mean(axis=0, keepdims=True), (len(missing), 1))
    extra_y = np.asarray(missing, dtype=np.int64)
    extra_w = np.full(len(missing), 1e-9, dtype=np.float64)
    return (
        np.vstack([X, extra_x]),
        np.concatenate([y, extra_y]),
        np.concatenate([weight, extra_w]),
    )


# ── Multi-output regression surrogate wrapper ────────────────────────────────
class _MultiOutputDTSurrogate(SklearnDecisionTreeAdapter):
    """Internal adapter that fits sklearn DecisionTreeRegressor on probability rows.

    We subclass the classifier adapter to inherit the metadata/serialisation
    plumbing, but override ``_fit_impl`` / ``_predict_proba_impl`` so the model
    is actually a regressor with multi-output targets.
    """

    framework = "sklearn_dt_mo"

    def __init__(
        self,
        classes: list[int],
        max_depth: int | None = 6,
        min_samples_leaf: int = 5,
        random_state: int | None = None,
        **extra: Any,
    ) -> None:
        # Skip the parent constructor's classifier setup — we set our own regressor.
        BaseModelAdapter.__init__(self)
        self._estimator = DecisionTreeRegressor(  # type: ignore[assignment]
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            **extra,
        )
        self._classes = list(classes)

    def fit_soft(
        self, X: FeatureMatrix, soft_labels: SoftLabels
    ) -> _MultiOutputDTSurrogate:
        """Fit on probability vectors rather than discrete labels."""
        X = np.asarray(X, dtype=np.float64)
        soft = np.asarray(soft_labels, dtype=np.float64)
        self._estimator.fit(X, soft)
        self._fitted = True
        return self

    def _fit_impl(self, X: FeatureMatrix, y: LabelVector) -> None:  # pragma: no cover
        raise DistillationError(
            "_MultiOutputDTSurrogate fits via fit_soft, not fit."
        )

    def _predict_proba_impl(self, X: FeatureMatrix) -> SoftLabels:
        # Regressor returns continuous values that may drift outside [0, 1].
        # Clip and renormalise; ``predict_proba`` in the base class also renormalises.
        raw = self._estimator.predict(X)
        if raw.ndim == 1:
            raw = raw.reshape(-1, 1)
        raw = np.clip(raw, 0.0, None)
        return np.asarray(raw, dtype=np.float64)

    def n_estimators(self) -> int:
        return 1

"""Adapters for scikit-learn tree estimators.

This module provides two adapters:

* :class:`SklearnRandomForestAdapter` — the **teacher** in our distillation
  pipeline. A random forest with hundreds of deep trees provides strong
  empirical predictions used as soft-label sources.
* :class:`SklearnDecisionTreeAdapter` — the **student / surrogate**. A single
  shallow tree trained on the teacher's soft labels. This is what's uploaded
  to the coordinator — small enough to fit in a few KB.

Both classes subclass :class:`BaseModelAdapter` and only override the three
hooks ``_fit_impl``, ``_predict_proba_impl``, ``n_estimators``. The public API
(fit/predict_proba/predict/serialize/deserialize/classes) is inherited verbatim.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from fedrf_distill.core.types import FeatureMatrix, LabelVector, SoftLabels
from fedrf_distill.models.base import BaseModelAdapter

try:
    from sklearn.ensemble import RandomForestClassifier
except ImportError as exc:  # pragma: no cover - sklearn is a hard dep
    raise RuntimeError("scikit-learn is required for FedRF-Distill.") from exc


# ── Random Forest (teacher) ──────────────────────────────────────────────────
class SklearnRandomForestAdapter(BaseModelAdapter):
    """Wraps :class:`sklearn.ensemble.RandomForestClassifier`.

    Used as the local **teacher** model on each client. Default hyper-params
    follow accepted FL/RF practice: 100 trees, no depth cap so trees fully grow
    the way Breiman intended, but with min_samples_leaf=2 to dampen pathological
    overfit on tiny client shards.

    All constructor kwargs are forwarded to the underlying sklearn estimator,
    so callers can tune `n_estimators`, `max_features`, `class_weight`, etc.
    via a config object.
    """

    framework: ClassVar[str] = "sklearn_rf"

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_leaf: int = 2,
        max_features: str | int | float | None = "sqrt",
        class_weight: str | dict[int, float] | None = None,
        n_jobs: int = -1,
        random_state: int | None = None,
        **extra: Any,
    ) -> None:
        super().__init__()
        self._estimator = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight=class_weight,
            n_jobs=n_jobs,
            random_state=random_state,
            **extra,
        )

    def _fit_impl(self, X: FeatureMatrix, y: LabelVector) -> None:
        self._estimator.fit(X, y)

    def _predict_proba_impl(self, X: FeatureMatrix) -> SoftLabels:
        proba = self._estimator.predict_proba(X)
        # sklearn may return a sub-set of classes if some labels are absent locally.
        # Re-pad to the canonical class set the adapter recorded at fit-time.
        return _pad_to_full_classes(proba, self._estimator.classes_, self._classes)

    def n_estimators(self) -> int:
        return int(self._estimator.n_estimators)


# ── Decision Tree (student / surrogate) ──────────────────────────────────────
class SklearnDecisionTreeAdapter(BaseModelAdapter):
    """Wraps :class:`sklearn.tree.DecisionTreeClassifier` for surrogate distillation.

    The default depth cap (``max_depth=6``) keeps surrogates compact —
    typically a few hundred nodes, well under 10 KB when pickled. This is the
    *communication cost* of the federated round: shipping one shallow tree
    instead of the full forest.

    When fitting on **soft labels** the caller should use
    :class:`SklearnDecisionTreeRegressorAdapter` instead, since classifier trees
    treat targets as discrete. Sklearn's classifier accepts `sample_weight`,
    so we route soft-label fitting through that path (see ``fit_soft``).
    """

    framework: ClassVar[str] = "sklearn_dt"

    def __init__(
        self,
        max_depth: int | None = 6,
        min_samples_leaf: int = 5,
        criterion: str = "gini",
        random_state: int | None = None,
        **extra: Any,
    ) -> None:
        super().__init__()
        self._estimator = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
            random_state=random_state,
            **extra,
        )

    def _fit_impl(self, X: FeatureMatrix, y: LabelVector) -> None:
        self._estimator.fit(X, y)

    def _predict_proba_impl(self, X: FeatureMatrix) -> SoftLabels:
        proba = self._estimator.predict_proba(X)
        return _pad_to_full_classes(proba, self._estimator.classes_, self._classes)

    def n_estimators(self) -> int:
        return 1


# ── Helpers ──────────────────────────────────────────────────────────────────
def _pad_to_full_classes(
    proba: np.ndarray,
    fitted_classes: np.ndarray,
    expected_classes: list[int] | None,
) -> SoftLabels:
    """Re-index a sub-class probability matrix to the global class set.

    When a client only sees classes ``{0, 2}`` (non-IID partition), sklearn
    returns a 2-column probability matrix. Downstream code expects a column for
    every globally-known class. We pad missing classes with zeros, preserving
    column order.
    """
    if expected_classes is None or len(fitted_classes) == len(expected_classes):
        return np.ascontiguousarray(proba, dtype=np.float64)

    full = np.zeros((proba.shape[0], len(expected_classes)), dtype=np.float64)
    expected_arr = np.asarray(expected_classes, dtype=np.int64)
    for j, cls in enumerate(fitted_classes):
        # Find the index of `cls` in expected_classes
        idx = int(np.where(expected_arr == cls)[0][0])
        full[:, idx] = proba[:, j]
    return full

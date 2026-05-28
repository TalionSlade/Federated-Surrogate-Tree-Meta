"""Adapter wrapping :class:`catboost.CatBoostClassifier`.

CatBoost shines on data with native categorical features and is the third
major boosting library we include for **cross-framework heterogeneity** (Hook B).

CatBoost prints a verbose training log by default; we disable it via
``verbose=False`` so federated runs stay log-tidy.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from fedrf_distill.core.types import FeatureMatrix, LabelVector, SoftLabels
from fedrf_distill.models.base import BaseModelAdapter

import catboost as cb  # noqa: E402


class CatBoostAdapter(BaseModelAdapter):
    """Wraps :class:`catboost.CatBoostClassifier`."""

    framework: ClassVar[str] = "catboost"

    def __init__(
        self,
        iterations: int = 200,
        depth: int = 6,
        learning_rate: float = 0.1,
        l2_leaf_reg: float = 3.0,
        random_state: int | None = None,
        thread_count: int = -1,
        verbose: bool = False,
        **extra: Any,
    ) -> None:
        super().__init__()
        self._estimator = cb.CatBoostClassifier(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            l2_leaf_reg=l2_leaf_reg,
            random_seed=random_state,
            thread_count=thread_count,
            verbose=verbose,
            **extra,
        )

    def _fit_impl(self, X: FeatureMatrix, y: LabelVector) -> None:
        self._estimator.fit(X, y)

    def _predict_proba_impl(self, X: FeatureMatrix) -> SoftLabels:
        proba = self._estimator.predict_proba(X)
        fitted_classes = np.asarray(self._estimator.classes_).astype(np.int64)
        if self._classes is None or len(fitted_classes) == len(self._classes):
            return np.ascontiguousarray(proba, dtype=np.float64)

        full = np.zeros((proba.shape[0], len(self._classes)), dtype=np.float64)
        expected = np.asarray(self._classes, dtype=np.int64)
        for j, cls in enumerate(fitted_classes):
            idx = int(np.where(expected == int(cls))[0][0])
            full[:, idx] = proba[:, j]
        return full

    def n_estimators(self) -> int:
        return int(self._estimator.tree_count_)

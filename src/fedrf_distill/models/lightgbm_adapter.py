"""Adapter wrapping :class:`lightgbm.LGBMClassifier`.

LightGBM is preferred over XGBoost on highly-sparse tabular data because of its
leaf-wise growth and EFB feature bundling. In the cross-framework heterogeneity
experiment (Hook B) we mix LightGBM clients with sklearn / XGBoost clients to
demonstrate the framework's true *teacher-agnostic* property.

Like the XGBoost adapter, this module's import is gated by the factory so an
install without LightGBM still functions.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from fedrf_distill.core.types import FeatureMatrix, LabelVector, SoftLabels
from fedrf_distill.models.base import BaseModelAdapter

import lightgbm as lgb  # noqa: E402


class LightGBMAdapter(BaseModelAdapter):
    """Wraps :class:`lightgbm.LGBMClassifier`."""

    framework: ClassVar[str] = "lightgbm"

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = -1,
        num_leaves: int = 63,
        learning_rate: float = 0.05,
        subsample: float = 0.9,
        colsample_bytree: float = 0.9,
        reg_lambda: float = 1.0,
        n_jobs: int = -1,
        random_state: int | None = None,
        verbose: int = -1,
        **extra: Any,
    ) -> None:
        super().__init__()
        self._estimator = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            **extra,
        )

    def _fit_impl(self, X: FeatureMatrix, y: LabelVector) -> None:
        self._estimator.fit(X, y)

    def _predict_proba_impl(self, X: FeatureMatrix) -> SoftLabels:
        proba = self._estimator.predict_proba(X)
        # LightGBM may return only the columns for classes it observed. Pad
        # missing columns with zeros, preserving the global class order.
        fitted_classes = np.asarray(self._estimator.classes_)
        if self._classes is None or len(fitted_classes) == len(self._classes):
            return np.ascontiguousarray(proba, dtype=np.float64)

        full = np.zeros((proba.shape[0], len(self._classes)), dtype=np.float64)
        expected = np.asarray(self._classes, dtype=np.int64)
        for j, cls in enumerate(fitted_classes):
            idx = int(np.where(expected == int(cls))[0][0])
            full[:, idx] = proba[:, j]
        return full

    def n_estimators(self) -> int:
        return int(self._estimator.n_estimators)

"""Adapter wrapping :class:`xgboost.XGBClassifier`.

Importing this module triggers ``import xgboost``; therefore the factory
imports it inside a ``try/except ImportError`` block so installations without
XGBoost still function. The hard dependency stays in ``pyproject.toml`` under
``optional-dependencies.xgboost``.

Implementation notes
--------------------
* Binary classification: XGBoost's ``predict_proba`` returns a 2-column matrix
  for ``XGBClassifier`` so we do **not** need to reshape — unlike the
  low-level ``Booster.predict`` path.
* Native serialisation: XGBoost has ``save_raw`` / ``load_raw`` which is faster
  and more compact than pickle, but for cross-platform safety with our
  wire format we still pickle the full adapter (Booster is picklable).
* ``use_label_encoder=False`` is set to silence XGBoost 1.x deprecation, and
  ``eval_metric`` is required to keep XGBoost 1.6+ from warning.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from fedrf_distill.core.types import FeatureMatrix, LabelVector, SoftLabels
from fedrf_distill.models.base import BaseModelAdapter

import xgboost as xgb  # noqa: E402 — import is the whole point of the module


class XGBoostAdapter(BaseModelAdapter):
    """Wraps :class:`xgboost.XGBClassifier` behind the uniform adapter API."""

    framework: ClassVar[str] = "xgboost"

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.9,
        colsample_bytree: float = 0.9,
        reg_lambda: float = 1.0,
        tree_method: str = "hist",
        n_jobs: int = -1,
        random_state: int | None = None,
        **extra: Any,
    ) -> None:
        super().__init__()
        self._estimator = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            tree_method=tree_method,
            n_jobs=n_jobs,
            random_state=random_state,
            use_label_encoder=False,
            eval_metric="mlogloss",
            **extra,
        )

    def _fit_impl(self, X: FeatureMatrix, y: LabelVector) -> None:
        # XGBoost requires labels to be 0..n_classes-1 contiguous integers. We
        # remap and stash the inverse transform on the estimator for later use.
        y_remapped, self._label_map = _remap_labels(y, self._classes)
        self._estimator.fit(X, y_remapped)

    def _predict_proba_impl(self, X: FeatureMatrix) -> SoftLabels:
        proba = self._estimator.predict_proba(X)
        # Column order in `proba` follows XGBoost's internal class ordering
        # (0..k-1 after our remap) which is, by construction, the same as
        # ``self._classes`` order. No re-padding needed.
        return np.ascontiguousarray(proba, dtype=np.float64)

    def n_estimators(self) -> int:
        return int(self._estimator.n_estimators)


# ── Helpers ──────────────────────────────────────────────────────────────────
def _remap_labels(
    y: LabelVector, classes: list[int] | None
) -> tuple[LabelVector, dict[int, int]]:
    """Remap arbitrary integer labels to 0..k-1 in the order given by ``classes``."""
    if classes is None:
        classes = sorted(int(c) for c in np.unique(y).tolist())
    forward = {orig: i for i, orig in enumerate(classes)}
    y_new = np.asarray([forward[int(v)] for v in y], dtype=np.int64)
    return y_new, forward

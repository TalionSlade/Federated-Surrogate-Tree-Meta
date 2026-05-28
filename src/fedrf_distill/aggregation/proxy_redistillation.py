"""Re-distillation aggregator — the framework's signature aggregation step.

Pipeline (executed once per round on the coordinator):

    proxy_X                          # public unlabeled features
       │
       ├── for each client k:
       │       student_k ← deserialise(surrogate_blob_k)
       │       P_k       ← student_k.predict_proba(proxy_X)
       │
       │   aggregate    P̄ = (1/K) · Σ_k P_k     (or weighted)
       │
       └── global_student ← SurrogateDistiller.distill(proxy_X, P̄)

The global student is then broadcast to clients in :class:`GlobalArtifact`.
This is *re-distillation* because we distil twice: once locally (teacher → local
student) and once globally (local students' ensemble → global student).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from fedrf_distill.core.exceptions import AggregationError
from fedrf_distill.core.types import ClientUpdate, FeatureMatrix
from fedrf_distill.distillation.surrogate import (
    DistillationMode,
    SurrogateDistiller,
)
from fedrf_distill.models.base import BaseModelAdapter
from fedrf_distill.utils.serialization import deserialize_adapter


@dataclass
class ProxyRedistillationAggregator:
    """Average client surrogate predictions on proxy data, then re-distil."""

    name: ClassVar[str] = "proxy_redistillation"

    student_framework: str = "sklearn_dt"
    distillation_mode: DistillationMode = DistillationMode.ARGMAX_WEIGHTED
    student_kwargs: dict = field(default_factory=dict)
    random_state: int | None = None
    weighting: str = "uniform"  # uniform | n_samples

    def aggregate(
        self,
        client_updates: list[ClientUpdate],
        proxy_X: FeatureMatrix,
        classes: list[int],
    ) -> BaseModelAdapter:
        if not client_updates:
            raise AggregationError("Cannot aggregate an empty client set.")
        if proxy_X.size == 0:
            raise AggregationError("Cannot aggregate with an empty proxy_X.")

        # 1) Get each client's predict_proba on the proxy
        per_client_p: list[np.ndarray] = []
        weights: list[float] = []
        for upd in client_updates:
            student = deserialize_adapter(
                upd.surrogate_blob, upd.surrogate_class_path
            )
            p = student.predict_proba(proxy_X)
            p = _align_columns(p, student.classes(), classes)
            per_client_p.append(p)
            weights.append(self._weight_for(upd))

        weights_arr = np.asarray(weights, dtype=np.float64)
        weights_arr /= weights_arr.sum() if weights_arr.sum() > 0 else 1.0
        ensemble = np.average(np.stack(per_client_p), axis=0, weights=weights_arr)
        # Renormalise defensively (averaging stochastic rows can drift epsilon-tiny)
        ensemble /= np.maximum(ensemble.sum(axis=1, keepdims=True), 1e-12)

        # 2) Re-distil into one global surrogate
        distiller = SurrogateDistiller(
            student_framework=self.student_framework,
            mode=self.distillation_mode,
            student_kwargs=self.student_kwargs,
            random_state=self.random_state,
        )
        return distiller.distill(proxy_X, ensemble, classes)

    def _weight_for(self, upd: ClientUpdate) -> float:
        if self.weighting == "uniform":
            return 1.0
        if self.weighting == "n_samples":
            return float(max(upd.n_samples, 1))
        raise AggregationError(f"Unknown weighting scheme: {self.weighting}")


def _align_columns(
    proba: np.ndarray,
    fitted_classes: list[int],
    global_classes: list[int],
) -> np.ndarray:
    """Re-index ``proba`` so its columns follow ``global_classes`` order."""
    if list(fitted_classes) == list(global_classes):
        return proba
    full = np.zeros((proba.shape[0], len(global_classes)), dtype=np.float64)
    g_index = {c: i for i, c in enumerate(global_classes)}
    for j, c in enumerate(fitted_classes):
        if c in g_index:
            full[:, g_index[c]] = proba[:, j]
    return full

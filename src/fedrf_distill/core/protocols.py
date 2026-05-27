"""Protocol definitions for every pluggable component.

We use `typing.Protocol` (structural typing) rather than ABCs so that:

* Implementors don't have to inherit any base class.
* mypy enforces interface compliance at static-analysis time.
* Adapters around third-party libraries (sklearn, xgboost, ...) need not be
  modified to fit.

Every protocol below is the contract a strategy must honour to be plugged
into the orchestration layer.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from fedrf_distill.core.types import (
    ClientID,
    ClientUpdate,
    FeatureMatrix,
    GlobalArtifact,
    LabelVector,
    PrivacyBudget,
    SoftLabels,
)


# ── Model layer ──────────────────────────────────────────────────────────────
@runtime_checkable
class ModelAdapterProtocol(Protocol):
    """Uniform interface over sklearn / xgboost / lightgbm / catboost tree models."""

    framework: str

    def fit(self, X: FeatureMatrix, y: LabelVector) -> None: ...
    def predict_proba(self, X: FeatureMatrix) -> SoftLabels: ...
    def predict(self, X: FeatureMatrix) -> LabelVector: ...
    def classes(self) -> list[int]: ...
    def serialize(self) -> bytes: ...
    @classmethod
    def deserialize(cls, blob: bytes) -> ModelAdapterProtocol: ...
    def n_estimators(self) -> int: ...


# ── Distillation layer ───────────────────────────────────────────────────────
@runtime_checkable
class SurrogateDistillerProtocol(Protocol):
    """Distills a teacher's `predict_proba` output into a compact student model.

    The student is a small decision tree (depth D) trained to mimic the teacher's
    soft predictions on `(X, soft_labels)`.
    """

    def distill(
        self,
        X: FeatureMatrix,
        soft_labels: SoftLabels,
        classes: list[int],
    ) -> ModelAdapterProtocol: ...


# ── Aggregation layer ────────────────────────────────────────────────────────
@runtime_checkable
class AggregatorProtocol(Protocol):
    """Combines K client surrogates into a single global surrogate.

    The default `ProxyRedistillationAggregator` runs each surrogate on a shared
    proxy dataset, averages the probability outputs, then re-distils that ensemble
    into one new global surrogate.
    """

    name: str

    def aggregate(
        self,
        client_updates: list[ClientUpdate],
        proxy_X: FeatureMatrix,
        classes: list[int],
    ) -> ModelAdapterProtocol: ...


# ── Privacy layer ────────────────────────────────────────────────────────────
@runtime_checkable
class DPMechanismProtocol(Protocol):
    """Adds calibrated noise to a tensor to achieve (eps, delta)-DP.

    Implementations must declare their sensitivity assumption and the noise
    distribution they sample from.
    """

    name: str
    sensitivity: float

    def privatize(
        self,
        x: SoftLabels,
        epsilon: float,
        delta: float = 0.0,
    ) -> SoftLabels: ...


# ── Meta-learning layer ──────────────────────────────────────────────────────
@runtime_checkable
class MetaLearnerProtocol(Protocol):
    """Refines a client's local model using the broadcast global surrogate.

    The canonical implementation is stacking: augment X_local with
    `global_surrogate.predict_proba(X_local)` as extra features, then retrain
    the local model on the augmented matrix with the original y_local.
    """

    name: str

    def refine(
        self,
        X_local: FeatureMatrix,
        y_local: LabelVector,
        global_surrogate: ModelAdapterProtocol,
        local_model_factory: Any,
    ) -> ModelAdapterProtocol: ...


# ── Data partitioning ────────────────────────────────────────────────────────
@runtime_checkable
class PartitionerProtocol(Protocol):
    """Splits a centralised dataset into per-client subsets.

    Implementations cover IID, Dirichlet non-IID, pathological non-IID, and
    quantity-skew partitioning.
    """

    name: str

    def partition(
        self,
        X: FeatureMatrix,
        y: LabelVector,
        n_clients: int,
        seed: int,
    ) -> list[tuple[FeatureMatrix, LabelVector]]: ...


# ── Topology layer ───────────────────────────────────────────────────────────
@runtime_checkable
class ClientProtocol(Protocol):
    """A federated client: holds local data, trains a model, distils a surrogate."""

    client_id: ClientID

    def local_train(self) -> ClientUpdate: ...
    def apply_global(self, artifact: GlobalArtifact) -> None: ...
    def evaluate_local(
        self, X: FeatureMatrix, y: LabelVector
    ) -> dict[str, float]: ...


@runtime_checkable
class CoordinatorProtocol(Protocol):
    """Server-side orchestrator: aggregates client updates, broadcasts globals."""

    def round(
        self,
        client_updates: list[ClientUpdate],
        round_idx: int,
        budget: PrivacyBudget,
    ) -> GlobalArtifact: ...

"""Per-client federated learning logic.

The federated client owns:

* its local data ``(X_local, y_local)``,
* a *teacher* (full random forest / boosting ensemble),
* a soft-label generator + DP mechanism + surrogate distiller (all injected
  via dependency injection so the client is policy-free).

Each round, the client:

#. Trains or refits the teacher on local data.
#. Predicts soft labels on local features.
#. Privatises soft labels via the DP mechanism (if any).
#. Distils into a small surrogate.
#. Packs the surrogate into a :class:`ClientUpdate` for upload.

On receiving the broadcast :class:`GlobalArtifact` it stashes the global
surrogate for use at evaluation time (and, optionally, for stacking).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from fedrf_distill.core.protocols import (
    DPMechanismProtocol,
    MetaLearnerProtocol,
    ModelAdapterProtocol,
    SurrogateDistillerProtocol,
)
from fedrf_distill.core.types import (
    ClientID,
    ClientUpdate,
    FeatureMatrix,
    GlobalArtifact,
    LabelVector,
)
from fedrf_distill.distillation.soft_labels import SoftLabelGenerator
from fedrf_distill.utils.serialization import (
    deserialize_adapter,
    serialize_adapter,
)


@dataclass
class FederatedClient:
    """Stateful federated client.

    Parameters
    ----------
    client_id:
        Unique identifier (0-indexed within the federation).
    X_local, y_local:
        The client's private training partition.
    teacher_factory:
        Zero-arg callable producing a fresh, unfitted *teacher* adapter.
    soft_label_generator:
        Calibrates teacher outputs (temperature, floor).
    distiller:
        Trains the small surrogate the client uploads.
    dp_mechanism:
        Differentially-private noise injector. May be ``NullDPMechanism`` to
        disable DP.
    epsilon_per_round, delta_per_round:
        Privacy budget the client is willing to spend in a single round.
    meta_learner:
        Optional :class:`MetaLearnerProtocol` for stacking-style refinement
        after every global broadcast.
    """

    client_id: ClientID
    X_local: FeatureMatrix
    y_local: LabelVector
    teacher_factory: Any  # Callable[[], ModelAdapterProtocol]
    soft_label_generator: SoftLabelGenerator
    distiller: SurrogateDistillerProtocol
    dp_mechanism: DPMechanismProtocol
    epsilon_per_round: float = 1.0
    delta_per_round: float = 1e-5
    meta_learner: MetaLearnerProtocol | None = None

    teacher: ModelAdapterProtocol | None = field(default=None, init=False)
    global_surrogate: ModelAdapterProtocol | None = field(default=None, init=False)
    refined_model: ModelAdapterProtocol | None = field(default=None, init=False)
    classes: list[int] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.X_local = np.asarray(self.X_local, dtype=np.float64)
        self.y_local = np.asarray(self.y_local, dtype=np.int64)
        if self.X_local.shape[0] != self.y_local.shape[0]:
            raise ValueError(
                f"client {self.client_id}: X has {self.X_local.shape[0]} rows, "
                f"y has {self.y_local.shape[0]}."
            )

    # ── Round operations ────────────────────────────────────────────────────
    def local_train(self) -> ClientUpdate:
        """Teacher fit → soft labels → DP → surrogate → packaged update."""
        # 1) Train teacher
        self.teacher = self.teacher_factory()
        self.teacher.fit(self.X_local, self.y_local)
        self.classes = self.teacher.classes()

        # 2) Soft labels
        proba = self.soft_label_generator.generate(self.teacher, self.X_local)

        # 3) Privacy
        proba_priv = self.dp_mechanism.privatize(
            proba, epsilon=self.epsilon_per_round, delta=self.delta_per_round
        )

        # 4) Distil → surrogate
        surrogate = self.distiller.distill(self.X_local, proba_priv, self.classes)
        blob, class_path = serialize_adapter(surrogate)

        # 5) Package
        return ClientUpdate(
            client_id=self.client_id,
            framework=surrogate.framework,
            surrogate_blob=blob,
            surrogate_class_path=class_path,
            n_samples=int(self.X_local.shape[0]),
            classes=list(self.classes),
            eps_spent_this_round=float(self.epsilon_per_round)
            if self.dp_mechanism.name != "null"
            else 0.0,
            metadata={"teacher_framework": self.teacher.framework},
        )

    def apply_global(self, artifact: GlobalArtifact) -> None:
        """Store the broadcast global surrogate; optionally run meta-learning."""
        self.global_surrogate = deserialize_adapter(
            artifact.global_surrogate_blob, artifact.global_surrogate_class_path
        )

        if self.meta_learner is not None:
            # The local model factory restores a teacher-style adapter; we want
            # the same teacher type as last round.
            self.refined_model = self.meta_learner.refine(  # type: ignore[assignment]
                X_local=self.X_local,
                y_local=self.y_local,
                global_surrogate=self.global_surrogate,
                local_model_factory=self.teacher_factory,
            )

    # ── Evaluation ──────────────────────────────────────────────────────────
    def evaluate_local(
        self, X: FeatureMatrix, y: LabelVector
    ) -> dict[str, float]:
        """Return accuracy / F1 of the *current best* local view of the model."""
        model = self.refined_model or self.global_surrogate or self.teacher
        if model is None:
            return {"accuracy": float("nan"), "f1_macro": float("nan")}
        from fedrf_distill.metrics.classification import classification_metrics

        y_pred = model.predict(X)
        return classification_metrics(y, y_pred)

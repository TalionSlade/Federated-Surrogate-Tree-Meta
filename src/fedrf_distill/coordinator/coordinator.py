"""Server-side coordinator.

A federated coordinator:

* Holds the *proxy dataset* used for re-distillation.
* Holds an :class:`AggregatorProtocol` that fuses the K client surrogates
  into one global surrogate.
* Maintains a :class:`PrivacyAccountantProtocol` that accumulates the running
  (ε, δ) cost across rounds.
* Emits a :class:`GlobalArtifact` per round for broadcast.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from fedrf_distill.core.protocols import (
    AggregatorProtocol,
)
from fedrf_distill.core.types import (
    ClientUpdate,
    FeatureMatrix,
    GlobalArtifact,
    PrivacyBudget,
)
from fedrf_distill.privacy.accountant import (
    BasicCompositionAccountant,
    PrivacyAccountantProtocol,
)
from fedrf_distill.utils.serialization import serialize_adapter


@dataclass
class FederatedCoordinator:
    """The server side of FedRF-Distill.

    Parameters
    ----------
    aggregator:
        Strategy that merges client surrogates into the global surrogate.
    proxy_X:
        Unlabelled feature matrix shared by all clients (server-held).
    classes:
        The canonical global class set; all client classes must be a subset.
    accountant:
        Tracks cumulative privacy expenditure across rounds.
    target_budget:
        Total (ε, δ) the coordinator is willing to spend. The coordinator
        raises :class:`PrivacyBudgetExhausted` once spending exceeds this.
    """

    aggregator: AggregatorProtocol
    proxy_X: FeatureMatrix
    classes: list[int]
    accountant: PrivacyAccountantProtocol = field(
        default_factory=BasicCompositionAccountant
    )
    target_budget: PrivacyBudget | None = None
    _history: list[GlobalArtifact] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self.proxy_X = np.asarray(self.proxy_X, dtype=np.float64)

    def round(
        self,
        client_updates: list[ClientUpdate],
        round_idx: int,
        budget: PrivacyBudget,
    ) -> GlobalArtifact:
        """Run one aggregation round and return the artifact to broadcast.

        ``budget`` is the *remaining* total budget at entry; we use it only
        to ensure we don't push ourselves over.
        """
        # 1) Record each client's privacy cost into the ledger.
        for upd in client_updates:
            self.accountant.record(
                round_idx=round_idx,
                client_id=int(upd.client_id),
                mechanism=upd.metadata.get("dp_mechanism", "unknown"),  # type: ignore[arg-type]
                epsilon=float(upd.eps_spent_this_round),
                delta=0.0,
                sigma=upd.metadata.get("sigma"),  # type: ignore[arg-type]
            )

        # 2) Refuse to continue if the running ledger has exceeded the target.
        running_eps, _ = self.accountant.cumulative()
        if self.target_budget is not None and running_eps > self.target_budget.epsilon:
            from fedrf_distill.core.exceptions import PrivacyBudgetExhausted

            raise PrivacyBudgetExhausted(
                requested_eps=running_eps,
                remaining_eps=self.target_budget.epsilon,
            )

        # 3) Aggregate
        global_student = self.aggregator.aggregate(
            client_updates, self.proxy_X, self.classes
        )

        # 4) Package
        blob, class_path = serialize_adapter(global_student)
        artifact = GlobalArtifact(
            round_idx=round_idx,
            global_surrogate_blob=blob,
            global_surrogate_class_path=class_path,
            classes=list(self.classes),
            aggregation_strategy=self.aggregator.name,
            metadata={
                "n_clients": len(client_updates),
                "proxy_size": int(self.proxy_X.shape[0]),
                "cumulative_eps": running_eps,
            },
        )
        self._history.append(artifact)
        return artifact

    @property
    def history(self) -> list[GlobalArtifact]:
        return list(self._history)

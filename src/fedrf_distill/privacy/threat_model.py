"""Formal threat model for FedRF-Distill.

A threat model declares **(a) what is protected**, **(b) what an adversary can
do**, and **(c) what the system guarantees**. Stating it in a typed dataclass
makes it impossible to "forget" any axis when writing the paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AdversaryCapability(str, Enum):
    """What the attacker can observe."""

    HONEST_BUT_CURIOUS_SERVER = "honest_but_curious_server"
    """Coordinator follows the protocol but inspects all client uploads."""

    EAVESDROPPER = "eavesdropper"
    """Observes wire traffic only; cannot inject."""

    COLLUDING_CLIENTS = "colluding_clients"
    """A subset of clients pool their views to attack a target client."""

    EXTERNAL_QUERIER = "external_querier"
    """Queries the released global surrogate; no internal access."""


class ProtectedAsset(str, Enum):
    """What we promise to keep private."""

    INDIVIDUAL_TRAINING_RECORD = "individual_training_record"
    """Membership privacy: whether any single training row was in a client's set."""

    LABEL_DISTRIBUTION = "label_distribution"
    """The per-class frequencies on each client (relevant under quantity skew)."""

    LOCAL_MODEL_STRUCTURE = "local_model_structure"
    """The teacher's internal tree splits — not protected here; out of scope."""


@dataclass(frozen=True, slots=True)
class ThreatModel:
    """Compact description of who's attacking what, with what assumptions.

    Defaults are the most common federated setting: honest-but-curious server,
    membership privacy for individual training records via row-level DP.
    """

    adversary: AdversaryCapability = AdversaryCapability.HONEST_BUT_CURIOUS_SERVER
    protected: tuple[ProtectedAsset, ...] = (
        ProtectedAsset.INDIVIDUAL_TRAINING_RECORD,
    )
    target_epsilon: float = 1.0
    target_delta: float = 1e-5
    notes: str = ""

    def summary(self) -> str:
        return (
            f"Adversary: {self.adversary.value}\n"
            f"Protects: {[p.value for p in self.protected]}\n"
            f"Target:   ε ≤ {self.target_epsilon}, δ ≤ {self.target_delta:.0e}\n"
            f"Notes:    {self.notes or '(none)'}"
        )

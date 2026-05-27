"""Differential Privacy (DP) primitives for FedRF-Distill.

This module realises **Hook A**: a formal (ε, δ)-DP guarantee on the soft
labels exchanged between clients and the coordinator.

Layout
------
* :mod:`mechanisms`  — concrete noise injectors (Laplace, Gaussian, etc.).
* :mod:`accountant`  — bookkeeping of cumulative (ε, δ) across rounds.
* :mod:`threat_model` — formal definition of what is *protected* and *not*.
* :mod:`attack`      — empirical attacks (Shokri-style MIA) used to validate
  the formal guarantees.

All mechanisms operate on **probability matrices** (the soft labels) and have
sensitivity Δ = 2 (L1) or Δ = √2 (L2), since each row sums to 1 and changing
one input sample swaps at most one row of two-norm 1.
"""

from fedrf_distill.privacy.accountant import (
    BasicCompositionAccountant,
    PrivacyAccountantProtocol,
    PrivacyLedgerEntry,
    RDPAccountant,
)
from fedrf_distill.privacy.mechanisms import (
    BaseDPMechanism,
    GaussianMechanism,
    LaplaceMechanism,
    NullDPMechanism,
)
from fedrf_distill.privacy.threat_model import ThreatModel

__all__ = [
    "BaseDPMechanism",
    "GaussianMechanism",
    "LaplaceMechanism",
    "NullDPMechanism",
    "PrivacyAccountantProtocol",
    "PrivacyLedgerEntry",
    "BasicCompositionAccountant",
    "RDPAccountant",
    "ThreatModel",
]

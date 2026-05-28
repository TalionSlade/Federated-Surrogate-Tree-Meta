"""Core protocols, types, and exceptions shared across the framework."""

from fedrf_distill.core.exceptions import (
    AggregationError,
    DistillationError,
    FedRFDistillError,
    PrivacyBudgetExhausted,
    UnsupportedModelError,
)
from fedrf_distill.core.protocols import (
    AggregatorProtocol,
    ClientProtocol,
    CoordinatorProtocol,
    DPMechanismProtocol,
    MetaLearnerProtocol,
    ModelAdapterProtocol,
    PartitionerProtocol,
    SurrogateDistillerProtocol,
)
from fedrf_distill.core.types import (
    ClientID,
    ClientUpdate,
    GlobalArtifact,
    PrivacyBudget,
    RoundResult,
    SoftLabels,
)

__all__ = [
    # exceptions
    "FedRFDistillError",
    "AggregationError",
    "DistillationError",
    "PrivacyBudgetExhausted",
    "UnsupportedModelError",
    # protocols
    "ModelAdapterProtocol",
    "SurrogateDistillerProtocol",
    "AggregatorProtocol",
    "DPMechanismProtocol",
    "ClientProtocol",
    "CoordinatorProtocol",
    "MetaLearnerProtocol",
    "PartitionerProtocol",
    # types
    "ClientID",
    "SoftLabels",
    "ClientUpdate",
    "GlobalArtifact",
    "PrivacyBudget",
    "RoundResult",
]

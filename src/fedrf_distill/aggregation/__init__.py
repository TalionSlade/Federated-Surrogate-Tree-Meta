"""Aggregation strategies for combining K client surrogates → one global surrogate.

Three concrete strategies are provided:

* :class:`ProxyRedistillationAggregator` (default, novel) — runs each client
  surrogate on a shared proxy ``X``, averages the probabilities, then *re-
  distils* the ensemble into one new global surrogate. The re-distillation
  step is the contribution that gives the framework its name.

* :class:`ConfidenceWeightedAggregator` — same as above but the average is
  weighted by each client's reported number of training samples (à la FedAvg).

* :class:`MajorityVoteAggregator` — hard-label majority vote over client
  predictions; included as a baseline.

All strategies implement :class:`AggregatorProtocol`.
"""

from fedrf_distill.aggregation.confidence_weighted import (
    ConfidenceWeightedAggregator,
)
from fedrf_distill.aggregation.majority_vote import MajorityVoteAggregator
from fedrf_distill.aggregation.proxy_redistillation import (
    ProxyRedistillationAggregator,
)

__all__ = [
    "ProxyRedistillationAggregator",
    "ConfidenceWeightedAggregator",
    "MajorityVoteAggregator",
]

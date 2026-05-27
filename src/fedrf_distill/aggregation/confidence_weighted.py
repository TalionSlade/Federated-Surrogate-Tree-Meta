"""FedAvg-style sample-count-weighted aggregator.

Identical to :class:`ProxyRedistillationAggregator` except clients with more
local data contribute proportionally more to the ensemble average. This is the
classical FedAvg weighting; we keep it as a separate class so configs can swap
between unweighted and weighted via a string in YAML.
"""

from __future__ import annotations

from dataclasses import dataclass

from fedrf_distill.aggregation.proxy_redistillation import (
    ProxyRedistillationAggregator,
)


@dataclass
class ConfidenceWeightedAggregator(ProxyRedistillationAggregator):
    """Re-distil with sample-count weighting."""

    name: str = "confidence_weighted"  # type: ignore[assignment]
    weighting: str = "n_samples"

"""Orchestration: turn a :class:`ExperimentConfig` into a running experiment.

The orchestration layer is what experiment scripts call. It wires up the
factory of components (model, distiller, DP mechanism, partitioner,
aggregator, meta-learner) according to the config and drives the round loop.
"""

from fedrf_distill.orchestration.runner import (
    ExperimentRunner,
    build_clients,
    build_coordinator,
)

__all__ = ["ExperimentRunner", "build_clients", "build_coordinator"]

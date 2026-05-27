"""Proxy-dataset construction for the re-distillation aggregator.

The coordinator needs a small *unlabeled* feature matrix to evaluate every
client surrogate on a common substrate. We support three sources of proxy
data, in order of preference:

#. **Synthetic** — drawn from a Gaussian mixture or uniform distribution over
   the feature range. No data leaves any client.
#. **Public** — supply an external dataset (e.g. UCI Adult test split) whose
   distribution roughly matches the federated learning task.
#. **Server-held** — coordinator owns a small labelled validation set; we
   keep only the features.

Choosing well is critical: too-narrow proxies under-represent client diversity
and bias the global student.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np

from fedrf_distill.core.exceptions import ProxyDataError
from fedrf_distill.core.types import FeatureMatrix


class ProxySource(str, Enum):
    SYNTHETIC_UNIFORM = "synthetic_uniform"
    SYNTHETIC_GAUSSIAN = "synthetic_gaussian"
    EXTERNAL = "external"


@dataclass
class ProxyDataConfig:
    """Declarative spec for proxy-dataset construction."""

    source: ProxySource = ProxySource.SYNTHETIC_UNIFORM
    n_samples: int = 2000
    seed: int = 0

    # When source == EXTERNAL the caller passes raw features in to
    # ``build_proxy_dataset``. The fields below are unused.
    n_features: int | None = None
    feature_low: float | None = None
    feature_high: float | None = None
    feature_mean: np.ndarray | None = None
    feature_cov: np.ndarray | None = None


def build_proxy_dataset(
    cfg: ProxyDataConfig,
    external: FeatureMatrix | None = None,
) -> FeatureMatrix:
    """Construct the proxy feature matrix described by ``cfg``.

    Parameters
    ----------
    cfg:
        Declarative configuration.
    external:
        Required when ``cfg.source == EXTERNAL``; otherwise ignored.
    """
    rng = np.random.default_rng(cfg.seed)
    if cfg.source == ProxySource.EXTERNAL:
        if external is None or external.size == 0:
            raise ProxyDataError(
                "ProxySource.EXTERNAL was selected but no `external` array passed."
            )
        # Sub-sample to n_samples
        if external.shape[0] > cfg.n_samples:
            idx = rng.choice(external.shape[0], size=cfg.n_samples, replace=False)
            external = external[idx]
        return np.asarray(external, dtype=np.float64)

    if cfg.n_features is None or cfg.n_features <= 0:
        raise ProxyDataError(
            "ProxyDataConfig.n_features must be set for synthetic proxies."
        )

    if cfg.source == ProxySource.SYNTHETIC_UNIFORM:
        low = cfg.feature_low if cfg.feature_low is not None else 0.0
        high = cfg.feature_high if cfg.feature_high is not None else 1.0
        return rng.uniform(low, high, size=(cfg.n_samples, cfg.n_features))

    if cfg.source == ProxySource.SYNTHETIC_GAUSSIAN:
        mean = (
            cfg.feature_mean
            if cfg.feature_mean is not None
            else np.zeros(cfg.n_features)
        )
        cov = (
            cfg.feature_cov
            if cfg.feature_cov is not None
            else np.eye(cfg.n_features)
        )
        return rng.multivariate_normal(mean, cov, size=cfg.n_samples)

    raise ProxyDataError(f"Unsupported proxy source: {cfg.source}")

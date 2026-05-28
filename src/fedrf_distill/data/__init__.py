"""Data partitioning and dataset loaders.

Two sub-areas:

* :mod:`partitioners` — split a centralised ``(X, y)`` dataset into K per-
  client shards under various IID / non-IID regimes.
* :mod:`loaders` — fetch and normalise the 10 benchmark datasets.

Both areas live under one package so experiment configs can refer to "data" as
a single namespace.
"""

from fedrf_distill.data.partitioners import (
    DirichletPartitioner,
    IIDPartitioner,
    PathologicalPartitioner,
    QuantitySkewPartitioner,
)
from fedrf_distill.data.proxy import (
    ProxyDataConfig,
    build_proxy_dataset,
)

__all__ = [
    "IIDPartitioner",
    "DirichletPartitioner",
    "PathologicalPartitioner",
    "QuantitySkewPartitioner",
    "ProxyDataConfig",
    "build_proxy_dataset",
]

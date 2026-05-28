"""Common scaffolding shared by every dataset loader.

* :class:`Dataset` — typed container for ``(X, y)`` plus metadata.
* :class:`DatasetRegistry` — string → loader-callable registry, populated by
  decorating each loader function with :func:`register`.
* :func:`load_dataset` — public entry point used by experiments.

Caching: every loader is given a directory ``~/.cache/fedrf_distill/<key>/``
that it may use however it likes. We do NOT impose a fixed cache format —
some datasets pull from sklearn's built-in cache, others use OpenML's, others
download CSVs ourselves.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from fedrf_distill.core.types import FeatureMatrix, LabelVector

LoaderFn = Callable[[Path], "Dataset"]


@dataclass(slots=True)
class Dataset:
    """A loaded benchmark dataset."""

    name: str
    X: FeatureMatrix
    y: LabelVector
    task_type: str  # "binary" | "multiclass"
    n_classes: int
    feature_names: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)


# ── Registry ────────────────────────────────────────────────────────────────
class DatasetRegistry:
    """Map of canonical dataset name → loader callable."""

    _store: dict[str, LoaderFn] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[LoaderFn], LoaderFn]:
        def deco(fn: LoaderFn) -> LoaderFn:
            cls._store[name.lower()] = fn
            return fn

        return deco

    @classmethod
    def get(cls, name: str) -> LoaderFn:
        key = name.lower()
        if key not in cls._store:
            available = ", ".join(sorted(cls._store)) or "<none>"
            raise KeyError(f"Unknown dataset '{name}'. Available: {available}.")
        return cls._store[key]

    @classmethod
    def available(cls) -> list[str]:
        return sorted(cls._store)


def _cache_dir(name: str) -> Path:
    base = Path(os.environ.get("FEDRF_CACHE", Path.home() / ".cache" / "fedrf_distill"))
    p = base / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_dataset(name: str) -> Dataset:
    """Public entry point used by orchestration / experiment scripts."""
    loader = DatasetRegistry.get(name)
    cache = _cache_dir(name.lower())
    ds = loader(cache)
    # Final shape sanity
    if ds.X.shape[0] != ds.y.shape[0]:
        raise ValueError(
            f"{name}: X has {ds.X.shape[0]} rows, y has {ds.y.shape[0]}."
        )
    ds.X = np.ascontiguousarray(ds.X, dtype=np.float64)
    ds.y = np.ascontiguousarray(ds.y, dtype=np.int64)
    return ds

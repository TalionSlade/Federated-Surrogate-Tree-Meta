"""Forest Cover Type (UCI). Multiclass terrain — 7 classes.

Sklearn ships a built-in fetcher (~500K rows). The first call may take ~30 s.
"""

from __future__ import annotations

from pathlib import Path

from sklearn.datasets import fetch_covtype

from fedrf_distill.data.loaders.base import Dataset, DatasetRegistry


@DatasetRegistry.register("covertype")
def load(cache_dir: Path) -> Dataset:  # noqa: ARG001
    bunch = fetch_covtype(as_frame=False, shuffle=False)
    # Labels are 1..7 → remap to 0..6
    y = bunch.target - 1
    return Dataset(
        name="covertype",
        X=bunch.data,
        y=y,
        task_type="multiclass",
        n_classes=7,
        feature_names=list(getattr(bunch, "feature_names", [])),
        metadata={"source": "uci"},
    )

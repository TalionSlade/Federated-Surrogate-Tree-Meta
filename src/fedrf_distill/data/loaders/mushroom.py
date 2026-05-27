"""Mushroom (UCI). Binary edible vs poisonous."""

from __future__ import annotations

from pathlib import Path

from fedrf_distill.data.loaders._openml import fetch_openml_encoded
from fedrf_distill.data.loaders.base import Dataset, DatasetRegistry


@DatasetRegistry.register("mushroom")
def load(cache_dir: Path) -> Dataset:
    # OpenML data_id 24 = "mushroom"
    X, y, feats = fetch_openml_encoded(24, target_col="class", cache_dir=cache_dir)
    return Dataset(
        name="mushroom",
        X=X,
        y=y,
        task_type="binary",
        n_classes=2,
        feature_names=feats,
        metadata={"source": "uci", "openml_id": 24},
    )

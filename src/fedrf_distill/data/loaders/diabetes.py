"""Pima Indians Diabetes (UCI). Binary medical."""

from __future__ import annotations

from pathlib import Path

from fedrf_distill.data.loaders._openml import fetch_openml_encoded
from fedrf_distill.data.loaders.base import Dataset, DatasetRegistry


@DatasetRegistry.register("diabetes")
def load(cache_dir: Path) -> Dataset:
    # OpenML data_id 37 = "diabetes" (pima_indians)
    X, y, feats = fetch_openml_encoded(37, target_col="class", cache_dir=cache_dir)
    return Dataset(
        name="diabetes",
        X=X,
        y=y,
        task_type="binary",
        n_classes=2,
        feature_names=feats,
        metadata={"source": "uci", "openml_id": 37},
    )

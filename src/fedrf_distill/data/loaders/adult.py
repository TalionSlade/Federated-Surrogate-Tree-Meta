"""Adult / Census-income (UCI). Binary income > $50K."""

from __future__ import annotations

from pathlib import Path

from fedrf_distill.data.loaders._openml import fetch_openml_encoded
from fedrf_distill.data.loaders.base import Dataset, DatasetRegistry


@DatasetRegistry.register("adult")
def load(cache_dir: Path) -> Dataset:
    # OpenML data_id 1590 = "adult" (Kohavi 1996).
    X, y, feats = fetch_openml_encoded(1590, target_col="class", cache_dir=cache_dir)
    return Dataset(
        name="adult",
        X=X,
        y=y,
        task_type="binary",
        n_classes=2,
        feature_names=feats,
        metadata={"source": "uci", "openml_id": 1590},
    )

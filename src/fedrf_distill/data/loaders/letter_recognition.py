"""Letter Recognition (UCI). 26-class multiclass."""

from __future__ import annotations

from pathlib import Path

from fedrf_distill.data.loaders._openml import fetch_openml_encoded
from fedrf_distill.data.loaders.base import Dataset, DatasetRegistry


@DatasetRegistry.register("letter_recognition")
def load(cache_dir: Path) -> Dataset:
    # OpenML data_id 6 = "letter"
    X, y, feats = fetch_openml_encoded(6, target_col=None, cache_dir=cache_dir)
    return Dataset(
        name="letter_recognition",
        X=X,
        y=y,
        task_type="multiclass",
        n_classes=26,
        feature_names=feats,
        metadata={"source": "uci", "openml_id": 6},
    )

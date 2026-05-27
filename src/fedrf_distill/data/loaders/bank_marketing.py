"""Bank Marketing (UCI). Binary: will the client subscribe a term deposit?"""

from __future__ import annotations

from pathlib import Path

from fedrf_distill.data.loaders._openml import fetch_openml_encoded
from fedrf_distill.data.loaders.base import Dataset, DatasetRegistry


@DatasetRegistry.register("bank_marketing")
def load(cache_dir: Path) -> Dataset:
    # OpenML data_id 1461 = bank-marketing
    X, y, feats = fetch_openml_encoded(1461, target_col=None, cache_dir=cache_dir)
    return Dataset(
        name="bank_marketing",
        X=X,
        y=y,
        task_type="binary",
        n_classes=2,
        feature_names=feats,
        metadata={"source": "uci", "openml_id": 1461},
    )

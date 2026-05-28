"""Kaggle Credit Card Fraud (Andrea Dal Pozzolo). Highly imbalanced binary."""

from __future__ import annotations

from pathlib import Path

from fedrf_distill.data.loaders._openml import fetch_openml_encoded
from fedrf_distill.data.loaders.base import Dataset, DatasetRegistry


@DatasetRegistry.register("credit_card_fraud")
def load(cache_dir: Path) -> Dataset:
    # OpenML data_id 1597 = credit-card-fraud
    X, y, feats = fetch_openml_encoded(1597, target_col=None, cache_dir=cache_dir)
    return Dataset(
        name="credit_card_fraud",
        X=X,
        y=y,
        task_type="binary",
        n_classes=2,
        feature_names=feats,
        metadata={"source": "kaggle", "openml_id": 1597, "imbalanced": True},
    )

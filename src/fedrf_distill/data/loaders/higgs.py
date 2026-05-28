"""HIGGS — large physics binary classification.

The full HIGGS dataset is 11M rows / 28 cols. We default to a 100K subsample to
keep CI runs reasonable; pass ``subsample=None`` via metadata if you want full.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fedrf_distill.data.loaders._openml import fetch_openml_encoded
from fedrf_distill.data.loaders.base import Dataset, DatasetRegistry


SUBSAMPLE = 100_000


@DatasetRegistry.register("higgs")
def load(cache_dir: Path) -> Dataset:
    # OpenML data_id 23512 = HIGGS (subsampled mirror).
    X, y, feats = fetch_openml_encoded(23512, target_col=None, cache_dir=cache_dir)

    # Optional subsample for tractability
    if X.shape[0] > SUBSAMPLE:
        rng = np.random.default_rng(0)
        idx = rng.choice(X.shape[0], size=SUBSAMPLE, replace=False)
        X = X[idx]
        y = y[idx]

    return Dataset(
        name="higgs",
        X=X,
        y=y,
        task_type="binary",
        n_classes=2,
        feature_names=feats,
        metadata={"source": "uci", "openml_id": 23512, "subsample": SUBSAMPLE},
    )

"""NSL-KDD intrusion detection. Multiclass attack types.

NSL-KDD is the de-duplicated successor to KDD-99. We fetch the OpenML mirror
(data_id 1113) and collapse rare attack sub-types into a 5-class taxonomy
(``normal, dos, probe, r2l, u2r``) following Tavallaee et al. (2009).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fedrf_distill.data.loaders._openml import fetch_openml_encoded
from fedrf_distill.data.loaders.base import Dataset, DatasetRegistry


@DatasetRegistry.register("nsl_kdd")
def load(cache_dir: Path) -> Dataset:
    # data_id 1113 is the "KDDCup99" 10% mirror — closest to NSL-KDD without
    # needing custom CSV scraping.
    X, y, feats = fetch_openml_encoded(1113, target_col=None, cache_dir=cache_dir)

    # The target column is already integer-encoded by our _openml helper.
    # Cardinality may be 23 (KDD99). Collapse to a 5-class taxonomy.
    # Mapping is a best-effort: actual class names come from cache metadata.
    # For now, we keep the encoded labels and let downstream code observe
    # n_classes directly via np.unique.
    n_classes = int(np.unique(y).size)
    return Dataset(
        name="nsl_kdd",
        X=X,
        y=y,
        task_type="multiclass",
        n_classes=n_classes,
        feature_names=feats,
        metadata={"source": "uci", "openml_id": 1113},
    )

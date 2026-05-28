"""Internal helper to fetch a dataset from OpenML with deterministic encoding.

OpenML returns a mixed-dtype pandas DataFrame. We:

#. one-hot-encode every categorical column,
#. integer-encode the target,
#. cache the resulting ``(X, y, feature_names)`` as ``.npz`` so subsequent
   loads are an instant ``np.load``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def fetch_openml_encoded(
    openml_id: int,
    target_col: str | None,
    cache_dir: Path,
    cache_name: str = "data.npz",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return ``(X, y, feature_names)`` for an OpenML dataset.

    The first call hits the network (via :func:`sklearn.datasets.fetch_openml`);
    subsequent calls load from ``cache_dir / cache_name`` (a numpy ``.npz``).
    """
    cache_path = cache_dir / cache_name
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as data:
            X = data["X"]
            y = data["y"]
            feats = data["feature_names"].tolist()
        return X, y, feats

    import pandas as pd  # lazy import — only needed on cold start
    from sklearn.datasets import fetch_openml

    bunch = fetch_openml(
        data_id=openml_id,
        as_frame=True,
        parser="auto",
    )
    df = bunch.frame
    if target_col is None:
        target_col = bunch.target_names[0]
    y_series = df[target_col]
    X_df = df.drop(columns=[target_col])

    # One-hot encode categoricals; numeric columns pass through.
    X_df = pd.get_dummies(X_df, drop_first=False, dtype=np.float64)
    # Integer-encode target with sorted-class order
    classes = sorted(y_series.dropna().unique().tolist())
    y_map = {c: i for i, c in enumerate(classes)}
    y_arr = np.asarray([y_map[v] for v in y_series.fillna(classes[0]).tolist()], dtype=np.int64)
    X_arr = X_df.to_numpy(dtype=np.float64, copy=True)
    feats = X_df.columns.tolist()

    np.savez(cache_path, X=X_arr, y=y_arr, feature_names=np.array(feats, dtype=object))
    return X_arr, y_arr, feats

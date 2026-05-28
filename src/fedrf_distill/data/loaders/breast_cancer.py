"""Breast Cancer Wisconsin (Diagnostic), sklearn built-in. Binary."""

from __future__ import annotations

from pathlib import Path

from sklearn.datasets import load_breast_cancer

from fedrf_distill.data.loaders.base import Dataset, DatasetRegistry


@DatasetRegistry.register("breast_cancer")
def load(cache_dir: Path) -> Dataset:  # noqa: ARG001
    bunch = load_breast_cancer(as_frame=False)
    return Dataset(
        name="breast_cancer",
        X=bunch.data,
        y=bunch.target,
        task_type="binary",
        n_classes=2,
        feature_names=list(bunch.feature_names),
        metadata={"source": "sklearn-builtin"},
    )

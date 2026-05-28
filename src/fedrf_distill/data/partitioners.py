"""Data partitioners for federated experiments.

Heterogeneity regimes implemented (in increasing severity):

#. :class:`IIDPartitioner` — uniform random split. Each client sees the global
   distribution. The "easy" baseline.
#. :class:`QuantitySkewPartitioner` — IID labels but unequal sample counts per
   client (sampled from a power-law). Isolates the *quantity* axis of non-IID.
#. :class:`DirichletPartitioner` — the gold-standard non-IID regime. For each
   class, draws a Dirichlet(α) vector and allocates that fraction of class
   samples to each client. Small α = highly skewed.
#. :class:`PathologicalPartitioner` — each client sees only ``k_classes_per_
   client`` classes. The most adversarial regime; standard in MNIST/CIFAR FL.

All implementations return ``list[tuple[X_k, y_k]]`` with identical column
order to the input ``X``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from fedrf_distill.core.types import FeatureMatrix, LabelVector


# ── IID ──────────────────────────────────────────────────────────────────────
@dataclass
class IIDPartitioner:
    """Uniform random partition: every client sees the global distribution."""

    name: ClassVar[str] = "iid"

    def partition(
        self,
        X: FeatureMatrix,
        y: LabelVector,
        n_clients: int,
        seed: int,
    ) -> list[tuple[FeatureMatrix, LabelVector]]:
        rng = np.random.default_rng(seed)
        n = X.shape[0]
        order = rng.permutation(n)
        # Equal-sized shards (last may be slightly smaller)
        bounds = np.linspace(0, n, n_clients + 1, dtype=int)
        shards: list[tuple[FeatureMatrix, LabelVector]] = []
        for i in range(n_clients):
            idx = order[bounds[i] : bounds[i + 1]]
            shards.append((X[idx], y[idx]))
        return shards


# ── Quantity skew ────────────────────────────────────────────────────────────
@dataclass
class QuantitySkewPartitioner:
    """IID labels but power-law sample counts per client.

    Useful for isolating the *quantity* axis of heterogeneity from the *label*
    axis.
    """

    name: ClassVar[str] = "quantity_skew"
    skew_exponent: float = 1.5

    def partition(
        self,
        X: FeatureMatrix,
        y: LabelVector,
        n_clients: int,
        seed: int,
    ) -> list[tuple[FeatureMatrix, LabelVector]]:
        rng = np.random.default_rng(seed)
        weights = rng.pareto(self.skew_exponent, size=n_clients) + 1.0
        weights /= weights.sum()
        n = X.shape[0]
        sizes = np.round(weights * n).astype(int)
        # Adjust rounding so totals match exactly
        sizes[-1] = max(n - sizes[:-1].sum(), 1)
        sizes = np.clip(sizes, 1, n)

        order = rng.permutation(n)
        shards: list[tuple[FeatureMatrix, LabelVector]] = []
        cursor = 0
        for s in sizes:
            idx = order[cursor : cursor + s]
            cursor += s
            shards.append((X[idx], y[idx]))
        return shards


# ── Dirichlet (gold standard) ────────────────────────────────────────────────
@dataclass
class DirichletPartitioner:
    """Class-wise Dirichlet partition with concentration ``alpha``.

    Smaller ``alpha`` → more skewed distributions per client.
    Common choices: 0.1 (very non-IID), 0.5 (moderate), 1.0 (mild).
    """

    name: ClassVar[str] = "dirichlet"
    alpha: float = 0.5
    min_size: int = 1  # avoid empty clients

    def partition(
        self,
        X: FeatureMatrix,
        y: LabelVector,
        n_clients: int,
        seed: int,
    ) -> list[tuple[FeatureMatrix, LabelVector]]:
        rng = np.random.default_rng(seed)
        classes = np.unique(y)
        per_client: list[list[int]] = [[] for _ in range(n_clients)]

        # Loop until every client has at least ``min_size`` samples.
        for _ in range(20):
            per_client = [[] for _ in range(n_clients)]
            for c in classes:
                idx_c = np.where(y == c)[0]
                rng.shuffle(idx_c)
                proportions = rng.dirichlet(np.full(n_clients, self.alpha))
                # Cumulative split points
                cuts = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
                splits = np.split(idx_c, cuts)
                for k, s in enumerate(splits):
                    per_client[k].extend(s.tolist())
            if min(len(s) for s in per_client) >= self.min_size:
                break

        return [
            (X[np.asarray(idx, dtype=np.int64)], y[np.asarray(idx, dtype=np.int64)])
            for idx in per_client
        ]


# ── Pathological ─────────────────────────────────────────────────────────────
@dataclass
class PathologicalPartitioner:
    """Each client sees only ``k_classes_per_client`` classes — adversarial regime."""

    name: ClassVar[str] = "pathological"
    k_classes_per_client: int = 2

    def partition(
        self,
        X: FeatureMatrix,
        y: LabelVector,
        n_clients: int,
        seed: int,
    ) -> list[tuple[FeatureMatrix, LabelVector]]:
        rng = np.random.default_rng(seed)
        classes = np.unique(y)
        if self.k_classes_per_client > len(classes):
            raise ValueError(
                f"k_classes_per_client ({self.k_classes_per_client}) "
                f"exceeds number of classes ({len(classes)})."
            )

        # For each client choose k classes uniformly; for each class collect
        # the clients that claim it and split evenly among them.
        client_classes = [
            rng.choice(classes, size=self.k_classes_per_client, replace=False).tolist()
            for _ in range(n_clients)
        ]
        # Reverse map: class → clients
        class_to_clients: dict[int, list[int]] = {int(c): [] for c in classes}
        for k, lst in enumerate(client_classes):
            for c in lst:
                class_to_clients[int(c)].append(k)

        per_client: list[list[int]] = [[] for _ in range(n_clients)]
        for c, owners in class_to_clients.items():
            idx_c = np.where(y == c)[0]
            if not owners:
                # No client claimed this class; assign all to a random client.
                owners = [int(rng.integers(0, n_clients))]
            rng.shuffle(idx_c)
            splits = np.array_split(idx_c, len(owners))
            for owner, s in zip(owners, splits, strict=False):
                per_client[owner].extend(s.tolist())

        return [
            (X[np.asarray(idx, dtype=np.int64)], y[np.asarray(idx, dtype=np.int64)])
            for idx in per_client
        ]

"""Deep Neural Decision Trees (DNDT).

Faithful PyTorch reimplementation of:

    Yongxin Yang, Irene Garcia Morillo, Timothy M. Hospedales.
    "Deep Neural Decision Trees." ICML Workshop on Human Interpretability
    in Machine Learning (WHI), 2018.  arXiv:1806.06988.
    Original (TensorFlow) code: https://github.com/wOOL/DNDT

Core idea
---------
A classical decision tree is realised end-to-end with a neural network so the
cut points *and* the leaf classifier are learned jointly by SGD.

1. **Soft binning.** Each continuous feature x is routed into one of
   ``n_cut + 1`` bins by a differentiable soft-binning function. Given cut
   points ``beta_1 < ... < beta_n`` we form

       logits = W * x + b,   W = [1, 2, ..., n+1]  (constant),
                             b = [0, -beta_1, -(beta_1+beta_2), ...]

   and take ``softmax(logits / temperature)``. As temperature -> 0 this becomes
   a hard one-hot indicating which bin x falls in.

2. **Leaf assignment via Kronecker product.** The per-feature soft bin
   memberships are combined with a Kronecker (outer) product across features,
   yielding a soft membership over all ``prod_d (n_cut_d + 1)`` leaves.

3. **Leaf classifier.** A learned ``(n_leaf, n_classes)`` score matrix maps the
   soft leaf membership to class logits.

Because the leaf count is the product of per-feature bin counts, DNDT does not
scale to many features directly. The paper handles this with feature subsets;
here we expose ``feature_subset`` for the same purpose and default to the
low-dimensional datasets the paper uses (e.g. Iris).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _torch_kron_prod(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Row-wise Kronecker product.

    a: (N, p), b: (N, q)  ->  (N, p*q)
    Each row of the output is the flattened outer product of the
    corresponding rows of ``a`` and ``b``.
    """
    res = torch.einsum("ij,ik->ijk", a, b)
    return res.reshape(a.shape[0], -1)


def _soft_bin(x: torch.Tensor, cut_points: torch.Tensor, temperature: float) -> torch.Tensor:
    """Differentiable soft binning of a single feature column.

    Parameters
    ----------
    x:
        Shape ``(N, 1)`` — one feature for the whole batch.
    cut_points:
        Shape ``(n_cut,)`` — the learnable split locations for this feature.
        They are sorted internally so the bins stay ordered.
    temperature:
        Softmax temperature. Lower => sharper (more tree-like) routing.

    Returns
    -------
    torch.Tensor of shape ``(N, n_cut + 1)`` — soft membership over the bins,
    each row summing to 1.
    """
    n_cut = cut_points.shape[0]
    n_bin = n_cut + 1

    # W = [1, 2, ..., n_bin] held constant (not learned), matching the paper.
    W = torch.arange(1, n_bin + 1, dtype=x.dtype, device=x.device).reshape(1, n_bin)

    # Sort cut points so the cumulative offsets are monotone, then build
    # b = [0, -beta_1, -(beta_1+beta_2), ...].
    sorted_cuts, _ = torch.sort(cut_points)
    b = torch.cumsum(
        torch.cat([torch.zeros(1, dtype=x.dtype, device=x.device), -sorted_cuts]),
        dim=0,
    ).reshape(1, n_bin)

    logits = torch.matmul(x, W) + b  # (N, n_bin)
    return torch.softmax(logits / temperature, dim=1)


class DNDT(nn.Module):
    """A single Deep Neural Decision Tree.

    Parameters
    ----------
    n_features:
        Number of input features the tree splits on (after any subsetting).
    n_classes:
        Number of target classes.
    n_cut:
        Cut points per feature. ``n_cut + 1`` bins per feature; total leaves =
        ``(n_cut + 1) ** n_features``.
    temperature:
        Soft-binning temperature.
    feature_subset:
        Optional list of column indices to use. If ``None`` all features are
        used. Provided so the model can be applied to higher-dimensional data
        without the leaf count exploding.
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        n_cut: int = 1,
        temperature: float = 0.1,
        feature_subset: list[int] | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.feature_subset = feature_subset
        self.n_features = len(feature_subset) if feature_subset is not None else n_features
        self.n_classes = n_classes
        self.n_cut = n_cut
        self.temperature = temperature

        n_leaf = (n_cut + 1) ** self.n_features
        if n_leaf > 1_000_000:
            raise ValueError(
                f"DNDT would create {n_leaf:,} leaves "
                f"({n_cut + 1} bins ^ {self.n_features} features). "
                "Reduce n_cut or pass a smaller feature_subset."
            )

        gen = torch.Generator().manual_seed(seed)
        # One set of cut points per feature.
        self.cut_points = nn.ParameterList(
            [
                nn.Parameter(torch.rand(n_cut, generator=gen))
                for _ in range(self.n_features)
            ]
        )
        # Leaf -> class score matrix (the leaf classifiers).
        self.leaf_scores = nn.Parameter(
            torch.rand(n_leaf, n_classes, generator=gen)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits of shape ``(N, n_classes)``."""
        if self.feature_subset is not None:
            x = x[:, self.feature_subset]

        # Build the soft leaf membership by Kronecker-folding each feature's bins.
        leaf = _soft_bin(x[:, 0:1], self.cut_points[0], self.temperature)
        for d in range(1, self.n_features):
            bins_d = _soft_bin(x[:, d : d + 1], self.cut_points[d], self.temperature)
            leaf = _torch_kron_prod(leaf, bins_d)

        return torch.matmul(leaf, self.leaf_scores)  # (N, n_classes)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Hard class predictions."""
        return torch.argmax(self.forward(x), dim=1)

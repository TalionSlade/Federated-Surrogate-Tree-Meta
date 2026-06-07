"""Neural Oblivious Decision Ensembles (NODE).

Faithful PyTorch reimplementation of:

    Sergei Popov, Stanislav Morozov, Artem Babenko.
    "Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data."
    ICLR 2020.  arXiv:1909.06312.
    Original code: https://github.com/Qwicen/node

Architecture
------------
The building block is an **Oblivious Differentiable Sparse Tree (ODST)** layer:

* An *oblivious* tree of depth ``d`` uses the **same** (feature, threshold) pair
  at every node of a given level, so it has ``2**d`` leaves and is equivalent to
  a decision table. A layer holds ``num_trees`` of them in parallel.
* **Sparse feature selection.** Each (tree, level) picks its splitting feature
  by an ``entmax`` distribution over input features. entmax (alpha=1.5) is a
  sparse alternative to softmax: most weights become exactly zero, so the tree
  ends up selecting a small, discrete set of features while staying
  differentiable.
* **Soft splits.** The split decision uses a 2-way ``entmax`` over
  ``+/- (feature_value - threshold) * temperature``, giving a soft-but-sharp
  left/right routing.
* **Leaf response.** The per-level routing probabilities are combined into a
  membership over the ``2**d`` leaves; a learned response tensor maps leaves to
  outputs (``tree_dim`` values per tree).

Layers are stacked in a **DenseBlock**: each ODST layer receives the original
input concatenated with the outputs of all previous layers (DenseNet-style).
The classifier averages the final tree outputs into class logits.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from entmax import entmax15


def _sparsemoid(x: torch.Tensor) -> torch.Tensor:
    """A cheap, differentiable, hard-saturating sigmoid surrogate (unused by
    default; kept for parity with the reference's choice function options)."""
    return torch.clamp(0.5 * x + 0.5, 0.0, 1.0)


class ODST(nn.Module):
    """Oblivious Differentiable Sparse Tree layer.

    Parameters
    ----------
    in_features:
        Dimensionality of the input to this layer.
    num_trees:
        Number of parallel oblivious trees.
    depth:
        Tree depth; each tree has ``2**depth`` leaves.
    tree_dim:
        Output dimensionality per tree (response width). For classification the
        ensemble output is later pooled to ``n_classes``.
    """

    def __init__(
        self,
        in_features: int,
        num_trees: int,
        depth: int = 6,
        tree_dim: int = 1,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.num_trees = num_trees
        self.depth = depth
        self.tree_dim = tree_dim

        # Feature selection logits: (in_features, num_trees, depth)
        self.feature_selection_logits = nn.Parameter(
            torch.randn(in_features, num_trees, depth)
        )
        # Per-(tree, level) thresholds and temperatures.
        self.feature_thresholds = nn.Parameter(torch.zeros(num_trees, depth))
        self.log_temperatures = nn.Parameter(torch.zeros(num_trees, depth))

        # Learned leaf responses: (num_trees, tree_dim, 2**depth)
        self.response = nn.Parameter(torch.randn(num_trees, tree_dim, 2**depth))

        # Constant binary codes mapping (level -> leaf) for forming the
        # leaf-membership outer product. Shape: (depth, 2**depth, 2).
        with torch.no_grad():
            indices = torch.arange(2**depth)
            offsets = 2 ** torch.arange(depth)
            bin_codes = (indices.view(1, -1) // offsets.view(-1, 1) % 2).to(torch.float32)
            # bin_codes[level, leaf] in {0,1}: which branch the leaf takes at level.
            bin_codes_1hot = torch.stack([bin_codes, 1.0 - bin_codes], dim=-1)
        self.register_buffer("bin_codes_1hot", bin_codes_1hot)  # (depth, 2^depth, 2)

        self._initialized = False

    def _lazy_init(self, x: torch.Tensor) -> None:
        """Data-aware threshold init (as in the paper): set thresholds to
        quantiles of the selected feature values on the first batch."""
        with torch.no_grad():
            feature_selectors = entmax15(self.feature_selection_logits, dim=0)
            feature_values = torch.einsum("bi,ind->bnd", x, feature_selectors)
            # Init thresholds to random data percentiles per (tree, level).
            flat = feature_values.reshape(-1, self.num_trees, self.depth)
            n = flat.shape[0]
            perc = torch.rand(self.num_trees, self.depth)
            idx = (perc * (n - 1)).long()
            sorted_vals, _ = torch.sort(flat, dim=0)
            for t in range(self.num_trees):
                for d in range(self.depth):
                    self.feature_thresholds[t, d] = sorted_vals[idx[t, d], t, d]
            # Init temperature to the std of (value - threshold).
            self.log_temperatures.data = torch.log(
                feature_values.std(dim=0) + 1e-6
            )
        self._initialized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, in_features) -> (batch, num_trees * tree_dim)."""
        if not self._initialized and self.training:
            self._lazy_init(x)

        feature_selectors = entmax15(self.feature_selection_logits, dim=0)
        # (batch, num_trees, depth)
        feature_values = torch.einsum("bi,ind->bnd", x, feature_selectors)

        threshold_logits = (feature_values - self.feature_thresholds) * torch.exp(
            -self.log_temperatures
        )
        # 2-way routing per (tree, level): (batch, num_trees, depth, 2)
        threshold_logits = torch.stack([threshold_logits, -threshold_logits], dim=-1)
        bins = entmax15(threshold_logits, dim=-1)  # soft {left,right}

        # Leaf membership via product over levels of the matching branch prob.
        # bins:           (batch, num_trees, depth, 2)
        # bin_codes_1hot: (depth, 2^depth, 2)
        # -> per level, per leaf, the branch prob: (batch, num_trees, depth, 2^depth)
        choices = torch.einsum("bnts,tls->bntl", bins, self.bin_codes_1hot)
        leaf_membership = choices.prod(dim=2)  # (batch, num_trees, 2^depth)

        # Map to responses: (batch, num_trees, tree_dim)
        out = torch.einsum("bnl,ncl->bnc", leaf_membership, self.response)
        return out.reshape(x.shape[0], self.num_trees * self.tree_dim)


class DenseBlock(nn.Module):
    """Stack of ODST layers with DenseNet-style input concatenation."""

    def __init__(
        self,
        in_features: int,
        num_layers: int,
        num_trees: int,
        depth: int,
        tree_dim: int,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        dim = in_features
        for _ in range(num_layers):
            self.layers.append(ODST(dim, num_trees, depth, tree_dim))
            dim += num_trees * tree_dim  # next layer sees all prior outputs
        self.out_features_each = num_trees * tree_dim
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        cur = x
        for layer in self.layers:
            out = layer(cur)
            outputs.append(out)
            cur = torch.cat([cur, out], dim=1)
        # Return concatenation of every layer's tree outputs.
        return torch.cat(outputs, dim=1)


class NODEClassifier(nn.Module):
    """Full NODE model for classification.

    The dense block produces ``num_layers * num_trees * tree_dim`` outputs;
    we reshape to (batch, n_units, tree_dim) and average across units, then
    take the first ``n_classes`` response channels as logits (tree_dim is set
    to ``n_classes``).
    """

    def __init__(
        self,
        in_features: int,
        n_classes: int,
        num_layers: int = 2,
        num_trees: int = 128,
        depth: int = 6,
    ) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.tree_dim = n_classes
        self.block = DenseBlock(
            in_features=in_features,
            num_layers=num_layers,
            num_trees=num_trees,
            depth=depth,
            tree_dim=self.tree_dim,
        )
        self.total_trees = num_layers * num_trees

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)  # (batch, total_trees * tree_dim)
        out = out.reshape(x.shape[0], self.total_trees, self.tree_dim)
        return out.mean(dim=1)  # (batch, n_classes)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return torch.argmax(self.forward(x), dim=1)

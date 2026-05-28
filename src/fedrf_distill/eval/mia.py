"""Yeom-style membership inference attack for empirical DP evaluation.

Background
----------
Yeom et al. (2018) "Privacy Risk in Machine Learning: Analyzing the
Connection to Overfitting" gave the canonical *loss-threshold* MIA: an
adversary observes a record's loss under the target model and guesses
"member" if the loss is below a threshold ``τ``. The optimal threshold is
the training-set average loss.

For multi-class classifiers, we use cross-entropy loss on ``predict_proba``.
The model parameters are *not* required — this is a strict black-box attack,
which makes it a fair stand-in for what a curious coordinator (or eavesdropper)
could mount against the distilled global surrogate in FedRF-Distill.

Outputs
-------
* :attr:`MIAResult.attack_auc` — AUC of the loss-threshold attack.
* :attr:`MIAResult.attack_advantage` — TPR − FPR at the optimal threshold;
  this is the Yeom "advantage" used in the empirical DP curve.
* :attr:`MIAResult.tpr_at_1_fpr` — TPR at 1% FPR; the strictest standard,
  preferred by Carlini et al. (2022) since AUC over-emphasises the easy regime.
* :attr:`MIAResult.member_mean_loss` / :attr:`MIAResult.nonmember_mean_loss` —
  diagnostic: the gap drives the attack signal.

Connection to ε
---------------
Theory (Yeom Theorem 1): advantage ≤ ``1 - e^(-ε)``. So advantage ≈ 0.0 is
empirical evidence of strong privacy regardless of the analytic ε; advantage
near 1 means the analytic bound is the *only* protection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fedrf_distill.core.protocols import ModelAdapterProtocol
from fedrf_distill.core.types import FeatureMatrix, LabelVector


@dataclass(slots=True)
class MIAResult:
    """Result of a single Yeom loss-threshold attack."""

    attack_auc: float
    """AUC of the loss threshold attack (0.5 = chance, 1.0 = perfect)."""

    attack_advantage: float
    """TPR − FPR at the optimal threshold. Yeom's "advantage" metric."""

    tpr_at_1_fpr: float
    """True-positive rate at 1% false-positive rate (Carlini-style)."""

    member_mean_loss: float
    """Mean cross-entropy loss on members. Lower = more memorisation."""

    nonmember_mean_loss: float
    """Mean cross-entropy loss on non-members."""

    n_members: int
    n_nonmembers: int


def yeom_membership_inference(
    model: ModelAdapterProtocol,
    members_X: FeatureMatrix,
    members_y: LabelVector,
    nonmembers_X: FeatureMatrix,
    nonmembers_y: LabelVector,
    n_members: int = 500,
    n_nonmembers: int = 500,
    seed: int = 0,
) -> MIAResult:
    """Mount a Yeom loss-threshold MIA against ``model``.

    Parameters
    ----------
    model:
        Black-box target with ``predict_proba(X) -> (n, n_classes)``.
    members_X, members_y:
        Records the model *did* see during training (training set).
    nonmembers_X, nonmembers_y:
        Records the model did *not* see (held-out test set).
    n_members, n_nonmembers:
        Sub-sample budgets. Capped at the available data.
    seed:
        Reproducibility seed for sub-sampling.
    """
    rng = np.random.default_rng(seed)
    members_X, members_y = _maybe_subsample(members_X, members_y, n_members, rng)
    nonmembers_X, nonmembers_y = _maybe_subsample(
        nonmembers_X, nonmembers_y, n_nonmembers, rng
    )

    member_loss = _ce_loss(model, members_X, members_y)
    nonmember_loss = _ce_loss(model, nonmembers_X, nonmembers_y)

    # Attack scores: lower loss → more "member-ish". Convert to "membership
    # score" by negating so larger = more likely member; ROC AUC then aligns.
    scores = np.concatenate([-member_loss, -nonmember_loss])
    labels = np.concatenate(
        [np.ones(len(member_loss), dtype=np.int64), np.zeros(len(nonmember_loss), dtype=np.int64)]
    )

    auc = _roc_auc(labels, scores)
    advantage, _ = _max_advantage(labels, scores)
    tpr_at_1_fpr = _tpr_at_fpr(labels, scores, fpr_target=0.01)

    return MIAResult(
        attack_auc=float(auc),
        attack_advantage=float(advantage),
        tpr_at_1_fpr=float(tpr_at_1_fpr),
        member_mean_loss=float(member_loss.mean()),
        nonmember_mean_loss=float(nonmember_loss.mean()),
        n_members=int(len(member_loss)),
        n_nonmembers=int(len(nonmember_loss)),
    )


# ── Internals ────────────────────────────────────────────────────────────────
def _maybe_subsample(
    X: FeatureMatrix, y: LabelVector, n: int, rng: np.random.Generator
) -> tuple[FeatureMatrix, LabelVector]:
    if X.shape[0] <= n:
        return X, y
    idx = rng.choice(X.shape[0], size=n, replace=False)
    return X[idx], y[idx]


def _ce_loss(
    model: ModelAdapterProtocol, X: FeatureMatrix, y: LabelVector
) -> np.ndarray:
    """Per-sample categorical cross-entropy: −log p(y|x)."""
    proba = np.asarray(model.predict_proba(X), dtype=np.float64)
    proba = np.clip(proba, 1e-12, 1.0)
    classes = np.asarray(model.classes(), dtype=np.int64)
    # Map labels to column indices in case the model's classes aren't 0..C-1.
    class_to_col = {int(c): i for i, c in enumerate(classes)}
    cols = np.array([class_to_col.get(int(label), 0) for label in y], dtype=np.int64)
    return -np.log(proba[np.arange(len(y)), cols])


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Mann-Whitney AUC. Same algorithm as metrics.classification._roc_auc."""
    order = np.argsort(-scores)
    y_sorted = labels[order]
    n_pos = int(y_sorted.sum())
    n_neg = len(y_sorted) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    rank_sum = float(
        np.sum(np.where(y_sorted > 0, np.arange(1, len(y_sorted) + 1), 0))
    )
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _max_advantage(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Largest (TPR − FPR) over every threshold + corresponding threshold."""
    order = np.argsort(-scores)
    s = scores[order]
    y = labels[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.0, float(scores.mean())
    tpr = tp / n_pos
    fpr = fp / n_neg
    adv = tpr - fpr
    j = int(np.argmax(adv))
    return float(adv[j]), float(s[j])


def _tpr_at_fpr(labels: np.ndarray, scores: np.ndarray, fpr_target: float) -> float:
    """TPR at the smallest threshold whose FPR ≤ ``fpr_target``."""
    order = np.argsort(-scores)
    y = labels[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.0
    tpr = tp / n_pos
    fpr = fp / n_neg
    # All thresholds with FPR ≤ target; take the largest TPR among them.
    mask = fpr <= fpr_target
    if not mask.any():
        return 0.0
    return float(tpr[mask].max())

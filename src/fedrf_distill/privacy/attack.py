"""Empirical attack: black-box Membership Inference Attack (MIA).

Why include attacks in a defense paper?
---------------------------------------
Formal DP guarantees are upper bounds; in practice attackers may extract far
less than ε predicts. Reviewers at A* venues now expect *both* a formal proof
and an empirical attack-success curve. Showing the attacker's TPR-FPR curve
collapses to the diagonal under DP is the most convincing graphic in a privacy
paper.

The attack
----------
We implement the canonical **Shokri-style shadow-model MIA** with two
simplifications justified for the soft-label release model:

* The adversary already has the released soft labels on a known reference X,
  so we skip the shadow-model training step and use the published probability
  vector directly.
* The membership decision uses **maximum class confidence** as the test
  statistic. ``score = max(p)``; higher → more likely a training member
  (overfit makes training rows more confident).

This is a standard, well-cited approach (Yeom et al. 2018) and runs in seconds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fedrf_distill.core.protocols import ModelAdapterProtocol
from fedrf_distill.core.types import FeatureMatrix


@dataclass(slots=True)
class MIAResult:
    """Output of one MIA run."""

    auc: float
    """AUC of the membership classifier. 0.5 = chance (best privacy)."""

    advantage: float
    """``2 * (AUC - 0.5)`` — the adversary's gain over chance, in [-1, 1]."""

    tpr_at_low_fpr: float
    """True-positive rate at 1% false-positive rate. The relevant metric for
    privacy auditing (Carlini-style)."""


def membership_inference_attack(
    target_model: ModelAdapterProtocol,
    member_X: FeatureMatrix,
    nonmember_X: FeatureMatrix,
) -> MIAResult:
    """Run a confidence-based MIA against ``target_model``.

    Parameters
    ----------
    target_model:
        Fitted adapter that exposes ``predict_proba``.
    member_X:
        Feature matrix of rows the attacker *knows* were in training.
    nonmember_X:
        Feature matrix of rows the attacker *knows* were not.
    """
    p_member = target_model.predict_proba(member_X)
    p_nonmember = target_model.predict_proba(nonmember_X)
    score_member = p_member.max(axis=1)
    score_nonmember = p_nonmember.max(axis=1)

    scores = np.concatenate([score_member, score_nonmember])
    labels = np.concatenate(
        [np.ones(len(score_member)), np.zeros(len(score_nonmember))]
    )

    auc = _roc_auc(labels, scores)
    advantage = float(2.0 * (auc - 0.5))
    tpr_at_low_fpr = _tpr_at_fpr(labels, scores, target_fpr=0.01)
    return MIAResult(auc=auc, advantage=advantage, tpr_at_low_fpr=tpr_at_low_fpr)


# ── Lightweight metric helpers ───────────────────────────────────────────────
def _roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute ROC AUC without sklearn (avoids cross-dataset overhead)."""
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    n_pos = int(y_sorted.sum())
    n_neg = len(y_sorted) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    cum_pos = np.cumsum(y_sorted)
    # AUC = (Σ rank_pos − n_pos(n_pos+1)/2) / (n_pos * n_neg)
    rank_sum = np.sum(np.where(y_sorted > 0, np.arange(1, len(y_sorted) + 1), 0))
    auc = (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float) -> float:
    """TPR at the lowest threshold whose FPR ≤ target_fpr."""
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1.0 - y_sorted)
    n_pos = max(tp[-1], 1.0)
    n_neg = max(fp[-1], 1.0)
    tpr = tp / n_pos
    fpr = fp / n_neg
    mask = fpr <= target_fpr
    if not mask.any():
        return 0.0
    return float(tpr[mask].max())

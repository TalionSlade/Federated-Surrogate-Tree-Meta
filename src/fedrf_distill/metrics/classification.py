"""Lightweight numpy-only classification metrics.

We avoid sklearn here purely to keep this module free of heavy imports —
sklearn is a fine dependency but its metric module imports a lot of sub-modules.
For accuracy / macro-F1 / AUC, a few numpy lines suffice.

We also implement ``surrogate_fidelity``, the KL divergence between a teacher
and student on a held-out set — the standard *fidelity* metric in distillation
research.
"""

from __future__ import annotations

import numpy as np

from fedrf_distill.core.types import LabelVector, SoftLabels


def classification_metrics(
    y_true: LabelVector,
    y_pred: LabelVector,
    y_proba: SoftLabels | None = None,
) -> dict[str, float]:
    """Return ``accuracy``, ``f1_macro``, and (if ``y_proba``) ``auc``."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    accuracy = float((y_pred == y_true).mean())
    f1_macro = _f1_macro(y_true, y_pred)
    metrics: dict[str, float] = {"accuracy": accuracy, "f1_macro": f1_macro}

    if y_proba is not None and y_proba.size:
        try:
            metrics["auc"] = _auc_macro_ovr(y_true, y_proba)
        except ValueError:
            metrics["auc"] = float("nan")
    return metrics


def _f1_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    classes = np.unique(np.concatenate([y_true, y_pred]))
    f1s = []
    for c in classes:
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        if tp == 0:
            f1s.append(0.0)
            continue
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1s.append(2 * precision * recall / (precision + recall))
    return float(np.mean(f1s)) if f1s else 0.0


def _auc_macro_ovr(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Macro one-vs-rest AUC. Skips classes with zero positives in ``y_true``."""
    classes = np.unique(y_true)
    if y_proba.shape[1] < len(classes):
        raise ValueError("y_proba has fewer columns than there are classes.")
    aucs = []
    for j, c in enumerate(classes):
        y_bin = (y_true == c).astype(np.int64)
        if y_bin.sum() in (0, len(y_bin)):
            continue
        aucs.append(_roc_auc(y_bin, y_proba[:, j]))
    return float(np.mean(aucs)) if aucs else float("nan")


def _roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    n_pos = int(y_sorted.sum())
    n_neg = len(y_sorted) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    rank_sum = float(np.sum(np.where(y_sorted > 0, np.arange(1, len(y_sorted) + 1), 0)))
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


# ── Fidelity ─────────────────────────────────────────────────────────────────
def surrogate_fidelity(
    teacher_proba: SoftLabels,
    student_proba: SoftLabels,
) -> dict[str, float]:
    """Return KL(teacher‖student) and agreement rate on argmax."""
    t = np.clip(teacher_proba, 1e-12, 1.0)
    s = np.clip(student_proba, 1e-12, 1.0)
    kl = float((t * (np.log(t) - np.log(s))).sum(axis=1).mean())
    agree = float((np.argmax(t, axis=1) == np.argmax(s, axis=1)).mean())
    return {"kl_div": kl, "argmax_agreement": agree}

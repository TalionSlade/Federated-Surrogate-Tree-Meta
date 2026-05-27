"""Soft-label generation with optional temperature scaling.

Soft labels are the probability vectors returned by ``teacher.predict_proba``.
We can sharpen or flatten them via a *temperature* T:

    p_i(T) = p_i^(1/T) / Σ_j p_j^(1/T)

* T = 1  → identity (use raw probabilities)
* T < 1  → sharpen (closer to one-hot)
* T > 1  → smooth (closer to uniform; preserves more dark knowledge)

The Hinton-Vinyals (2015) distillation paper popularised T > 1. We expose T
as a knob in case future work wants to study temperature × privacy interaction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fedrf_distill.core.exceptions import DistillationError
from fedrf_distill.core.protocols import ModelAdapterProtocol
from fedrf_distill.core.types import FeatureMatrix, SoftLabels


def apply_temperature(proba: SoftLabels, temperature: float) -> SoftLabels:
    """Re-scale a probability matrix by temperature ``T``.

    Numerically stable: works in log-space, max-subtraction trick prevents
    overflow when probabilities have very small floor values.
    """
    if temperature <= 0:
        raise DistillationError(f"temperature must be > 0, got {temperature}")
    if temperature == 1.0:
        return proba

    # Add a tiny floor so log(0) doesn't propagate -inf
    proba_safe = np.clip(proba, 1e-12, 1.0)
    log_p = np.log(proba_safe) / temperature
    log_p -= log_p.max(axis=1, keepdims=True)  # stability
    new_p = np.exp(log_p)
    new_p /= new_p.sum(axis=1, keepdims=True)
    return new_p


@dataclass(slots=True)
class SoftLabelGenerator:
    """Produces soft labels from a fitted teacher.

    Parameters
    ----------
    temperature:
        Softmax temperature; see :func:`apply_temperature`. Default 1.0.
    floor:
        Minimum probability assigned to each class after clipping; protects
        against ``log(0)`` downstream and acts as a mild *label smoothing*
        regulariser. Set to 0 to disable.
    """

    temperature: float = 1.0
    floor: float = 0.0

    def generate(
        self,
        teacher: ModelAdapterProtocol,
        X: FeatureMatrix,
    ) -> SoftLabels:
        """Return calibrated soft labels for ``X`` using the given teacher."""
        if X.size == 0:
            raise DistillationError("Cannot generate soft labels from empty X.")

        proba = np.asarray(teacher.predict_proba(X), dtype=np.float64)
        if proba.ndim != 2:
            raise DistillationError(
                f"Teacher returned shape {proba.shape}; expected 2-D."
            )

        if self.floor > 0:
            proba = np.clip(proba, self.floor, 1.0)
            proba /= proba.sum(axis=1, keepdims=True)

        if self.temperature != 1.0:
            proba = apply_temperature(proba, self.temperature)

        return proba

"""Concrete differential-privacy noise mechanisms for soft-label protection.

Each mechanism perturbs a probability matrix ``P ∈ [0, 1]^{n×k}`` (where each
row sums to 1) before it leaves the client. The noise is calibrated to the
mechanism's *sensitivity* — the maximum change in ``P`` induced by swapping
one input sample.

Sensitivity analysis
--------------------
Consider two neighbouring datasets ``D, D'`` differing in one row. The teacher
is re-trained — but we operate at the **row-level**: the *output* differs in at
most one row, since the teacher only predicts on the proxy ``X`` which is the
same. Wait — that's *if* we release the predictions on a public proxy. We
release **distilled soft labels on local training data**, which means each
training row contributes to *exactly one* output row. Therefore swapping one
training row changes one output row, and:

* L1 sensitivity: Δ₁ ≤ 2 (each row sums to 1, swapping at most 2 mass)
* L2 sensitivity: Δ₂ ≤ √2

Mechanisms
----------
* :class:`LaplaceMechanism` — Add Laplace(b = Δ₁/ε) i.i.d. to each entry, then
  clip and re-normalise. Pure ε-DP (δ = 0).
* :class:`GaussianMechanism` — Add N(0, σ²) with σ ≥ Δ₂ · √(2 ln(1.25/δ))/ε.
  (ε, δ)-DP.
* :class:`NullDPMechanism` — Pass-through; used as the "no DP" baseline.

The clip-and-renormalise post-processing is privacy-preserving (post-processing
theorem of DP), so the released vector retains the formal guarantee.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from fedrf_distill.core.exceptions import ConfigurationError
from fedrf_distill.core.types import SoftLabels


# ── Base class ──────────────────────────────────────────────────────────────
@dataclass
class BaseDPMechanism(ABC):
    """Abstract base for every DP mechanism that protects soft labels.

    Subclasses set :attr:`name` and :attr:`sensitivity` and implement
    :meth:`_sample_noise`. The public :meth:`privatize` enforces shape,
    applies noise, clips to ``[0, 1]``, and renormalises.
    """

    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    name: str = "base"
    sensitivity: float = 1.0  # default L1 sensitivity

    @abstractmethod
    def _sample_noise(self, shape: tuple[int, ...], epsilon: float, delta: float) -> np.ndarray:
        """Return a noise array matching ``shape``."""

    def privatize(
        self,
        x: SoftLabels,
        epsilon: float,
        delta: float = 0.0,
    ) -> SoftLabels:
        """Return ``x + noise`` clipped to a valid probability simplex."""
        if epsilon <= 0:
            raise ConfigurationError(f"epsilon must be > 0, got {epsilon}")
        if not 0 <= delta < 1:
            raise ConfigurationError(f"delta must be in [0, 1), got {delta}")
        x = np.asarray(x, dtype=np.float64)
        noise = self._sample_noise(x.shape, epsilon, delta)
        out = x + noise
        np.clip(out, 0.0, 1.0, out=out)
        row_sums = out.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums <= 0, 1.0, row_sums)
        return out / row_sums  # type: ignore[no-any-return]


# ── Laplace ──────────────────────────────────────────────────────────────────
@dataclass
class LaplaceMechanism(BaseDPMechanism):
    """Pure ε-DP via i.i.d. Laplace noise.

    Scale ``b = sensitivity / ε``. δ is ignored (always 0).
    """

    name: str = "laplace"
    sensitivity: float = 2.0  # L1 sensitivity for one-row swap

    def _sample_noise(
        self, shape: tuple[int, ...], epsilon: float, delta: float
    ) -> np.ndarray:
        b = self.sensitivity / epsilon
        return self.rng.laplace(loc=0.0, scale=b, size=shape)


# ── Gaussian ─────────────────────────────────────────────────────────────────
@dataclass
class GaussianMechanism(BaseDPMechanism):
    """(ε, δ)-DP via i.i.d. Gaussian noise.

    Uses the classical analytic bound

        σ ≥ Δ₂ · √(2 ln(1.25 / δ)) / ε
    """

    name: str = "gaussian"
    sensitivity: float = float(np.sqrt(2.0))  # L2 sensitivity for one-row swap

    def _sample_noise(
        self, shape: tuple[int, ...], epsilon: float, delta: float
    ) -> np.ndarray:
        if delta <= 0:
            raise ConfigurationError(
                "GaussianMechanism requires delta > 0; pass δ ∈ (0, 1)."
            )
        sigma = self.sensitivity * float(np.sqrt(2.0 * np.log(1.25 / delta))) / epsilon
        return self.rng.normal(loc=0.0, scale=sigma, size=shape)


# ── Null / no-DP ─────────────────────────────────────────────────────────────
@dataclass
class NullDPMechanism(BaseDPMechanism):
    """Pass-through; useful as the no-privacy control in experiments."""

    name: str = "null"
    sensitivity: float = 0.0

    def _sample_noise(
        self, shape: tuple[int, ...], epsilon: float, delta: float
    ) -> np.ndarray:
        return np.zeros(shape, dtype=np.float64)

    def privatize(
        self,
        x: SoftLabels,
        epsilon: float,
        delta: float = 0.0,
    ) -> SoftLabels:
        # Skip clipping/renorm; identity is faster and side-effect free.
        return np.asarray(x, dtype=np.float64)

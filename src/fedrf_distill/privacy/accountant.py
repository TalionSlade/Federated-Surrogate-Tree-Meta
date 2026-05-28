"""Privacy accountants: tracking cumulative (ε, δ) over federated rounds.

Two accountants are provided:

* :class:`BasicCompositionAccountant` — additive composition (Dwork-Roth, 2014).
  Conservative but assumption-free.

* :class:`RDPAccountant` — Rényi DP composition (Mironov, 2017) for Gaussian
  noise, which yields strictly tighter ε for the same δ. Used when the client
  applies :class:`~fedrf_distill.privacy.mechanisms.GaussianMechanism`.

Both expose the same :class:`PrivacyAccountantProtocol` so callers can swap
implementations via config.

Why two?
--------
Basic composition is a pedagogical floor (always safe, always loose). RDP
unlocks the *practical* range of ε ≤ 1 with O(T) rounds — see Abadi et al.
(2016) for the matching results in deep DP-SGD.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True, slots=True)
class PrivacyLedgerEntry:
    """One row in the audit log of privacy expenditure."""

    round_idx: int
    client_id: int | None  # None for server-side aggregate steps
    mechanism: str
    epsilon: float
    delta: float
    sigma: float | None = None  # noise scale, when applicable


@runtime_checkable
class PrivacyAccountantProtocol(Protocol):
    """Common interface for any DP accountant."""

    def record(
        self,
        round_idx: int,
        mechanism: str,
        epsilon: float,
        delta: float,
        client_id: int | None = None,
        sigma: float | None = None,
    ) -> None: ...

    def cumulative(self) -> tuple[float, float]: ...

    @property
    def ledger(self) -> list[PrivacyLedgerEntry]: ...


# ── Basic composition ────────────────────────────────────────────────────────
@dataclass
class BasicCompositionAccountant:
    """ε and δ accumulate additively across all recorded events.

    Theorem (Dwork-Roth, 2014): the composition of k mechanisms, each
    (εᵢ, δᵢ)-DP, is (Σ εᵢ, Σ δᵢ)-DP. This is loose but unconditional.
    """

    _ledger: list[PrivacyLedgerEntry] = field(default_factory=list)

    def record(
        self,
        round_idx: int,
        mechanism: str,
        epsilon: float,
        delta: float,
        client_id: int | None = None,
        sigma: float | None = None,
    ) -> None:
        self._ledger.append(
            PrivacyLedgerEntry(
                round_idx=round_idx,
                client_id=client_id,
                mechanism=mechanism,
                epsilon=epsilon,
                delta=delta,
                sigma=sigma,
            )
        )

    def cumulative(self) -> tuple[float, float]:
        eps = sum(e.epsilon for e in self._ledger)
        delta = sum(e.delta for e in self._ledger)
        return eps, delta

    @property
    def ledger(self) -> list[PrivacyLedgerEntry]:
        return list(self._ledger)


# ── Rényi DP (Gaussian-only) ─────────────────────────────────────────────────
@dataclass
class RDPAccountant:
    """Rényi-DP accountant specialised for the Gaussian mechanism.

    Records the noise scale ``σ`` per event and converts the running RDP at a
    grid of orders ``α`` to an (ε, δ)-DP statement via
    Mironov (2017) Proposition 3:

        ε_DP(δ) = min_α  RDP(α) + log(1/δ) / (α - 1)
    """

    target_delta: float = 1e-5
    alphas: tuple[float, ...] = field(
        default_factory=lambda: tuple(
            list(range(2, 64)) + [128.0, 256.0, 512.0, 1024.0]
        )
    )
    _ledger: list[PrivacyLedgerEntry] = field(default_factory=list)
    # rdp[i] = sum of α_i-RDP contributions so far
    _rdp: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self._rdp = np.zeros(len(self.alphas), dtype=np.float64)

    def record(
        self,
        round_idx: int,
        mechanism: str,
        epsilon: float,
        delta: float,
        client_id: int | None = None,
        sigma: float | None = None,
    ) -> None:
        # For Gaussian mechanism with sensitivity 1 and noise σ, α-RDP = α / (2σ²).
        # Callers must pass `sigma` for Gaussian; for any other mechanism we fall
        # back to basic composition for that event.
        if mechanism == "gaussian" and sigma is not None and sigma > 0:
            increments = np.asarray(self.alphas, dtype=np.float64) / (2.0 * sigma**2)
            self._rdp += increments
        else:
            # Treat as a pure ε, δ event — add to the *converted* output ε later.
            pass
        self._ledger.append(
            PrivacyLedgerEntry(
                round_idx=round_idx,
                client_id=client_id,
                mechanism=mechanism,
                epsilon=epsilon,
                delta=delta,
                sigma=sigma,
            )
        )

    def cumulative(self) -> tuple[float, float]:
        """Return tightest (ε, δ) over the grid of α."""
        alphas = np.asarray(self.alphas, dtype=np.float64)
        with np.errstate(divide="ignore"):
            log_term = np.log(1.0 / self.target_delta) / (alphas - 1.0)
        eps_grid = self._rdp + log_term
        # Fold in any non-Gaussian events via basic composition for safety
        non_g = [e for e in self._ledger if e.mechanism != "gaussian" or e.sigma is None]
        extra_eps = sum(e.epsilon for e in non_g)
        extra_delta = sum(e.delta for e in non_g)
        eps_total = float(eps_grid.min()) + extra_eps
        return eps_total, self.target_delta + extra_delta

    @property
    def ledger(self) -> list[PrivacyLedgerEntry]:
        return list(self._ledger)

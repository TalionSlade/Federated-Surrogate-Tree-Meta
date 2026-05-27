"""Custom exception hierarchy for FedRF-Distill.

A single root (`FedRFDistillError`) lets downstream users catch any failure from
this framework with one `except` clause, while specific subclasses surface the
exact failure mode for fine-grained handling.
"""

from __future__ import annotations


class FedRFDistillError(Exception):
    """Base class for every FedRF-Distill error."""


class UnsupportedModelError(FedRFDistillError):
    """Raised when a requested model framework is not registered."""


class DistillationError(FedRFDistillError):
    """Raised when soft-label distillation produces invalid output."""


class AggregationError(FedRFDistillError):
    """Raised when surrogate aggregation fails (shape mismatch, empty set, ...)."""


class PrivacyBudgetExhausted(FedRFDistillError):
    """Raised when a requested operation would exceed the configured (eps, delta) budget."""

    def __init__(
        self,
        requested_eps: float,
        remaining_eps: float,
        message: str | None = None,
    ) -> None:
        self.requested_eps = requested_eps
        self.remaining_eps = remaining_eps
        super().__init__(
            message
            or f"Privacy budget exhausted: requested eps={requested_eps:.4f}, "
            f"only eps={remaining_eps:.4f} remaining."
        )


class ConfigurationError(FedRFDistillError):
    """Raised when a config object fails validation that is not Pydantic-catchable."""


class ProxyDataError(FedRFDistillError):
    """Raised when the proxy dataset is missing, malformed, or empty."""

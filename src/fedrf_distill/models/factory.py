"""Factory + registry for model adapters.

Why a registry?
---------------
Hook B of the research project is *cross-framework heterogeneity*: a federated
client may locally run sklearn, XGBoost, LightGBM, or CatBoost — yet upload a
wire-compatible :class:`ClientUpdate`. A registry lets the orchestration layer
spell out the framework as a string in a YAML config and resolve it to a
concrete adapter at runtime, with zero hard-coded ``if/elif`` ladders.

Optional adapters
-----------------
XGBoost / LightGBM / CatBoost are *optional* dependencies. We attempt to import
each adapter, and silently skip registration when the underlying library is
missing. The sklearn adapter is always available (sklearn is a hard dep).

Adding a new framework
----------------------
1. Implement ``MyAdapter(BaseModelAdapter)``.
2. Call ``register_adapter("myframework", MyAdapter)``.

The orchestration layer can then instantiate it via ``make_model("myframework")``.
"""

from __future__ import annotations

import logging
from typing import Any

from fedrf_distill.core.exceptions import UnsupportedModelError
from fedrf_distill.models.base import BaseModelAdapter

_logger = logging.getLogger(__name__)

# ── Registry ────────────────────────────────────────────────────────────────
_REGISTRY: dict[str, type[BaseModelAdapter]] = {}


def register_adapter(name: str, adapter_cls: type[BaseModelAdapter]) -> None:
    """Register ``adapter_cls`` under the canonical lower-cased ``name``.

    Overrides existing registrations silently — useful for tests that want to
    swap in a mock adapter.
    """
    if not issubclass(adapter_cls, BaseModelAdapter):
        raise TypeError(
            f"{adapter_cls.__name__} is not a subclass of BaseModelAdapter."
        )
    _REGISTRY[name.lower()] = adapter_cls
    _logger.debug("Registered model adapter '%s' → %s", name, adapter_cls.__name__)


def make_model(name: str, **kwargs: Any) -> BaseModelAdapter:
    """Instantiate a model adapter by framework name.

    Parameters
    ----------
    name:
        Canonical framework key (e.g. ``"sklearn_rf"``, ``"xgboost"``).
        Case-insensitive.
    **kwargs:
        Forwarded to the adapter's constructor.

    Raises
    ------
    UnsupportedModelError
        If ``name`` is not registered. Lists available alternatives.
    """
    key = name.lower()
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise UnsupportedModelError(
            f"Unknown model framework '{name}'. Available: {available}."
        )
    return _REGISTRY[key](**kwargs)


def AVAILABLE_FRAMEWORKS() -> list[str]:
    """Return the sorted list of currently-registered framework keys."""
    return sorted(_REGISTRY)


# ── Auto-register ───────────────────────────────────────────────────────────
def _autoregister() -> None:
    """Best-effort import of every adapter; skip ones whose backend is missing."""
    # sklearn — always available
    from fedrf_distill.models.sklearn_adapter import (
        SklearnDecisionTreeAdapter,
        SklearnRandomForestAdapter,
    )

    register_adapter("sklearn_rf", SklearnRandomForestAdapter)
    register_adapter("sklearn_dt", SklearnDecisionTreeAdapter)
    # Aliases for ergonomics
    register_adapter("sklearn", SklearnRandomForestAdapter)
    register_adapter("rf", SklearnRandomForestAdapter)
    register_adapter("dt", SklearnDecisionTreeAdapter)

    # XGBoost
    try:
        from fedrf_distill.models.xgboost_adapter import XGBoostAdapter

        register_adapter("xgboost", XGBoostAdapter)
        register_adapter("xgb", XGBoostAdapter)
    except ImportError:  # pragma: no cover - depends on env
        _logger.info("XGBoost not installed; xgboost adapter unavailable.")

    # LightGBM
    try:
        from fedrf_distill.models.lightgbm_adapter import LightGBMAdapter

        register_adapter("lightgbm", LightGBMAdapter)
        register_adapter("lgbm", LightGBMAdapter)
    except ImportError:  # pragma: no cover - depends on env
        _logger.info("LightGBM not installed; lightgbm adapter unavailable.")

    # CatBoost
    try:
        from fedrf_distill.models.catboost_adapter import CatBoostAdapter

        register_adapter("catboost", CatBoostAdapter)
        register_adapter("cb", CatBoostAdapter)
    except ImportError:  # pragma: no cover - depends on env
        _logger.info("CatBoost not installed; catboost adapter unavailable.")


_autoregister()

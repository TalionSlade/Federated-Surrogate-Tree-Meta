"""Deterministic seeding across numpy, Python, and (optionally) ML backends.

Used by experiments to guarantee reproducibility. Call ``set_global_seed(s)``
once at the top of any script.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and ML backend RNGs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Optional dependencies — silently skip if missing
    try:  # pragma: no cover
        import xgboost  # noqa: F401

        # XGBoost respects the seed passed to the estimator; no global seed.
    except ImportError:
        pass

    try:  # pragma: no cover
        import lightgbm  # noqa: F401
    except ImportError:
        pass

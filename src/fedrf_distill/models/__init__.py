"""Model adapters provide a uniform interface over heterogeneous tree frameworks.

Each adapter wraps one framework (sklearn, XGBoost, LightGBM, CatBoost) and
exposes the same `fit / predict_proba / predict / serialize / deserialize` API.
The factory function :func:`make_model` looks up the right adapter by name.

This is the foundation for **cross-framework heterogeneity**: clients can mix
frameworks freely because all distillation, aggregation and meta-learning code
only talks to the adapter, never to a framework's native API.
"""

from fedrf_distill.models.base import BaseModelAdapter
from fedrf_distill.models.factory import (
    AVAILABLE_FRAMEWORKS,
    make_model,
    register_adapter,
)
from fedrf_distill.models.sklearn_adapter import (
    SklearnDecisionTreeAdapter,
    SklearnRandomForestAdapter,
)

__all__ = [
    "BaseModelAdapter",
    "make_model",
    "register_adapter",
    "AVAILABLE_FRAMEWORKS",
    "SklearnRandomForestAdapter",
    "SklearnDecisionTreeAdapter",
]

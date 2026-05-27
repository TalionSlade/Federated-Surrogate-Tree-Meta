"""Adapter (de)serialisation helpers.

Surrogates travel between client and coordinator as ``bytes`` plus a
``surrogate_class_path`` (e.g. ``"fedrf_distill.models.sklearn_adapter:Sklearn
DecisionTreeAdapter"``). On the receiving side we *import* that class and call
its ``deserialize`` classmethod, refusing to load anything that isn't a
:class:`BaseModelAdapter` subclass.

Why the class path?
-------------------
Pickle blobs already contain the class reference, but resolving them implicitly
imports arbitrary modules. By stating the class path explicitly we (a) reject
unexpected classes early, (b) make the protocol easier to audit, and (c) leave
room for non-pickle wire formats in the future (e.g. XGBoost's native JSON).
"""

from __future__ import annotations

import importlib
from typing import cast

from fedrf_distill.core.exceptions import FedRFDistillError
from fedrf_distill.models.base import BaseModelAdapter


def serialize_adapter(adapter: BaseModelAdapter) -> tuple[bytes, str]:
    """Return ``(blob, class_path)`` for safe transport over the wire."""
    blob = adapter.serialize()
    class_path = f"{type(adapter).__module__}:{type(adapter).__qualname__}"
    return blob, class_path


def deserialize_adapter(blob: bytes, class_path: str) -> BaseModelAdapter:
    """Import the declared class and call its :meth:`deserialize`.

    Raises
    ------
    FedRFDistillError
        If the class cannot be imported or is not a BaseModelAdapter.
    """
    try:
        module_name, qualname = class_path.split(":", maxsplit=1)
        module = importlib.import_module(module_name)
        cls = module
        for part in qualname.split("."):
            cls = getattr(cls, part)
    except (ValueError, ImportError, AttributeError) as exc:
        raise FedRFDistillError(
            f"Cannot resolve adapter class '{class_path}': {exc}"
        ) from exc

    if not (isinstance(cls, type) and issubclass(cls, BaseModelAdapter)):
        raise FedRFDistillError(
            f"Resolved class '{class_path}' is not a BaseModelAdapter subclass."
        )

    obj = cls.deserialize(blob)
    return cast(BaseModelAdapter, obj)

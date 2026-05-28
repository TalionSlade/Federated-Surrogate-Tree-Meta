"""Tiny context-manager timer used by the orchestrator for wall-clock metrics."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Timer:
    """Records elapsed wall-clock seconds when used as a context manager."""

    elapsed: float = 0.0
    _start: float = field(default=0.0, init=False, repr=False)

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.elapsed = time.perf_counter() - self._start

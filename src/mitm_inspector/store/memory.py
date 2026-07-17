"""Placeholder interface for newest-first bounded flow retention."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator, Mapping
from typing import Any


class MemoryStore:
    """Small bounded message store; raw mitmproxy flow objects are not accepted."""

    def __init__(self, max_items: int = 2_000) -> None:
        if max_items < 1:
            raise ValueError("max_items must be positive")
        self._items: deque[Mapping[str, Any]] = deque(maxlen=max_items)

    def append(self, message: Mapping[str, Any]) -> None:
        self._items.append(dict(message))

    def newest_first(self) -> Iterator[Mapping[str, Any]]:
        return reversed(self._items)

"""Placeholder interface for newest-first bounded flow retention."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator

from mitm_inspector.protocol import ProtocolMessage


class MemoryStore:
    """Small bounded message store; raw mitmproxy flow objects are not accepted."""

    def __init__(self, max_items: int = 2_000) -> None:
        if max_items < 1:
            raise ValueError("max_items must be positive")
        self._items: deque[ProtocolMessage] = deque(maxlen=max_items)

    def append(self, message: ProtocolMessage) -> None:
        self._items.append(message)

    def newest_first(self) -> Iterator[ProtocolMessage]:
        return reversed(self._items)

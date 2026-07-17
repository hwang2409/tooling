"""Placeholder interface for newest-first bounded flow retention."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator

from mitm_inspector.protocol import ParsedMessage, ParsedMessageResult, require_parsed_message


class MemoryStore:
    """Small bounded message store; raw mitmproxy flow objects are not accepted."""

    def __init__(self, max_items: int = 2_000) -> None:
        if max_items < 1:
            raise ValueError("max_items must be positive")
        self._items: deque[ParsedMessageResult] = deque(maxlen=max_items)

    def append(self, message: ParsedMessage) -> None:
        self._items.append(require_parsed_message(message))

    def newest_first(self) -> Iterator[ParsedMessageResult]:
        return reversed(self._items)

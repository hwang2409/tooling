"""Bounded, non-blocking message sink used at the capture boundary."""

from __future__ import annotations

import base64
from collections import deque
from collections.abc import Iterator, Mapping
from threading import Lock

from mitm_inspector.protocol import ParsedMessage, ParsedMessageResult, require_parsed_message


class BoundedMessageSink:
    """A queue which never waits for its consumer.

    Capture hooks call :meth:`offer` from mitmproxy's forwarding path.  A slow
    API/UI consumer therefore causes an explicit drop rather than backpressure
    on the proxy.  The queue contains only revalidated, project-owned protocol
    messages; callers can inspect the counters without receiving mutable data.
    """

    def __init__(self, max_pending: int = 4_096, max_body_bytes: int = 128 * 1024 * 1024) -> None:
        if max_pending < 1:
            raise ValueError("max_pending must be positive")
        if max_body_bytes < 0:
            raise ValueError("max_body_bytes must not be negative")
        self._items: deque[ParsedMessageResult] = deque(maxlen=max_pending)
        self._max_pending = max_pending
        self._max_body_bytes = max_body_bytes
        self._body_bytes = 0
        self._lock = Lock()
        self._accepted = 0
        self._dropped = 0
        self._body_budget_drops = 0

    def offer(self, message: ParsedMessage) -> bool:
        """Queue a message without waiting; return whether it was accepted."""

        retained = require_parsed_message(message)
        body_bytes = _message_body_bytes(retained)
        with self._lock:
            if len(self._items) >= self._max_pending:
                self._dropped += 1
                return False
            if self._body_bytes + body_bytes > self._max_body_bytes:
                self._dropped += 1
                self._body_budget_drops += 1
                return False
            self._items.append(retained)
            self._body_bytes += body_bytes
            self._accepted += 1
            return True

    __call__ = offer

    def drain(self, limit: int | None = None) -> list[ParsedMessageResult]:
        """Remove up to ``limit`` messages for a consumer."""

        if limit is not None and limit < 1:
            raise ValueError("limit must be positive")
        with self._lock:
            count = len(self._items) if limit is None else min(limit, len(self._items))
            result = []
            for _ in range(count):
                message = self._items.popleft()
                self._body_bytes -= _message_body_bytes(message)
                result.append(message)
            return result

    def __iter__(self) -> Iterator[ParsedMessageResult]:
        return iter(self.drain())

    @property
    def accepted_count(self) -> int:
        with self._lock:
            return self._accepted

    @property
    def dropped_count(self) -> int:
        with self._lock:
            return self._dropped

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._items)

    @property
    def body_budget_drops(self) -> int:
        with self._lock:
            return self._body_budget_drops


def _message_body_bytes(message: ParsedMessageResult) -> int:
    payload = message.message if hasattr(message, "message") else message.payload
    message_type = payload.get("type")
    if message_type == "body.chunk":
        value = payload.get("data_base64")
        return _decoded_length(value)
    if message_type == "body.end":
        body = payload.get("body")
        return _descriptor_bytes(body)
    if message_type == "flow.metadata":
        metadata = payload.get("metadata")
        if isinstance(metadata, Mapping):
            return _descriptor_bytes(metadata.get("request_body")) + _descriptor_bytes(
                metadata.get("response_body")
            )
    return 0


def _descriptor_bytes(value: object) -> int:
    if not isinstance(value, Mapping):
        return 0
    return _decoded_length(value.get("data"))


def _decoded_length(value: object) -> int:
    if not isinstance(value, str):
        return 0
    return len(base64.b64decode(value, validate=True))

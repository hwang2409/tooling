"""Bounded, non-blocking message sink used at the capture boundary."""

from __future__ import annotations

import base64
from collections import deque
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from itertools import count
from threading import Lock

from mitm_inspector.protocol import (
    KnownParsedMessage,
    ParsedMessage,
    ParsedMessageResult,
    parse_message,
    parsed_message_to_plain_json,
    require_parsed_message,
)


@dataclass(frozen=True)
class _QueuedMessage:
    position: int
    message: ParsedMessageResult
    body_bytes: int


class BoundedMessageSink:
    """A bounded queue which never waits for its producer or consumer.

    Every offered message gets an additive ``delivery_position``.  A full
    queue, body-budget rejection, or lock contention records the position as a
    loss.  :meth:`drain` emits protocol-v1 ``stream.gap`` messages before the
    next retained message, so loss is resyncable in-band rather than merely a
    side-channel counter.
    """

    def __init__(
        self,
        max_pending: int = 4_096,
        max_body_bytes: int = 128 * 1024 * 1024,
    ) -> None:
        if max_pending < 1:
            raise ValueError("max_pending must be positive")
        if max_body_bytes < 0:
            raise ValueError("max_body_bytes must not be negative")
        self._items: deque[_QueuedMessage] = deque(maxlen=max_pending)
        self._max_pending = max_pending
        self._max_body_bytes = max_body_bytes
        self._body_bytes = 0
        self._lock = Lock()
        self._positions = count(1)
        self._drop_ranges: deque[list[int]] = deque()
        self._last_delivered_position = 0
        self._accepted = 0
        self._dropped_total = 0
        self._body_budget_drops = 0

    def offer(self, message: ParsedMessage) -> bool:
        """Queue a message or record an in-band loss without waiting."""

        retained = require_parsed_message(message)
        body_bytes = _message_body_bytes(retained)
        if not self._lock.acquire(blocking=False):
            self._record_drop(next(self._positions))
            return False
        try:
            position = next(self._positions)
            if len(self._items) >= self._max_pending:
                self._record_drop(position)
                return False
            if self._body_bytes + body_bytes > self._max_body_bytes:
                self._body_budget_drops += 1
                self._record_drop(position)
                return False
            positioned = _with_delivery_position(retained, position)
            self._items.append(_QueuedMessage(position, positioned, body_bytes))
            self._body_bytes += body_bytes
            self._accepted += 1
            return True
        finally:
            self._lock.release()

    __call__ = offer

    def drain(self, limit: int | None = None) -> list[ParsedMessageResult]:
        """Detach queued work under the lock, then decode gaps outside it."""

        if limit is not None and limit < 1:
            raise ValueError("limit must be positive")
        self._lock.acquire()
        try:
            count_to_drain = len(self._items) if limit is None else min(limit, len(self._items))
            detached = [self._items.popleft() for _ in range(count_to_drain)]
            self._body_bytes -= sum(item.body_bytes for item in detached)
            dropped = list(self._drop_ranges)
            self._drop_ranges.clear()
        finally:
            self._lock.release()

        dropped = _merge_ranges(dropped)
        output: list[ParsedMessageResult] = []
        for item in detached:
            dropped_before_item = any(end <= item.position for _, end in dropped)
            if dropped_before_item:
                output.append(_gap(self._last_delivered_position, item.position))
                dropped = [[start, end] for start, end in dropped if end > item.position]
            output.append(item.message)
            self._last_delivered_position = item.position
        for start, end in dropped:
            self._record_pending_range(start, end)
        return output

    def __iter__(self) -> Iterator[ParsedMessageResult]:
        return iter(self.drain())

    @property
    def accepted_count(self) -> int:
        return self._accepted

    @property
    def dropped_count(self) -> int:
        return self._dropped_total

    @property
    def pending_count(self) -> int:
        return len(self._items)

    @property
    def body_budget_drops(self) -> int:
        return self._body_budget_drops

    def _record_drop(self, position: int) -> None:
        # This path intentionally does not acquire the queue lock.  Capture
        # runs on one proxy thread in normal operation; deque append/mutation
        # is atomic under CPython, and drain merges ranges after detaching.
        if self._drop_ranges and self._drop_ranges[-1][1] + 1 == position:
            self._drop_ranges[-1][1] = position
        else:
            self._drop_ranges.append([position, position])
        self._dropped_total += 1

    def _record_pending_range(self, start: int, end: int) -> None:
        if self._drop_ranges and self._drop_ranges[-1][1] + 1 == start:
            self._drop_ranges[-1][1] = end
        else:
            self._drop_ranges.append([start, end])


def _merge_ranges(ranges: Iterable[list[int]]) -> list[list[int]]:
    merged: list[list[int]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def _with_delivery_position(
    message: ParsedMessageResult, position: int
) -> ParsedMessageResult:
    payload = parsed_message_to_plain_json(message)
    payload["delivery_position"] = str(position)
    return parse_message(payload)


def _gap(expected: int, actual: int) -> ParsedMessageResult:
    if actual <= expected:
        raise AssertionError("delivery positions must increase")
    return parse_message(
        {
            "protocol_version": "1",
            "type": "stream.gap",
            "expected_sequence": str(expected),
            "actual_sequence": str(actual),
            "dropped_count": str(actual - expected - 1),
        }
    )


def _message_body_bytes(message: ParsedMessageResult) -> int:
    payload = message.message if isinstance(message, KnownParsedMessage) else message.payload
    message_type = payload.get("type")
    if message_type == "body.chunk":
        return _decoded_length(payload.get("data_base64"))
    if message_type == "body.end":
        return _descriptor_bytes(payload.get("body"))
    if message_type == "flow.metadata":
        metadata = payload.get("metadata")
        if isinstance(metadata, Mapping):
            return _descriptor_bytes(metadata.get("request_body")) + _descriptor_bytes(
                metadata.get("response_body")
            )
    if message_type == "browser.snapshot":
        flows = payload.get("flows")
        return sum(_flow_metadata_body_bytes(flow) for flow in _mappings(flows))
    if message_type == "browser.delta":
        changes = payload.get("changes")
        return sum(
            _flow_metadata_body_bytes(change.get("flow"))
            for change in _mappings(changes)
            if change.get("op") == "upsert"
        )
    return 0


def _flow_metadata_body_bytes(value: object) -> int:
    if not isinstance(value, Mapping):
        return 0
    return _descriptor_bytes(value.get("request_body")) + _descriptor_bytes(
        value.get("response_body")
    )


def _mappings(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, list | tuple):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _descriptor_bytes(value: object) -> int:
    if not isinstance(value, Mapping):
        return 0
    return _decoded_length(value.get("data"))


def _decoded_length(value: object) -> int:
    if not isinstance(value, str):
        return 0
    return len(base64.b64decode(value, validate=True))

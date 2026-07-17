"""Bounded, non-blocking message sink used at the capture boundary."""

from __future__ import annotations

import base64
from collections import deque
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from itertools import count
from queue import Empty, SimpleQueue
from threading import Lock

from mitm_inspector.protocol import (
    MAX_U64,
    KnownParsedMessage,
    OpaqueParsedMessage,
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
    weight: int


class BoundedMessageSink:
    """A bounded queue which never waits for its producer or consumer.

    Producers reserve a delivery position and publish either a retained item or
    a drop event.  Only the consumer constructs positioned messages and gaps;
    consequently no parsing, copying, or payload traversal occurs under the
    queue lock.  The drop event queue is the synchronization boundary for all
    loss accounting, including lock contention and externally recorded loss.
    """

    def __init__(
        self,
        max_pending: int = 4_096,
        max_body_bytes: int = 128 * 1024 * 1024,
        *,
        max_memory_bytes: int = 128 * 1024 * 1024,
    ) -> None:
        if max_pending < 1:
            raise ValueError("max_pending must be positive")
        if max_body_bytes < 0:
            raise ValueError("max_body_bytes must not be negative")
        if max_memory_bytes < 0:
            raise ValueError("max_memory_bytes must not be negative")
        self._items: deque[_QueuedMessage] = deque()
        self._max_pending = max_pending
        self._max_body_bytes = max_body_bytes
        self._max_memory_bytes = max_memory_bytes
        self._body_bytes = 0
        self._memory_bytes = 0
        self._lock = Lock()
        self._positions = count(1)
        self._drop_events: SimpleQueue[int] = SimpleQueue()
        self._pending_drops: list[int] = []
        self._last_delivered_position = 0
        self._accepted = 0
        self._observed_drop_count = 0
        self._body_budget_drops = 0
        self._memory_budget_drops = 0
        self._inflight = 0

    def offer(self, message: ParsedMessage) -> bool:
        """Queue a message or publish an in-band loss without waiting."""

        try:
            self._inflight += 1
            # Probe the lock before touching the payload.  A stream producer
            # that loses the race must do only position/drop bookkeeping.
            if not self._lock.acquire(blocking=False):
                position = next(self._positions)
                self._record_drop(position)
                return False
            self._lock.release()
            _validate_message_numbers(message)
            retained = require_parsed_message(message)
            body_bytes = _message_body_bytes(retained)
            weight = _message_weight(retained)
            if not self._lock.acquire(blocking=False):
                position = next(self._positions)
                self._record_drop(position)
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
                if self._memory_bytes + weight > self._max_memory_bytes:
                    self._memory_budget_drops += 1
                    self._record_drop(position)
                    return False
                self._items.append(_QueuedMessage(position, retained, body_bytes, weight))
                self._body_bytes += body_bytes
                self._memory_bytes += weight
                self._accepted += 1
                return True
            finally:
                self._lock.release()
        finally:
            self._inflight -= 1

    def record_loss(self) -> None:
        """Record a bounded-store/addon loss as a synthetic delivery position."""

        try:
            self._inflight += 1
            self._record_drop(next(self._positions))
        finally:
            self._inflight -= 1

    __call__ = offer

    def drain(self, limit: int | None = None) -> list[ParsedMessageResult]:
        """Detach queue entries under lock and build output outside it."""

        if limit is not None and limit < 1:
            raise ValueError("limit must be positive")
        with self._lock:
            if self._inflight:
                return []
            count_to_drain = len(self._items) if limit is None else min(limit, len(self._items))
            detached = [self._items.popleft() for _ in range(count_to_drain)]
            self._body_bytes -= sum(item.body_bytes for item in detached)
            self._memory_bytes -= sum(item.weight for item in detached)

        # A producer may publish a drop while the queue is being detached.  A
        # second non-blocking drain of the event queue includes all events that
        # completed before this consumer pass; later events remain for the next
        # pass and cannot mutate the consumer's local ranges.
        dropped = self._pending_drops
        self._pending_drops = []
        newly_observed = 0
        while True:
            try:
                dropped.append(self._drop_events.get_nowait())
                newly_observed += 1
            except Empty:
                break
        self._observed_drop_count += newly_observed
        dropped.sort()

        output: list[ParsedMessageResult] = []
        for item in detached:
            prior = [
                position
                for position in dropped
                if self._last_delivered_position < position < item.position
            ]
            if prior:
                output.append(_gap(self._last_delivered_position, item.position))
                self._last_delivered_position = item.position - 1
                dropped = [position for position in dropped if position >= item.position]
            output.append(_with_delivery_position(item.message, item.position))
            self._last_delivered_position = item.position
            dropped = [position for position in dropped if position > item.position]

        # A drop must remain visible even if no later retained item exists.  The
        # next possible delivery position is one beyond the final drop.
        remaining = sorted(set(dropped))
        index = 0
        while index < len(remaining):
            position = remaining[index]
            if position <= self._last_delivered_position:
                index += 1
                continue
            if position != self._last_delivered_position + 1:
                # Positions not yet observed are left for a later pass; this is
                # the only safe choice during a concurrent producer reservation.
                self._pending_drops.extend(remaining[index:])
                break
            end = position
            index += 1
            while index < len(remaining) and remaining[index] == end + 1:
                end = remaining[index]
                index += 1
            output.append(_gap(self._last_delivered_position, end + 1))
            self._last_delivered_position = end
        return output

    def __iter__(self) -> Iterator[ParsedMessageResult]:
        return iter(self.drain())

    @property
    def accepted_count(self) -> int:
        return self._accepted

    @property
    def dropped_count(self) -> int:
        return self._observed_drop_count + self._drop_events.qsize()

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._items)

    @property
    def body_budget_drops(self) -> int:
        return self._body_budget_drops

    @property
    def memory_budget_drops(self) -> int:
        return self._memory_budget_drops

    def _record_drop(self, position: int) -> None:
        self._drop_events.put(position)


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
        return sum(_flow_metadata_body_bytes(flow) for flow in _mappings(payload.get("flows")))
    if message_type == "browser.delta":
        return sum(
            _flow_metadata_body_bytes(change.get("flow"))
            for change in _mappings(payload.get("changes"))
            if change.get("op") == "upsert"
        )
    return 0


def _message_weight(message: ParsedMessageResult) -> int:
    """Count canonical retained data, including additive/nested fields."""

    payload = message.message if isinstance(message, KnownParsedMessage) else message.payload
    return _canonical_weight(payload)


def _validate_message_numbers(message: ParsedMessage) -> None:
    if isinstance(message, KnownParsedMessage):
        payload = message.message
    elif isinstance(message, OpaqueParsedMessage):
        payload = message.payload
    else:
        return
    _validate_bounded_numbers(payload)


def _validate_bounded_numbers(value: object) -> None:
    if type(value) is int:
        if value < -(1 << 63) or value > MAX_U64:
            raise ValueError("integer exceeds the bounded protocol numeric range")
        return
    if isinstance(value, Mapping):
        for item in value.values():
            _validate_bounded_numbers(item)
        return
    if isinstance(value, list | tuple):
        for item in value:
            _validate_bounded_numbers(item)


def _canonical_weight(value: object) -> int:
    if value is None or isinstance(value, bool):
        return 1
    if isinstance(value, int | float):
        if type(value) is int and (value < -(1 << 63) or value > MAX_U64):
            raise ValueError("integer exceeds the bounded protocol numeric range")
        return 8
    if isinstance(value, str):
        return len(value)
    if isinstance(value, Mapping):
        return 8 + sum(len(key) + _canonical_weight(item) for key, item in value.items())
    if isinstance(value, list | tuple):
        return 8 + sum(_canonical_weight(item) for item in value)
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

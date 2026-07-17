"""Bounded, non-blocking message sink used at the capture boundary."""

from __future__ import annotations

import base64
from collections import deque
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from itertools import count
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

MAX_LOSS_RANGES = 256
_PREPARED_TOKEN = object()


@dataclass(frozen=True)
class _LossRange:
    start: int
    end: int


class _PreparedEnvelope:
    """Private immutable message plus metrics trusted by the sink."""

    __slots__ = ("_message", "_body_bytes", "_weight", "_token")
    _message: ParsedMessageResult
    _body_bytes: int
    _weight: int
    _token: object

    def __init__(
        self,
        message: ParsedMessageResult,
        body_bytes: int,
        weight: int,
        *,
        token: object,
    ) -> None:
        if token is not _PREPARED_TOKEN:
            raise TypeError("prepared envelopes are created by BoundedMessageSink.prepare")
        object.__setattr__(self, "_message", message)
        object.__setattr__(self, "_body_bytes", body_bytes)
        object.__setattr__(self, "_weight", weight)
        object.__setattr__(self, "_token", token)

    @property
    def message(self) -> ParsedMessageResult:
        return self._message

    @property
    def body_bytes(self) -> int:
        return self._body_bytes

    @property
    def weight(self) -> int:
        return self._weight


@dataclass(frozen=True)
class _QueuedMessage:
    position: int
    message: ParsedMessageResult
    body_bytes: int
    weight: int


class BoundedMessageSink:
    """A bounded queue with nonblocking producer admission.

    Capture code prepares an immutable envelope before admission.  The single
    admission lock then serializes position assignment, queue insertion, and
    loss-range updates without holding the queue lock during any payload work.
    Raw ``offer`` remains as a compatibility wrapper; the capture hot path uses
    :meth:`prepare` and :meth:`offer_prepared`.

    Losses are retained as at most ``MAX_LOSS_RANGES`` ranges.  If concurrent
    reservations fragment that bounded range set, the ranges collapse into a
    resync interval; queued messages inside that interval are intentionally
    discarded so the resulting stream.gap covers every position exactly.
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
        self._loss_lock = Lock()
        self._positions = count(1)
        self._loss_ranges: deque[_LossRange] = deque()
        self._last_delivered_position = 0
        self._accepted = 0
        self._dropped_total = 0
        self._forced_loss_count = 0
        self._body_budget_drops = 0
        self._memory_budget_drops = 0
        self._loss_range_collapses = 0
        self._loss_resync_active = False
        self._inflight = 0

    def prepare(self, message: ParsedMessage) -> _PreparedEnvelope:
        """Canonicalize once and cache bounded metrics before producer admission."""

        _validate_message_numbers(message)
        retained = require_parsed_message(message)
        return _PreparedEnvelope(
            retained,
            _message_body_bytes(retained),
            _message_weight(retained),
            token=_PREPARED_TOKEN,
        )

    def offer(self, message: ParsedMessage | _PreparedEnvelope) -> bool:
        """Queue a prepared message or compatibility-wrap a parsed message."""

        if isinstance(message, _PreparedEnvelope):
            return self.offer_prepared(message)
        return self.offer_prepared(self.prepare(message))

    def offer_prepared(self, envelope: _PreparedEnvelope) -> bool:
        """Perform one nonblocking admission using a trusted prepared envelope."""

        try:
            self._inflight += 1
            if not self._lock.acquire(blocking=False):
                self._record_drop(next(self._positions))
                return False
            try:
                if envelope._token is not _PREPARED_TOKEN:
                    raise TypeError("invalid prepared envelope")
                return self._enqueue(envelope)
            finally:
                self._lock.release()
        finally:
            self._inflight -= 1

    def _enqueue(self, envelope: _PreparedEnvelope) -> bool:
        position = next(self._positions)
        if len(self._items) >= self._max_pending:
            self._record_drop(position)
            return False
        if self._body_bytes + envelope.body_bytes > self._max_body_bytes:
            self._body_budget_drops += 1
            self._record_drop(position)
            return False
        if self._memory_bytes + envelope.weight > self._max_memory_bytes:
            self._memory_budget_drops += 1
            self._record_drop(position)
            return False
        self._items.append(
            _QueuedMessage(position, envelope.message, envelope.body_bytes, envelope.weight)
        )
        self._body_bytes += envelope.body_bytes
        self._memory_bytes += envelope.weight
        self._accepted += 1
        return True

    def record_loss(self) -> None:
        """Record a bounded-store/addon loss as a synthetic delivery position."""

        try:
            self._inflight += 1
            self._record_drop(next(self._positions))
        finally:
            self._inflight -= 1

    __call__ = offer

    def drain(self, limit: int | None = None) -> list[ParsedMessageResult]:
        """Detach bounded work under lock, then canonicalize gaps outside it."""

        if limit is not None and limit < 1:
            raise ValueError("limit must be positive")
        # A producer never waits for this consumer lock.  The consumer may
        # briefly wait for a producer's preparation, but no queue work is held
        # while messages or gaps are decoded.
        with self._lock:
            if self._inflight:
                return []
            count_to_drain = (
                len(self._items)
                if limit is None
                else min(limit, len(self._items))
            )
            detached = [self._items.popleft() for _ in range(count_to_drain)]
            self._body_bytes -= sum(item.body_bytes for item in detached)
            self._memory_bytes -= sum(item.weight for item in detached)
        with self._loss_lock:
            ranges = [*self._loss_ranges]
            self._loss_ranges.clear()
            resync_active = self._loss_resync_active
            self._loss_resync_active = False
            ranges = _normalize_ranges(ranges)
        return self._drain_detached(detached, ranges, resync_active=resync_active)

    def _drain_detached(
        self,
        detached: list[_QueuedMessage],
        ranges: list[_LossRange],
        *,
        resync_active: bool,
    ) -> list[ParsedMessageResult]:
        output: list[ParsedMessageResult] = []
        for item in detached:
            _discard_expired_ranges(ranges, self._last_delivered_position)
            while ranges and ranges[0].start <= self._last_delivered_position + 1:
                loss = ranges.pop(0)
                if loss.end <= self._last_delivered_position:
                    continue
                output.append(_gap(self._last_delivered_position, loss.end + 1))
                self._last_delivered_position = loss.end
            if item.position <= self._last_delivered_position:
                self._forced_loss_count += 1
                continue
            if ranges and ranges[0].start < item.position:
                # A range can begin after an already delivered position only
                # when the producer/consumer overlap leaves an accepted item
                # in front of it.  Emit that bounded missing interval first.
                loss = ranges.pop(0)
                output.append(_gap(self._last_delivered_position, item.position))
                self._last_delivered_position = item.position - 1
                if loss.end >= item.position:
                    ranges.insert(0, _LossRange(item.position, loss.end))
            if ranges and ranges[0].start == item.position:
                continue
            output.append(_with_delivery_position(item.message, item.position))
            self._last_delivered_position = item.position

        _discard_expired_ranges(ranges, self._last_delivered_position)
        while ranges and ranges[0].start <= self._last_delivered_position + 1:
            loss = ranges.pop(0)
            output.append(_gap(self._last_delivered_position, loss.end + 1))
            self._last_delivered_position = max(self._last_delivered_position, loss.end)
        if ranges:
            with self._loss_lock:
                self._loss_ranges = deque(
                    _normalize_ranges([*self._loss_ranges, *ranges])
                )
                self._loss_resync_active |= resync_active
        return output

    def __iter__(self) -> Iterator[ParsedMessageResult]:
        return iter(self.drain())

    @property
    def accepted_count(self) -> int:
        return self._accepted

    @property
    def dropped_count(self) -> int:
        return self._dropped_total + self._forced_loss_count

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

    @property
    def loss_range_count(self) -> int:
        with self._loss_lock:
            return len(self._loss_ranges)

    @property
    def loss_range_collapses(self) -> int:
        return self._loss_range_collapses

    def _record_drop(self, position: int) -> None:
        with self._loss_lock:
            if self._loss_resync_active and self._loss_ranges:
                current = self._loss_ranges[0]
                self._loss_ranges = deque(
                    [
                        _LossRange(
                            min(current.start, position),
                            max(current.end, position),
                        )
                    ]
                )
                collapsed = False
            else:
                self._loss_ranges, collapsed = _insert_range(
                    self._loss_ranges, position, position
                )
            self._dropped_total += 1
            if collapsed:
                self._loss_range_collapses += 1
                self._loss_resync_active = True


def _insert_range(
    ranges: deque[_LossRange], start: int, end: int
) -> tuple[deque[_LossRange], bool]:
    values = [*ranges, _LossRange(start, end)]
    values = _merge_ranges(values)
    if len(values) <= MAX_LOSS_RANGES:
        return deque(values), False
    collapsed = _LossRange(values[0].start, values[-1].end)
    return deque([collapsed]), True


def _merge_ranges(ranges: list[_LossRange]) -> list[_LossRange]:
    if not ranges:
        return []
    merged: list[_LossRange] = []
    for current in sorted(ranges, key=lambda item: (item.start, item.end)):
        if merged and current.start <= merged[-1].end + 1:
            previous = merged[-1]
            merged[-1] = _LossRange(previous.start, max(previous.end, current.end))
        else:
            merged.append(current)
    return merged


def _normalize_ranges(ranges: list[_LossRange]) -> list[_LossRange]:
    merged = _merge_ranges(ranges)
    if len(merged) > MAX_LOSS_RANGES:
        return [_LossRange(merged[0].start, merged[-1].end)]
    return merged


def _discard_expired_ranges(ranges: list[_LossRange], last: int) -> None:
    while ranges and ranges[0].end <= last:
        ranges.pop(0)


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

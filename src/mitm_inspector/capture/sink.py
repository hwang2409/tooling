"""Bounded, non-blocking message sink used at the capture boundary."""

from __future__ import annotations

import base64
from collections import deque
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
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


@dataclass(frozen=True)
class _LossRange:
    start: int
    end: int


class _PositionExhausted(RuntimeError):
    """The bounded uint64 delivery-position namespace is terminal."""


@dataclass(frozen=True)
class _QueuedMessage:
    position: int
    message: ParsedMessageResult
    body_bytes: int
    weight: int


class BoundedMessageSink:
    """A bounded queue with nonblocking producer admission.

    A producer first reserves an admission slot with a nonblocking,
    sink-owned lock.  Only that successful path canonicalizes and weighs the
    message.  Queue insertion and delivery-position assignment are committed
    after preparation; a preparation failure therefore consumes neither a
    queue slot nor a delivery position.

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
        self._position_lock = Lock()
        self._next_position_value = 1
        self._exhausted = False
        self._reservation_lock = Lock()
        self._consumer_lock = Lock()
        self._reserved_slots = 0
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

    def offer(self, message: ParsedMessage | Mapping[str, object]) -> bool:
        """Admit one raw or parsed message without exposing trusted metrics."""

        self._inflight += 1
        try:
            if self._exhausted or not self._reservation_lock.acquire(False):
                self._record_new_loss()
                return False
            reserved = False
            try:
                if self._exhausted:
                    return False
                if not self._lock.acquire(False):
                    self._record_new_loss()
                    return False
                try:
                    if len(self._items) + self._reserved_slots >= self._max_pending:
                        self._record_new_loss()
                        return False
                    self._reserved_slots += 1
                    reserved = True
                finally:
                    self._lock.release()

                # This is deliberately outside the queue lock.  It is reached
                # only after the sink-owned reservation succeeded.
                _validate_message_numbers(message)
                retained = (
                    parse_message(message)
                    if isinstance(message, Mapping)
                    else require_parsed_message(message)
                )
                body_bytes = _message_body_bytes(retained)
                weight = _message_weight(retained)

                if self._exhausted or not self._lock.acquire(False):
                    self._record_new_loss()
                    return False
                try:
                    return self._commit(retained, body_bytes, weight)
                finally:
                    self._lock.release()
            finally:
                if reserved:
                    with self._lock:
                        self._reserved_slots -= 1
                self._reservation_lock.release()
        finally:
            self._inflight -= 1

    def _commit(self, message: ParsedMessageResult, body_bytes: int, weight: int) -> bool:
        if self._exhausted:
            return False
        try:
            position = self._allocate_position()
        except _PositionExhausted:
            return False
        if self._body_bytes + body_bytes > self._max_body_bytes:
            self._body_budget_drops += 1
            self._record_drop(position)
            return False
        if self._memory_bytes + weight > self._max_memory_bytes:
            self._memory_budget_drops += 1
            self._record_drop(position)
            return False
        item = _QueuedMessage(position, message, body_bytes, weight)
        before_body = self._body_bytes
        before_memory = self._memory_bytes
        before_accepted = self._accepted
        try:
            self._items.append(item)
            self._body_bytes += body_bytes
            self._memory_bytes += weight
            self._accepted += 1
            return True
        except BaseException:
            if self._items and self._items[-1] == item:
                self._items.pop()
            self._body_bytes = before_body
            self._memory_bytes = before_memory
            self._accepted = before_accepted
            self._record_drop(position)
            raise

    def record_loss(self) -> bool:
        """Record a bounded-store/addon loss as a synthetic delivery position."""

        self._inflight += 1
        try:
            return self._record_new_loss()
        finally:
            self._inflight -= 1

    def _record_new_loss(self) -> bool:
        try:
            position = self._allocate_position()
        except _PositionExhausted:
            return False
        self._record_drop(position)
        return True

    def _allocate_position(self) -> int:
        with self._position_lock:
            if self._exhausted:
                raise _PositionExhausted("delivery positions exhausted")
            position = self._next_position_value
            if position == MAX_U64:
                self._exhausted = True
            else:
                self._next_position_value = position + 1
            return position

    __call__ = offer

    def drain(self, limit: int | None = None) -> list[ParsedMessageResult]:
        """Serialize complete consumer transactions, including rollback."""

        with self._consumer_lock:
            return self._drain_once(limit)

    def _drain_once(self, limit: int | None = None) -> list[ParsedMessageResult]:
        """Detach bounded work, then canonicalize it outside the queue lock.

        Detachment is transactional: a canonicalization failure restores the
        exact queue, counters, loss ranges, and delivery cursor.
        """

        if limit is not None and limit < 1:
            raise ValueError("limit must be positive")
        if not self._reservation_lock.acquire(False):
            return []
        # A producer never waits for this consumer lock.  The consumer also
        # does not detach work while a producer is preparing a message.
        try:
            with self._lock:
                if self._inflight > 0:
                    return []
                with self._loss_lock:
                    if self._exhausted and self._loss_ranges:
                        # There is no representable sequence after MAX_U64;
                        # retain the queued/loss state in a stable terminal
                        # condition rather than constructing MAX_U64 + 1.
                        return []
                count_to_drain = (
                    len(self._items)
                    if limit is None
                    else min(limit, len(self._items))
                )
                detached = [self._items.popleft() for _ in range(count_to_drain)]
                self._body_bytes -= sum(item.body_bytes for item in detached)
                self._memory_bytes -= sum(item.weight for item in detached)
                old_last = self._last_delivered_position
                old_forced = self._forced_loss_count
            with self._loss_lock:
                ranges = [*self._loss_ranges]
                self._loss_ranges.clear()
                resync_active = self._loss_resync_active
                self._loss_resync_active = False
                ranges = _normalize_ranges(ranges)
            original_ranges = [*ranges]
        finally:
            self._reservation_lock.release()
        try:
            return self._drain_detached(detached, ranges, resync_active=resync_active)
        except BaseException:
            with self._lock:
                self._items.extendleft(reversed(detached))
                self._body_bytes += sum(item.body_bytes for item in detached)
                self._memory_bytes += sum(item.weight for item in detached)
                self._last_delivered_position = old_last
                self._forced_loss_count = old_forced
            with self._loss_lock:
                self._loss_ranges = deque(
                    _normalize_ranges([*self._loss_ranges, *original_ranges])
                )
                self._loss_resync_active |= resync_active
            raise

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
            while (
                ranges
                and self._last_delivered_position < MAX_U64
                and ranges[0].start <= self._last_delivered_position + 1
            ):
                loss = ranges.pop(0)
                if loss.end <= self._last_delivered_position:
                    continue
                output.append(_gap_after_loss(self._last_delivered_position, loss.end))
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
        while (
            ranges
            and self._last_delivered_position < MAX_U64
            and ranges[0].start <= self._last_delivered_position + 1
        ):
            loss = ranges.pop(0)
            output.append(_gap_after_loss(self._last_delivered_position, loss.end))
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

    @property
    def exhausted(self) -> bool:
        """Whether no further uint64 delivery position can be allocated."""

        return self._exhausted

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
        if merged and (
            current.start <= merged[-1].end
            or (
                merged[-1].end < MAX_U64
                and current.start == merged[-1].end + 1
            )
        ):
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


def _gap_after_loss(expected: int, loss_end: int) -> ParsedMessageResult:
    """Build the gap following a loss without forming MAX_U64 + 1."""

    if loss_end == MAX_U64:
        raise _PositionExhausted("loss reaches the end of the uint64 namespace")
    return _gap(expected, loss_end + 1)


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


def _validate_message_numbers(message: ParsedMessage | Mapping[str, object]) -> None:
    payload: object
    if isinstance(message, KnownParsedMessage):
        payload = message.message
    elif isinstance(message, OpaqueParsedMessage):
        payload = message.payload
    elif isinstance(message, Mapping):
        payload = message
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

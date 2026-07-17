"""Bounded, non-blocking message sink used at the capture boundary."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import Enum, auto
from threading import Condition, Lock, RLock

from mitm_inspector.capture.metrics import (
    body_bytes as _shared_body_bytes,
)
from mitm_inspector.capture.metrics import (
    canonical_weight as _shared_canonical_weight,
)
from mitm_inspector.capture.metrics import (
    validate_bounded_numbers as _validate_bounded_numbers,
)
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


@dataclass(frozen=True)
class _LossState:
    ranges: tuple[_LossRange, ...]
    resync_active: bool
    dropped_total: int
    range_collapses: int


class _PositionExhausted(RuntimeError):
    """The bounded uint64 delivery-position namespace is terminal."""


@dataclass(frozen=True)
class _QueuedMessage:
    position: int
    message: ParsedMessageResult
    body_bytes: int
    weight: int


class _OwnershipPhase(Enum):
    UNOWNED = auto()
    OWNED = auto()
    RELEASING = auto()
    RELEASED = auto()


class _ReservationGate:
    """Nonblocking reservation state with an explicit ownership bit."""

    def __init__(self) -> None:
        self._condition = Condition(Lock())
        self._owned = False

    def acquire(self, blocking: bool = True) -> bool:
        if blocking:
            with self._condition:
                while self._owned:
                    self._condition.wait()
                self._owned = True
                return True
        if not self._condition.acquire(False):
            return False
        try:
            if self._owned:
                return False
            self._owned = True
            return True
        finally:
            self._condition.release()

    def release(self) -> None:
        if not self.release_if_owned():
            raise RuntimeError("reservation gate is not owned")

    def release_if_owned(self) -> bool:
        with self._condition:
            if not self._owned:
                return False
            self._owned = False
            self._condition.notify()
            return True


class _ReservationLease:
    """Own a reservation gate and make release idempotent and observable."""

    def __init__(self, gate: _ReservationGate) -> None:
        self._gate = gate
        self.phase = _OwnershipPhase.UNOWNED

    def acquire(self, blocking: bool = False) -> bool:
        try:
            acquired = self._gate.acquire(blocking)
        except BaseException:
            for _ in range(2):
                try:
                    if not self._gate.release_if_owned():
                        break
                except BaseException:
                    continue
            raise
        if acquired:
            self.phase = _OwnershipPhase.OWNED
        return acquired

    def close(self) -> BaseException | None:
        if self.phase is not _OwnershipPhase.OWNED:
            return None
        self.phase = _OwnershipPhase.RELEASING
        try:
            if not self._gate.release_if_owned():
                raise RuntimeError("owned reservation was already released")
        except BaseException as error:
            # The gate clears ownership before notification, so a release
            # exception cannot strand the gate.  One bounded retry handles an
            # exception injected immediately before or after the clear.
            try:
                self._gate.release_if_owned()
            except BaseException:
                pass
            self.phase = _OwnershipPhase.RELEASED
            return error
        self.phase = _OwnershipPhase.RELEASED
        return None


class _SlotLease:
    """Own the one bounded pending-slot reservation for an offer."""

    def __init__(self, sink: BoundedMessageSink) -> None:
        self._sink = sink
        self.phase = _OwnershipPhase.UNOWNED
        self._before = 0

    def reserve(self) -> bool:
        queue = _ReservationLease(self._sink._lock)
        if not queue.acquire():
            return False
        primary: BaseException | None = None
        try:
            if (
                len(self._sink._items) + self._sink._reserved_slots
                >= self._sink._max_pending
            ):
                return False
            self._before = self._sink._reserved_slots
            self._sink._reserved_slots += 1
            self.phase = _OwnershipPhase.OWNED
            return True
        except BaseException as error:
            primary = error
            self._sink._reserved_slots = self._before
            raise
        finally:
            queue_error = queue.close()
            if primary is None and queue_error is not None:
                raise queue_error

    def close(self) -> BaseException | None:
        if self.phase is not _OwnershipPhase.OWNED:
            return None
        self.phase = _OwnershipPhase.RELEASING
        try:
            self._sink._reserved_slots -= 1
        except BaseException as error:
            self._sink._reserved_slots = self._before
            self.phase = _OwnershipPhase.RELEASED
            return error
        self.phase = _OwnershipPhase.RELEASED
        return None


class _AdmissionPhase(Enum):
    SNAPSHOTTED = auto()
    POSITION_ALLOCATED = auto()
    APPEND_PENDING = auto()
    QUEUE_APPENDED = auto()
    LOSS_RECORDED = auto()
    COMMITTED = auto()
    ROLLED_BACK = auto()


class _DrainPhase(Enum):
    RESERVATION_ACQUIRED = auto()
    SNAPSHOTTED = auto()
    ITEMS_DETACHED = auto()
    RANGES_DETACHED = auto()
    RESERVATION_RELEASED = auto()
    MATERIALIZED = auto()
    ROLLED_BACK = auto()


@dataclass(frozen=True)
class _AdmissionSnapshot:
    items: tuple[_QueuedMessage, ...]
    body_bytes: int
    memory_bytes: int
    accepted: int
    body_budget_drops: int
    memory_budget_drops: int
    position: tuple[int, bool]
    loss: _LossState


class _AdmissionTransaction:
    """Complete sink-owned commit snapshot and phase machine."""

    def __init__(self, sink: BoundedMessageSink) -> None:
        self.sink = sink
        self.snapshot = _AdmissionSnapshot(
            tuple(sink._items),
            sink._body_bytes,
            sink._memory_bytes,
            sink._accepted,
            sink._body_budget_drops,
            sink._memory_budget_drops,
            (sink._next_position_value, sink._exhausted),
            sink._loss_state(),
        )
        self.phase = _AdmissionPhase.SNAPSHOTTED
        self.position: int | None = None
        self.position_after_allocation: tuple[int, bool] | None = None
        self.item: _QueuedMessage | None = None

    def allocate_position(self) -> None:
        self.position = self.sink._allocate_position()
        self.position_after_allocation = (
            self.sink._next_position_value,
            self.sink._exhausted,
        )
        self.phase = _AdmissionPhase.POSITION_ALLOCATED

    def append(self, item: _QueuedMessage) -> None:
        self.item = item
        self.phase = _AdmissionPhase.APPEND_PENDING
        self.sink._items.append(item)
        self.phase = _AdmissionPhase.QUEUE_APPENDED

    def commit_counters(self, body_bytes: int, weight: int) -> None:
        self.sink._body_bytes += body_bytes
        self.sink._memory_bytes += weight
        self.sink._accepted += 1
        self.phase = _AdmissionPhase.COMMITTED

    def record_loss(self, position: int) -> None:
        self.sink._record_drop(position)
        self.phase = _AdmissionPhase.LOSS_RECORDED

    def rollback(self) -> None:
        self._restore_snapshot()
        if self.phase in {
            _AdmissionPhase.APPEND_PENDING,
            _AdmissionPhase.QUEUE_APPENDED,
            _AdmissionPhase.COMMITTED,
        } and self.position is not None:
            try:
                self.sink._record_drop(self.position)
            except BaseException:
                self._restore_snapshot()
            else:
                assert self.position_after_allocation is not None
                self.sink._restore_position_state(self.position_after_allocation)
        self.phase = _AdmissionPhase.ROLLED_BACK

    def _restore_snapshot(self) -> None:
        sink = self.sink
        sink._items.clear()
        sink._items.extend(self.snapshot.items)
        sink._body_bytes = self.snapshot.body_bytes
        sink._memory_bytes = self.snapshot.memory_bytes
        sink._accepted = self.snapshot.accepted
        sink._body_budget_drops = self.snapshot.body_budget_drops
        sink._memory_budget_drops = self.snapshot.memory_budget_drops
        sink._restore_position_state(self.snapshot.position)
        sink._restore_loss_state(self.snapshot.loss)


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
        self._lock = _ReservationGate()
        self._loss_lock = Lock()
        self._position_lock = RLock()
        self._next_position_value = 1
        self._exhausted = False
        self._reservation_lock = _ReservationGate()
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

    def offer(self, message: ParsedMessage | Mapping[str, object]) -> bool:
        """Admit one raw or parsed message without exposing trusted metrics."""

        if self._exhausted:
            self._record_new_loss()
            return False
        reservation = _ReservationLease(self._reservation_lock)
        if not reservation.acquire():
            self._record_new_loss()
            return False
        slot = _SlotLease(self)
        primary: BaseException | None = None
        try:
            if self._exhausted or not slot.reserve():
                self._record_new_loss()
                return False

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

            queue = _ReservationLease(self._lock)
            if self._exhausted or not queue.acquire():
                self._record_new_loss()
                return False
            queue_primary: BaseException | None = None
            try:
                try:
                    with self._position_lock:
                        transaction = _AdmissionTransaction(self)
                        try:
                            return self._commit(transaction, retained, body_bytes, weight)
                        except BaseException:
                            transaction.rollback()
                            raise
                except BaseException as error:
                    queue_primary = error
                    raise
            finally:
                queue_error = queue.close()
                if queue_primary is None and queue_error is not None:
                    raise queue_error
        except BaseException as error:
            primary = error
            raise
        finally:
            slot_error = slot.close()
            reservation_error = reservation.close()
            if primary is None:
                if slot_error is not None:
                    raise slot_error
                if reservation_error is not None:
                    raise reservation_error

    def _commit(
        self,
        transaction: _AdmissionTransaction,
        message: ParsedMessageResult,
        body_bytes: int,
        weight: int,
    ) -> bool:
        if self._exhausted:
            return False
        try:
            transaction.allocate_position()
        except _PositionExhausted:
            return False
        assert transaction.position is not None
        if self._body_bytes + body_bytes > self._max_body_bytes:
            self._body_budget_drops += 1
            transaction.record_loss(transaction.position)
            return False
        if self._memory_bytes + weight > self._max_memory_bytes:
            self._memory_budget_drops += 1
            transaction.record_loss(transaction.position)
            return False
        item = _QueuedMessage(transaction.position, message, body_bytes, weight)
        transaction.append(item)
        transaction.commit_counters(body_bytes, weight)
        return True

    def record_loss(self) -> bool:
        """Record a bounded-store/addon loss as a synthetic delivery position."""

        return self._record_new_loss()

    def _record_new_loss(self) -> bool:
        with self._position_lock:
            transaction = _AdmissionTransaction(self)
            try:
                try:
                    transaction.allocate_position()
                except _PositionExhausted:
                    return False
                assert transaction.position is not None
                transaction.record_loss(transaction.position)
                return True
            except BaseException:
                transaction.rollback()
                raise

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

    def _loss_state(self) -> _LossState:
        with self._loss_lock:
            return _LossState(
                tuple(self._loss_ranges),
                self._loss_resync_active,
                self._dropped_total,
                self._loss_range_collapses,
            )

    def _restore_loss_state(self, state: _LossState) -> None:
        with self._loss_lock:
            self._loss_ranges = deque(state.ranges)
            self._loss_resync_active = state.resync_active
            self._dropped_total = state.dropped_total
            self._loss_range_collapses = state.range_collapses

    def _restore_position_state(self, state: tuple[int, bool]) -> None:
        self._next_position_value, self._exhausted = state

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
        reservation = _ReservationLease(self._reservation_lock)
        if not reservation.acquire():
            return []
        detached: list[_QueuedMessage] = []
        ranges: list[_LossRange] = []
        original_items: tuple[_QueuedMessage, ...] = ()
        original_ranges: tuple[_LossRange, ...] = ()
        original_resync = False
        old_last = self._last_delivered_position
        old_forced = self._forced_loss_count
        before_body = self._body_bytes
        before_memory = self._memory_bytes
        phase = _DrainPhase.RESERVATION_ACQUIRED
        primary: BaseException | None = None
        queue = _ReservationLease(self._lock)
        try:
            if not queue.acquire(blocking=True):
                return []
            queue_primary: BaseException | None = None
            try:
                # Position, queue, and loss snapshots are taken before the
                # first mutation.  The position lock also excludes a loss
                # producer for this short detachment transaction.
                with self._position_lock, self._loss_lock:
                    if self._exhausted and self._loss_ranges:
                        # There is no representable sequence after MAX_U64;
                        # retain the queued/loss state in a stable terminal
                        # condition rather than constructing MAX_U64 + 1.
                        return []
                    original_items = tuple(self._items)
                    original_ranges = tuple(self._loss_ranges)
                    original_resync = self._loss_resync_active
                    old_last = self._last_delivered_position
                    old_forced = self._forced_loss_count
                    before_body = self._body_bytes
                    before_memory = self._memory_bytes
                    phase = _DrainPhase.SNAPSHOTTED
                    count_to_drain = (
                        len(self._items)
                        if limit is None
                        else min(limit, len(self._items))
                    )
                    for _ in range(count_to_drain):
                        detached.append(self._items.popleft())
                    phase = _DrainPhase.ITEMS_DETACHED
                    self._body_bytes -= sum(item.body_bytes for item in detached)
                    self._memory_bytes -= sum(item.weight for item in detached)
                    removed_ranges: list[_LossRange] = []
                    while self._loss_ranges:
                        removed_ranges.append(self._loss_ranges.popleft())
                    phase = _DrainPhase.RANGES_DETACHED
                    self._loss_resync_active = False
                    ranges = _normalize_ranges(removed_ranges)
            except BaseException as error:
                queue_primary = error
                raise
            finally:
                queue_error = queue.close()
                if queue_primary is None and queue_error is not None:
                    raise queue_error
            release_error = reservation.close()
            phase = _DrainPhase.RESERVATION_RELEASED
            if release_error is not None:
                raise release_error
            result = self._drain_detached(detached, ranges, resync_active=original_resync)
            phase = _DrainPhase.MATERIALIZED
            return result
        except BaseException as error:
            primary = error
            rollback_queue = _ReservationLease(self._lock)
            try:
                if rollback_queue.acquire(blocking=True):
                    try:
                        if phase in {
                            _DrainPhase.RESERVATION_RELEASED,
                            _DrainPhase.MATERIALIZED,
                        }:
                            self._items.extendleft(reversed(detached))
                            self._body_bytes += sum(item.body_bytes for item in detached)
                            self._memory_bytes += sum(item.weight for item in detached)
                        elif phase in {
                            _DrainPhase.SNAPSHOTTED,
                            _DrainPhase.ITEMS_DETACHED,
                            _DrainPhase.RANGES_DETACHED,
                        }:
                            self._items.clear()
                            self._items.extend(original_items)
                            self._body_bytes = before_body
                            self._memory_bytes = before_memory
                        self._last_delivered_position = old_last
                        self._forced_loss_count = old_forced
                    finally:
                        rollback_queue.close()
                    with self._loss_lock:
                        if phase in {
                            _DrainPhase.RESERVATION_RELEASED,
                            _DrainPhase.MATERIALIZED,
                        }:
                            self._loss_ranges = deque(
                                _normalize_ranges([*self._loss_ranges, *original_ranges])
                            )
                            self._loss_resync_active |= original_resync
                        elif phase in {
                            _DrainPhase.SNAPSHOTTED,
                            _DrainPhase.ITEMS_DETACHED,
                            _DrainPhase.RANGES_DETACHED,
                        }:
                            self._loss_ranges = deque(original_ranges)
                            self._loss_resync_active = original_resync
            except BaseException:
                # The injected failure belongs to the drain operation.  A
                # rollback cleanup failure must not mask it or trigger a
                # second release attempt.
                pass
            phase = _DrainPhase.ROLLED_BACK
            raise
        finally:
            release_error = reservation.close()
            if release_error is not None and primary is None:
                raise release_error

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
        lease = _ReservationLease(self._lock)
        if not lease.acquire(blocking=True):
            return 0
        try:
            return len(self._items)
        finally:
            lease.close()

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
    return _shared_body_bytes(payload)


def _message_weight(message: ParsedMessageResult) -> int:
    """Count canonical retained data, including additive/nested fields."""

    payload = message.message if isinstance(message, KnownParsedMessage) else message.payload
    return _shared_canonical_weight(payload)


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

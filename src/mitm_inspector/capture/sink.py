"""Bounded, non-blocking capture sink backed by one delivery sequencer."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import replace
from threading import Lock

from mitm_inspector.capture.gate import OwnershipPhase as _OwnershipPhase
from mitm_inspector.capture.gate import ReservationGate as _ReservationGate
from mitm_inspector.capture.gate import ReservationLease as _ReservationLease
from mitm_inspector.capture.metrics import (
    body_bytes as _shared_body_bytes,
)
from mitm_inspector.capture.metrics import (
    canonical_weight as _shared_canonical_weight,
)
from mitm_inspector.capture.metrics import (
    validate_bounded_numbers as _validate_bounded_numbers,
)
from mitm_inspector.capture.sequencer import (
    DeliverySequencer,
    SequencerState,
)
from mitm_inspector.capture.sequencer import (
    gap_after_loss as _gap_after_loss,
)
from mitm_inspector.capture.sequencer import (
    with_delivery_position as _with_delivery_position,
)
from mitm_inspector.protocol import (
    MAX_U64,
    KnownParsedMessage,
    OpaqueParsedMessage,
    ParsedMessage,
    ParsedMessageResult,
    parse_message,
    require_parsed_message,
)


class BoundedMessageSink:
    """Bounded capture sink with one state owner for every delivery mutation."""

    def __init__(
        self,
        max_pending: int = 4_096,
        max_body_bytes: int = 128 * 1024 * 1024,
        *,
        max_memory_bytes: int = 128 * 1024 * 1024,
    ) -> None:
        _validate_sink_limit(max_pending, "max_pending", minimum=1)
        _validate_sink_limit(max_body_bytes, "max_body_bytes", minimum=0)
        _validate_sink_limit(max_memory_bytes, "max_memory_bytes", minimum=0)
        self._max_pending = max_pending
        self._max_body_bytes = max_body_bytes
        self._max_memory_bytes = max_memory_bytes
        self._sequencer = DeliverySequencer()
        # Both names intentionally point at the one admission gate.  There
        # is no second queue/position lock domain to create stale snapshots.
        self._lock = _ReservationGate()
        self._reservation_lock = self._lock
        self._consumer_lock = Lock()
        self._lease_state_lock = Lock()
        self._pending_lease: _ReservationLease | None = None

    @property
    def _state(self) -> SequencerState:
        return self._sequencer.state

    @_state.setter
    def _state(self, value: SequencerState) -> None:
        self._sequencer.state = value

    # Read-only compatibility views expose the sequencer value; they do not
    # maintain a second copy of delivery state.
    @property
    def _next_position_value(self) -> int:
        return self._state.next_position

    @_next_position_value.setter
    def _next_position_value(self, value: int) -> None:
        with self._sequencer.lock:
            self._state = replace(self._state, next_position=value)

    @property
    def _exhausted(self) -> bool:
        return self._state.exhausted

    @_exhausted.setter
    def _exhausted(self, value: bool) -> None:
        with self._sequencer.lock:
            self._state = replace(self._state, exhausted=value)

    @property
    def _body_bytes(self) -> int:
        return self._state.body_bytes

    @_body_bytes.setter
    def _body_bytes(self, value: int) -> None:
        with self._sequencer.lock:
            self._state = replace(self._state, body_bytes=value)

    @property
    def _memory_bytes(self) -> int:
        return self._state.memory_bytes

    @_memory_bytes.setter
    def _memory_bytes(self, value: int) -> None:
        with self._sequencer.lock:
            self._state = replace(self._state, memory_bytes=value)

    @property
    def _last_delivered_position(self) -> int:
        return self._state.last_delivered

    @_last_delivered_position.setter
    def _last_delivered_position(self, value: int) -> None:
        with self._sequencer.lock:
            self._state = replace(self._state, last_delivered=value)

    def offer(self, message: ParsedMessage | Mapping[str, object]) -> bool:
        """Prepare only after immediate gate admission; never waits for drain."""

        pending_error = self._retry_pending_lease()
        if pending_error is not None:
            raise pending_error
        reservation = _ReservationLease(self._reservation_lock)
        if not reservation.acquire(False):
            self._sequencer.record_loss()
            return False
        if not self._sequencer.lock.acquire(False):
            close_error = self._finish_reservation(reservation)
            self._sequencer.record_loss()
            if close_error is not None:
                raise close_error
            return False
        if not self._sequencer.can_accept_locked(self._max_pending):
            self._sequencer.lock.release()
            close_error = self._finish_reservation(reservation)
            self._sequencer.record_loss()
            if close_error is not None:
                raise close_error
            return False
        committed = False
        primary: BaseException | None = None
        before_state = self._sequencer.state
        try:
            _validate_message_numbers(message)
            retained = (
                parse_message(message)
                if isinstance(message, Mapping)
                else require_parsed_message(message)
            )
            body_bytes = _message_body_bytes(retained)
            weight = _message_weight(retained)
            committed = self._sequencer.append_locked(
                retained,
                body_bytes,
                weight,
                self._max_pending,
                self._max_body_bytes,
                self._max_memory_bytes,
            )
            return committed
        except BaseException as error:
            # A fault injected immediately after the sequencer's atomic
            # state replacement has a committed outcome even though the
            # caller did not receive ``True``.
            if self._sequencer.state.accepted > before_state.accepted:
                committed = True
                _mark_committed_exception(error)
            primary = error
            raise
        finally:
            self._sequencer.lock.release()
            close_error = self._finish_reservation(reservation)
            if close_error is not None:
                if committed:
                    _mark_committed_exception(close_error)
                if primary is None:
                    raise close_error

    def record_loss(self) -> bool:
        return self._sequencer.record_loss()

    def drain(self, limit: int | None = None) -> list[ParsedMessageResult]:
        limit = _validate_drain_limit(limit)
        with self._consumer_lock:
            pending_error = self._retry_pending_lease()
            if pending_error is not None:
                raise pending_error
            reservation = _ReservationLease(self._reservation_lock)
            if not reservation.acquire(True):
                return []
            committed = False
            primary: BaseException | None = None
            result: list[ParsedMessageResult] | None = None
            try:
                recovered = self._sequencer.take_committed_batch()
                if recovered is not None:
                    result = recovered
                else:
                    result = self._sequencer.drain(
                        limit, _with_delivery_position, _gap_after_loss
                    )
                    self._sequencer.acknowledge_committed_batch()
                committed = True
            except BaseException as error:
                primary = error
                if self._sequencer.has_committed_batch():
                    _mark_committed_exception(error)
                raise
            finally:
                close_error = self._finish_reservation(reservation)
                if close_error is not None:
                    if committed:
                        # Sequencer commit is already authoritative.  Do not
                        # turn a cleanup-only fault into a lost committed
                        # batch; the caller receives it exactly once.
                        _mark_committed_exception(close_error)
                    elif primary is None:
                        raise close_error
            if result is None:
                raise RuntimeError("drain committed without a result")
            return result

    def _finish_reservation(self, reservation: _ReservationLease) -> BaseException | None:
        with self._lease_state_lock:
            error = reservation.close()
            if reservation.phase is _OwnershipPhase.RELEASING:
                self._pending_lease = reservation
            elif self._pending_lease is reservation:
                self._pending_lease = None
        return error

    def _retry_pending_lease(self) -> BaseException | None:
        with self._lease_state_lock:
            reservation = self._pending_lease
        if reservation is None:
            return None
        error = self._finish_reservation(reservation)
        return error

    def __iter__(self) -> Iterator[ParsedMessageResult]:
        return iter(self.drain())

    @property
    def accepted_count(self) -> int:
        return self._state.accepted

    @property
    def dropped_count(self) -> int:
        self._sequencer.flush_for_read()
        return self._state.dropped + self._state.forced

    @property
    def pending_count(self) -> int:
        return len(self._state.items)

    @property
    def body_budget_drops(self) -> int:
        return self._state.body_budget_drops

    @property
    def memory_budget_drops(self) -> int:
        return self._state.memory_budget_drops

    @property
    def loss_range_count(self) -> int:
        return len(self._sequencer.loss_runs())

    @property
    def exhausted(self) -> bool:
        return self._state.exhausted


def _message_body_bytes(message: ParsedMessageResult) -> int:
    payload = message.message if isinstance(message, KnownParsedMessage) else message.payload
    return _shared_body_bytes(payload)


def _message_weight(message: ParsedMessageResult) -> int:
    payload = message.message if isinstance(message, KnownParsedMessage) else message.payload
    return _shared_canonical_weight(payload)


def _validate_sink_limit(value: object, name: str, *, minimum: int) -> None:
    if type(value) is not int or value < minimum or value > MAX_U64:
        raise ValueError(f"{name} must be an exact bounded integer")


def _validate_drain_limit(limit: object) -> int | None:
    if limit is None:
        return None
    if type(limit) is not int or limit < 1 or limit > MAX_U64:
        raise ValueError("limit must be an exact bounded positive integer")
    return limit


def _mark_committed_exception(error: BaseException) -> None:
    try:
        object.__setattr__(error, "capture_committed", True)
    except BaseException:
        pass


def _validate_message_numbers(message: ParsedMessage | Mapping[str, object]) -> None:
    if isinstance(message, KnownParsedMessage):
        payload: object = message.message
    elif isinstance(message, OpaqueParsedMessage):
        payload = message.payload
    elif isinstance(message, Mapping):
        payload = message
    else:
        return
    _validate_bounded_numbers(payload)

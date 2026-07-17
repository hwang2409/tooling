"""Single-owner delivery sequencing for the capture sink.

The sequencer is the only authority that turns observations into protocol
delivery positions.  Producers that cannot acquire it do not touch its state:
they reserve one ticket from the CPython atomic ``itertools.count`` and leave
the ticket stream for the owner to fold into one loss run later.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass, replace
from threading import RLock

from mitm_inspector.protocol import (
    MAX_U64,
    ParsedMessageResult,
    parse_message,
    parsed_message_to_plain_json,
)


@dataclass(frozen=True)
class LossRun:
    start: int
    end: int


@dataclass(frozen=True)
class QueuedMessage:
    position: int
    message: ParsedMessageResult
    body_bytes: int
    weight: int
    loss_before: tuple[LossRun, ...] = ()


@dataclass(frozen=True)
class SequencerState:
    next_position: int = 1
    exhausted: bool = False
    pending_marker: int = -1
    trailing_losses: tuple[LossRun, ...] = ()
    items: tuple[QueuedMessage, ...] = ()
    body_bytes: int = 0
    memory_bytes: int = 0
    accepted: int = 0
    dropped: int = 0
    forced: int = 0
    last_delivered: int = 0
    body_budget_drops: int = 0
    memory_budget_drops: int = 0


class PositionExhausted(RuntimeError):
    """The bounded uint64 delivery-position namespace is terminal."""


class DeliverySequencer:
    """Own all queue, position, loss, counter, and cursor mutations."""

    def __init__(self) -> None:
        self.lock = RLock()
        self.state = SequencerState()
        self._pending_tickets = itertools.count()

    def note_loss_without_lock(self) -> bool:
        """Record an acknowledged loss without waiting for the owner.

        ``itertools.count.__next__`` is one C-level operation in the supported
        CPython runtime.  It is the producer fallback: no Python lock,
        payload traversal, allocation snapshot, or queue copy is performed.
        """

        next(self._pending_tickets)
        return True

    def _take_pending(self, state: SequencerState) -> SequencerState:
        marker = next(self._pending_tickets)
        pending = marker - state.pending_marker - 1
        if pending <= 0:
            return replace(state, pending_marker=marker)
        state = replace(state, pending_marker=marker)
        return self._append_loss_count(state, pending)

    def _append_loss_count(self, state: SequencerState, count: int) -> SequencerState:
        if count <= 0:
            return state
        if state.exhausted:
            return replace(state, dropped=state.dropped + count)
        available = MAX_U64 - state.next_position + 1
        represented = min(count, available)
        if represented:
            state = self._append_loss_interval(
                state,
                state.next_position,
                state.next_position + represented - 1,
            )
            next_position = state.next_position
            exhausted = state.exhausted
            if state.next_position + represented - 1 == MAX_U64:
                exhausted = True
            else:
                next_position += represented
            state = replace(
                state,
                next_position=next_position,
                exhausted=exhausted,
                dropped=state.dropped + represented,
            )
        if represented < count:
            state = replace(state, dropped=state.dropped + count - represented)
        return state

    def _append_loss_interval(
        self, state: SequencerState, start: int, end: int
    ) -> SequencerState:
        if state.trailing_losses and state.trailing_losses[-1].end < MAX_U64:
            previous = state.trailing_losses[-1]
            if previous.end + 1 == start:
                return replace(
                    state,
                    trailing_losses=state.trailing_losses[:-1]
                    + (LossRun(previous.start, end),),
                )
        return replace(state, trailing_losses=state.trailing_losses + (LossRun(start, end),))

    def _allocate_position(self, state: SequencerState) -> tuple[SequencerState, int]:
        if state.exhausted:
            raise PositionExhausted("delivery positions exhausted")
        position = state.next_position
        if position == MAX_U64:
            return replace(state, exhausted=True), position
        return replace(state, next_position=position + 1), position

    def record_loss(self) -> bool:
        """Record one loss, taking the sequencer only when immediately free."""

        if not self.lock.acquire(False):
            return self.note_loss_without_lock()
        try:
            state = self._take_pending(self.state)
            if state.exhausted:
                self.state = state
                return False
            state, position = self._allocate_position(state)
            state = self._append_loss_interval(state, position, position)
            state = replace(state, dropped=state.dropped + 1)
            self.state = state
            return True
        finally:
            self.lock.release()

    def append(
        self,
        message: ParsedMessageResult,
        body_bytes: int,
        weight: int,
        max_pending: int,
        max_body_bytes: int,
        max_memory_bytes: int,
    ) -> bool:
        """Append one prepared message under one state-owner transaction."""

        with self.lock:
            return self.append_locked(
                message,
                body_bytes,
                weight,
                max_pending,
                max_body_bytes,
                max_memory_bytes,
            )

    def append_locked(
        self,
        message: ParsedMessageResult,
        body_bytes: int,
        weight: int,
        max_pending: int,
        max_body_bytes: int,
        max_memory_bytes: int,
    ) -> bool:
        """Append while the caller owns ``lock``."""

        state = self._take_pending(self.state)
        if state.exhausted or len(state.items) >= max_pending:
            if not state.exhausted:
                state = self._append_loss_count(state, 1)
            self.state = state
            return False
        if state.body_bytes + body_bytes > max_body_bytes:
            state = self._append_loss_count(state, 1)
            self.state = replace(
                state,
                body_budget_drops=state.body_budget_drops + 1,
            )
            return False
        if state.memory_bytes + weight > max_memory_bytes:
            state = self._append_loss_count(state, 1)
            self.state = replace(
                state,
                memory_budget_drops=state.memory_budget_drops + 1,
            )
            return False
        state, position = self._allocate_position(state)
        item = QueuedMessage(
            position,
            message,
            body_bytes,
            weight,
            state.trailing_losses,
        )
        self.state = replace(
            state,
            trailing_losses=(),
            items=state.items + (item,),
            body_bytes=state.body_bytes + body_bytes,
            memory_bytes=state.memory_bytes + weight,
            accepted=state.accepted + 1,
        )
        return True

    def drain(
        self,
        limit: int | None,
        with_position: Callable[[ParsedMessageResult, int], ParsedMessageResult],
        gap_after_loss: Callable[[int, int], ParsedMessageResult],
    ) -> list[ParsedMessageResult]:
        """Materialize and commit one snapshot; rollback is one assignment."""

        with self.lock:
            original = self.state
            state = self._take_pending(original)
            count = len(state.items) if limit is None else min(limit, len(state.items))
            detached = state.items[:count]
            trailing = state.trailing_losses if count == len(state.items) else ()
            if state.exhausted and any(
                loss.end == MAX_U64
                for item in detached
                for loss in item.loss_before
            ) or state.exhausted and any(loss.end == MAX_U64 for loss in trailing):
                self.state = state
                return []
            output: list[ParsedMessageResult] = []
            cursor = state.last_delivered
            forced = state.forced
            try:
                for item in detached:
                    for loss in item.loss_before:
                        output.append(gap_after_loss(cursor, loss.end))
                        cursor = loss.end
                    if item.position <= cursor:
                        forced += 1
                        continue
                    output.append(with_position(item.message, item.position))
                    cursor = item.position
                for loss in trailing:
                    output.append(gap_after_loss(cursor, loss.end))
                    cursor = loss.end
            except BaseException:
                # Keep the pending marker consumed by this snapshot.  The
                # marker is not a loss ticket; rewinding it would count the
                # snapshot itself as a new loss on the next drain.
                self.state = state
                raise

            body = sum(item.body_bytes for item in detached)
            memory = sum(item.weight for item in detached)
            remaining = state.items[count:]
            committed = replace(
                state,
                items=remaining,
                trailing_losses=()
                if count == len(state.items)
                else state.trailing_losses,
                body_bytes=state.body_bytes - body,
                memory_bytes=state.memory_bytes - memory,
                forced=forced,
                last_delivered=cursor,
            )
            self.state = committed
            return output

    def flush_for_read(self) -> None:
        with self.lock:
            self.state = self._take_pending(self.state)

    def loss_runs(self) -> tuple[LossRun, ...]:
        with self.lock:
            state = self._take_pending(self.state)
            self.state = state
            runs = list(state.trailing_losses)
            for item in state.items:
                runs.extend(item.loss_before)
            return tuple(runs)


def with_delivery_position(
    message: ParsedMessageResult, position: int
) -> ParsedMessageResult:
    payload = parsed_message_to_plain_json(message)
    payload["delivery_position"] = str(position)
    return parse_message(payload)


def gap_after_loss(expected: int, loss_end: int) -> ParsedMessageResult:
    if loss_end == MAX_U64:
        raise PositionExhausted("loss reaches the end of the uint64 namespace")
    actual = loss_end + 1
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

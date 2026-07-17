"""Generation-scoped owner gate used to serialize sink admission and drain."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum, auto
from threading import Condition, Lock

_UNSCOPED_OWNER = object()


class OwnershipPhase(Enum):
    UNOWNED = auto()
    OWNED = auto()
    RELEASING = auto()
    RELEASED = auto()


@dataclass(frozen=True)
class _GateState:
    next_generation: int = 1
    owner: object | None = None
    owner_generation: int | None = None
    phase: OwnershipPhase = OwnershipPhase.UNOWNED
    notify_pending: bool = False
    released_generation: int | None = None


class ReservationGate:
    """A condition-backed gate with one atomic, generation-scoped state."""

    def __init__(self) -> None:
        self._condition = Condition(Lock())
        self._state = _GateState()

    def acquire(self, blocking: bool = True, owner: object | None = None) -> bool:
        token = _UNSCOPED_OWNER if owner is None else owner
        if blocking:
            with self._condition:
                while self._state.owner is not None:
                    self._condition.wait()
                self._publish_acquisition(token)
                return True
        if not self._condition.acquire(False):
            return False
        try:
            if self._state.owner is not None:
                return False
            self._publish_acquisition(token)
            return True
        finally:
            self._condition.release()

    def _publish_acquisition(self, owner: object) -> None:
        generation = self._state.next_generation
        self._state = replace(
            self._state,
            next_generation=generation + 1,
            owner=owner,
            owner_generation=generation,
            phase=OwnershipPhase.OWNED,
            notify_pending=False,
        )

    def generation_for(self, owner: object) -> int | None:
        with self._condition:
            if self._state.owner is owner:
                return self._state.owner_generation
            return None

    def release(self) -> None:
        if not self.release_if_owned():
            raise RuntimeError("reservation gate is not owned")

    def begin_cleanup(self, owner: object, generation: int | None = None) -> None:
        """Publish a releasing obligation before any external cleanup call."""

        with self._condition:
            state = self._state
            if state.owner is not owner:
                return
            if generation is not None and state.owner_generation != generation:
                return
            if state.phase is OwnershipPhase.OWNED:
                self._state = replace(
                    state,
                    phase=OwnershipPhase.RELEASING,
                    notify_pending=True,
                )

    def abort_acquisition(self, owner: object) -> BaseException | None:
        """Turn a fault after acquisition publication into retryable cleanup."""

        generation = self.generation_for(owner)
        if generation is None:
            return None
        self.begin_cleanup(owner, generation)
        try:
            self.release_if_owned(owner)
        except BaseException as error:
            return error
        return None

    def release_if_owned(
        self, owner: object | None = None, generation: int | None = None
    ) -> bool:
        with self._condition:
            state = self._state
            if generation is None and owner is not None:
                generation = getattr(owner, "_generation", None)
            if state.owner is None:
                return (
                    generation is not None
                    and state.released_generation is not None
                    and state.released_generation >= generation
                )
            if owner is not None and state.owner is not owner:
                return False
            if generation is not None and state.owner_generation != generation:
                return False
            current_generation = state.owner_generation
            if current_generation is None:
                return False
            if state.phase is not OwnershipPhase.RELEASING:
                self._state = replace(
                    state,
                    phase=OwnershipPhase.RELEASING,
                    notify_pending=True,
                )
                state = self._state
            if state.notify_pending:
                # The obligation is published while the owner is still
                # present.  A notify fault therefore leaves this generation
                # retryable and cannot strand a waiter behind an unowned gate.
                self._condition.notify()
            # Only after the notification obligation is clear may ownership
            # be removed.  Keep the obligation bit set until this one state
            # publication clears it with ownership; if that publication is
            # interrupted, retrying notifies the waiter again.
            self._state = replace(
                self._state,
                owner=None,
                owner_generation=None,
                phase=OwnershipPhase.UNOWNED,
                released_generation=max(
                    current_generation, self._state.released_generation or 0
                ),
            )
            return True

    def retry_cleanup(self) -> BaseException | None:
        """Retry only the gate's currently published releasing generation."""

        with self._condition:
            state = self._state
            if state.owner is None or state.phase is not OwnershipPhase.RELEASING:
                return None
            owner = state.owner
        try:
            self.release_if_owned(owner)
        except BaseException as error:
            return error
        return None

    def is_owned(self, owner: object, generation: int | None = None) -> bool:
        with self._condition:
            return self._state.owner is owner and (
                generation is None or self._state.owner_generation == generation
            )

    def obligations_clear(
        self, owner: object | None = None, generation: int | None = None
    ) -> bool:
        """Return whether this exact generation has no cleanup obligation."""

        with self._condition:
            state = self._state
            if generation is not None:
                return (
                    state.released_generation is not None
                    and state.released_generation >= generation
                )
            return (
                state.owner is not owner
                and state.phase is OwnershipPhase.UNOWNED
                and not state.notify_pending
            )


class ReservationLease:
    """Owner-token lease whose cleanup is durable in the gate state."""

    def __init__(self, gate: ReservationGate) -> None:
        self._gate = gate
        self._generation: int | None = None
        self.phase = OwnershipPhase.UNOWNED

    def acquire(self, blocking: bool = False) -> bool:
        try:
            acquired = self._gate.acquire(blocking, self)
            if acquired:
                self._generation = self._gate.generation_for(self)
                self.phase = OwnershipPhase.OWNED
            return acquired
        except BaseException:
            self._generation = self._gate.generation_for(self)
            if self._generation is not None:
                self.phase = OwnershipPhase.RELEASING
            self._gate.abort_acquisition(self)
            raise

    def close(self) -> BaseException | None:
        if self.phase not in (OwnershipPhase.OWNED, OwnershipPhase.RELEASING):
            return None
        was_releasing = self.phase is OwnershipPhase.RELEASING
        self.prepare_close()
        first_error: BaseException | None = None
        for _ in range(2):
            try:
                released = self._gate.release_if_owned(self)
            except BaseException as error:
                if first_error is None:
                    first_error = error
                released = False
            if released and self._gate.obligations_clear(
                self, self._generation
            ):
                self.phase = OwnershipPhase.RELEASED
                return first_error
            if self._gate.obligations_clear(self, self._generation):
                self.phase = OwnershipPhase.RELEASED
                if was_releasing and first_error is None:
                    return None
                return first_error or RuntimeError("owned reservation was already released")
        return first_error or RuntimeError("owned reservation was not released")

    def prepare_close(self) -> None:
        if self.phase is OwnershipPhase.OWNED:
            self.phase = OwnershipPhase.RELEASING
        if self.phase is OwnershipPhase.RELEASING:
            self._gate.begin_cleanup(self, self._generation)

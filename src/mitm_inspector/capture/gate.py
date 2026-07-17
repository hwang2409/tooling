"""Owner-token gate used to serialize sink admission and consumption."""

from __future__ import annotations

from enum import Enum, auto
from threading import Condition, Lock

_UNSCOPED_OWNER = object()


class OwnershipPhase(Enum):
    UNOWNED = auto()
    OWNED = auto()
    RELEASING = auto()
    RELEASED = auto()


class ReservationGate:
    """A condition-backed gate with owner identity and retryable notify."""

    def __init__(self) -> None:
        self._condition = Condition(Lock())
        self._owner: object | None = None
        self._notification_pending = False

    def acquire(self, blocking: bool = True, owner: object | None = None) -> bool:
        if blocking:
            with self._condition:
                while self._owner is not None:
                    self._condition.wait()
                self._owner = _UNSCOPED_OWNER if owner is None else owner
                return True
        if not self._condition.acquire(False):
            return False
        try:
            if self._owner is not None:
                return False
            self._owner = _UNSCOPED_OWNER if owner is None else owner
            return True
        finally:
            self._condition.release()

    def release(self) -> None:
        if not self.release_if_owned():
            raise RuntimeError("reservation gate is not owned")

    def release_if_owned(self, owner: object | None = None) -> bool:
        with self._condition:
            if self._owner is None:
                if self._notification_pending:
                    self._notify_waiter()
                return False
            if owner is not None and self._owner is not owner:
                return False
            self._owner = None
            self._notify_waiter()
            return True

    def is_owned(self, owner: object) -> bool:
        """Return whether ``owner`` still owns the gate."""

        with self._condition:
            return self._owner is owner

    def _notify_waiter(self) -> None:
        try:
            self._condition.notify()
        except BaseException:
            self._notification_pending = True
            raise
        self._notification_pending = False


class ReservationLease:
    """Owner-token lease with exception-safe, idempotent cleanup."""

    def __init__(self, gate: ReservationGate) -> None:
        self._gate = gate
        self.phase = OwnershipPhase.UNOWNED

    def acquire(self, blocking: bool = False) -> bool:
        try:
            acquired = self._gate.acquire(blocking, self)
        except BaseException:
            try:
                self._gate.release_if_owned(self)
            except BaseException:
                pass
            raise
        if acquired:
            self.phase = OwnershipPhase.OWNED
        return acquired

    def close(self) -> BaseException | None:
        if self.phase not in (OwnershipPhase.OWNED, OwnershipPhase.RELEASING):
            return None
        self.phase = OwnershipPhase.RELEASING
        first_error: BaseException | None = None
        for _ in range(2):
            try:
                released = self._gate.release_if_owned(self)
            except BaseException as error:
                if first_error is None:
                    first_error = error
                released = False
            if released:
                self.phase = OwnershipPhase.RELEASED
                return first_error
            if not self._gate.is_owned(self) and first_error is None:
                self.phase = OwnershipPhase.RELEASED
                return RuntimeError("owned reservation was already released")
            if not self._gate.is_owned(self) and first_error is not None and _ == 1:
                self.phase = OwnershipPhase.RELEASED
                return first_error
        # A failed notification must not make an uncleared owner look
        # released.  The caller can retry close() with the same token.
        return first_error or RuntimeError("owned reservation was not released")

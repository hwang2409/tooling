"""Deterministic bounded in-memory retention for project-owned messages."""

from __future__ import annotations

import base64
import time
from collections import deque
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field

from mitm_inspector.protocol import (
    KnownParsedMessage,
    ParsedMessage,
    ParsedMessageResult,
    require_parsed_message,
)

DEFAULT_MAX_COMPLETED_FLOWS = 2_000
DEFAULT_MAX_AGE_SECONDS = 30 * 60
DEFAULT_MAX_BODY_BYTES = 128 * 1024 * 1024
DEFAULT_MAX_MESSAGES_PER_FLOW = 256
DEFAULT_MAX_MESSAGES = DEFAULT_MAX_COMPLETED_FLOWS * 16


@dataclass
class _StoredMessage:
    order: int
    created_at: float
    message: ParsedMessageResult
    body_bytes: int
    weight: int
    key: tuple[object, ...] | None = None


@dataclass
class _FlowRecord:
    flow_id: str
    created_at: float
    messages: list[_StoredMessage] = field(default_factory=list)
    indexes: dict[tuple[object, ...], int] = field(default_factory=dict)
    completed: bool = False
    completed_at: float | None = None
    completion_order: int | None = None
    evicted_messages: int = 0


class MemoryStore:
    """Retain messages under flow, age, count, and decoded-body bounds.

    All input shapes, including incomplete flows and standalone browser
    envelopes, consume the same bounded message/body accounting.  Replacements
    preserve only the newest coalesced message and update its own order; a
    metadata update never moves an entire flow's history.
    """

    def __init__(
        self,
        max_items: int = DEFAULT_MAX_COMPLETED_FLOWS,
        *,
        max_age_seconds: float = DEFAULT_MAX_AGE_SECONDS,
        max_body_bytes: int = DEFAULT_MAX_BODY_BYTES,
        max_memory_bytes: int = DEFAULT_MAX_BODY_BYTES,
        max_messages: int = DEFAULT_MAX_MESSAGES,
        max_messages_per_flow: int = DEFAULT_MAX_MESSAGES_PER_FLOW,
        max_standalone_messages: int | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_items < 1:
            raise ValueError("max_items must be positive")
        if max_age_seconds < 0:
            raise ValueError("max_age_seconds must not be negative")
        if max_body_bytes < 0:
            raise ValueError("max_body_bytes must not be negative")
        if max_memory_bytes < 0:
            raise ValueError("max_memory_bytes must not be negative")
        if max_messages < 1:
            raise ValueError("max_messages must be positive")
        if max_messages_per_flow < 1:
            raise ValueError("max_messages_per_flow must be positive")
        if max_standalone_messages is not None and max_standalone_messages < 1:
            raise ValueError("max_standalone_messages must be positive")
        self.max_items = max_items
        self.max_age_seconds = max_age_seconds
        self.max_body_bytes = max_body_bytes
        self.max_memory_bytes = max_memory_bytes
        self.max_messages = max_messages
        self.max_messages_per_flow = max_messages_per_flow
        self.max_standalone_messages = max_standalone_messages or max_messages
        self._clock = clock
        self._flows: dict[str, _FlowRecord] = {}
        self._standalone: deque[_StoredMessage] = deque()
        self._order = 0
        self._message_count = 0
        self._body_bytes = 0
        self._memory_bytes = 0
        self._expired_flows = 0
        self._evicted_flows = 0
        self._message_evictions = 0
        self._dropped_messages = 0
        self._per_flow_drops = 0
        self._body_budget_drops = 0
        self._memory_budget_drops = 0

    def append(self, message: ParsedMessage) -> None:
        """Revalidate and retain a protocol message without retaining aliases."""

        now = self._clock()
        self._purge_expired(now)
        retained = require_parsed_message(message)
        payload = retained.message if isinstance(retained, KnownParsedMessage) else retained.payload
        message_type = payload.get("type")
        flow_id = _flow_id_for(payload)
        self._order += 1
        body_bytes = _message_body_bytes(payload)
        weight = _message_weight(payload)

        if flow_id is None:
            if len(self._standalone) >= self.max_standalone_messages:
                self._remove_standalone(0)
                self._message_evictions += 1
            self._make_room_for_message()
            self._standalone.append(
                _StoredMessage(self._order, now, retained, body_bytes, weight)
            )
            self._message_count += 1
            self._body_bytes += body_bytes
            self._memory_bytes += weight
            self._enforce_body_budget()
            self._enforce_memory_budget()
            self._assert_invariants()
            return

        record = self._flows.get(flow_id)
        if record is None:
            record = _FlowRecord(flow_id=flow_id, created_at=now)
            self._flows[flow_id] = record
        key = _coalescing_key(payload, message_type)
        if key is not None and key in record.indexes:
            index = record.indexes[key]
            previous = record.messages[index]
            self._body_bytes -= previous.body_bytes
            self._memory_bytes -= previous.weight
            replacement = _StoredMessage(self._order, now, retained, body_bytes, weight, key)
            record.messages[index] = replacement
            self._body_bytes += body_bytes
            self._memory_bytes += weight
        else:
            if len(record.messages) >= self.max_messages_per_flow:
                if not self._make_room_for_flow(record, incoming_terminal=_is_terminal(retained)):
                    self._dropped_messages += 1
                    self._per_flow_drops += 1
                    self._assert_invariants()
                    return
            else:
                self._make_room_for_message(protected=record)
            stored = _StoredMessage(self._order, now, retained, body_bytes, weight, key)
            if key is not None:
                record.indexes[key] = len(record.messages)
            record.messages.append(stored)
            self._message_count += 1
            self._body_bytes += body_bytes
            self._memory_bytes += weight

        if message_type == "flow.lifecycle" and payload.get("state") == "flow_completed":
            if not record.completed:
                record.completed = True
                record.completed_at = now
                record.completion_order = self._order
            self._enforce_completed_limit()
        self._enforce_body_budget()
        self._enforce_memory_budget()
        self._assert_invariants()

    def newest_first(self) -> Iterator[ParsedMessageResult]:
        """Yield independent messages in true message-newest-first order."""

        self._purge_expired(self._clock())
        entries = [*self._standalone]
        for record in self._flows.values():
            entries.extend(record.messages)
        for stored in sorted(entries, key=lambda item: item.order, reverse=True):
            yield require_parsed_message(stored.message)

    @property
    def counters(self) -> dict[str, int]:
        """Return observable bounded-retention counters."""

        self._purge_expired(self._clock())
        return {
            "completed_flows": sum(record.completed for record in self._flows.values()),
            "retained_flows": len(self._flows),
            "retained_messages": self._message_count,
            "standalone_messages": len(self._standalone),
            "body_bytes": self._body_bytes,
            "evicted_flows": self._evicted_flows,
            "expired_flows": self._expired_flows,
            "message_evictions": self._message_evictions,
            "dropped_messages": self._dropped_messages,
            "per_flow_drops": self._per_flow_drops,
            "flow_message_evictions": sum(
                record.evicted_messages for record in self._flows.values()
            ),
            "body_budget_drops": self._body_budget_drops,
            "memory_bytes": self._memory_bytes,
            "memory_budget_drops": self._memory_budget_drops,
        }

    def clear(self) -> None:
        """Drop all retained state while preserving observable counters."""

        self._flows.clear()
        self._standalone.clear()
        self._message_count = 0
        self._body_bytes = 0
        self._memory_bytes = 0
        self._assert_invariants()

    def _purge_expired(self, now: float) -> None:
        expired = [
            flow_id
            for flow_id, record in self._flows.items()
            if now - record.created_at >= self.max_age_seconds
        ]
        for flow_id in expired:
            self._remove_flow(flow_id)
            self._expired_flows += 1
        stale_standalone = [
            index
            for index, stored in enumerate(self._standalone)
            if now - stored.created_at >= self.max_age_seconds
        ]
        for index in reversed(stale_standalone):
            self._remove_standalone(index)
        self._assert_invariants()

    def _enforce_completed_limit(self) -> None:
        while sum(record.completed for record in self._flows.values()) > self.max_items:
            completed = [
                record
                for record in self._flows.values()
                if record.completed and record.completion_order is not None
            ]
            if not completed:
                return
            oldest = min(completed, key=lambda record: (record.completion_order, record.flow_id))
            self._remove_flow(oldest.flow_id)
            self._evicted_flows += 1

    def _make_room_for_message(self, protected: _FlowRecord | None = None) -> None:
        while self._message_count >= self.max_messages:
            oldest = self._oldest_message(exclude=protected)
            if oldest is None and protected is not None:
                oldest = self._oldest_message()
            if oldest is None:
                return
            kind, owner, index = oldest
            if kind == "flow":
                flow_owner = cast_flow(owner)
                self._remove_flow_message(flow_owner, index, keep_empty=flow_owner is protected)
            else:
                self._remove_standalone(index)
            self._message_evictions += 1

    def _make_room_for_flow(
        self, record: _FlowRecord, *, incoming_terminal: bool
    ) -> bool:
        candidates = [
            (stored.order, index)
            for index, stored in enumerate(record.messages)
            if not _is_terminal(stored.message)
        ]
        if not candidates:
            if not incoming_terminal:
                return False
            # Terminal messages are protected from per-flow chunk eviction, but
            # the global message cap remains absolute.  If this record is the
            # only global candidate, replace its oldest terminal atomically;
            # never leave an empty record and append into an orphan.
            if self._message_count >= self.max_messages:
                self._make_room_for_message(protected=record)
                if self._message_count >= self.max_messages:
                    return False
            return True
        _, index = min(candidates)
        self._remove_flow_message(record, index, keep_empty=True)
        self._message_evictions += 1
        record.evicted_messages += 1
        return True

    def _enforce_body_budget(self) -> None:
        while self._body_bytes > self.max_body_bytes:
            oldest = self._oldest_body_message()
            if oldest is None:
                return
            kind, owner, index = oldest
            if kind == "flow":
                self._remove_flow_message(cast_flow(owner), index)
            else:
                self._remove_standalone(index)
            self._body_budget_drops += 1
            self._message_evictions += 1

    def _enforce_memory_budget(self) -> None:
        while self._memory_bytes > self.max_memory_bytes:
            oldest = self._oldest_weighted_message()
            if oldest is None:
                return
            kind, owner, index = oldest
            if kind == "flow":
                self._remove_flow_message(cast_flow(owner), index)
            else:
                self._remove_standalone(index)
            self._memory_budget_drops += 1
            self._message_evictions += 1

    def _oldest_message(self, exclude: _FlowRecord | None = None) -> tuple[str, object, int] | None:
        candidates: list[tuple[int, str, object, int]] = []
        for record in self._flows.values():
            if record is exclude:
                continue
            for index, stored in enumerate(record.messages):
                candidates.append((stored.order, "flow", record, index))
        candidates.extend(
            (stored.order, "standalone", self._standalone, index)
            for index, stored in enumerate(self._standalone)
        )
        if not candidates:
            return None
        _, kind, owner, index = min(candidates, key=lambda item: (item[0], item[1]))
        return kind, owner, index

    def _oldest_body_message(self) -> tuple[str, object, int] | None:
        candidates: list[tuple[int, str, object, int]] = []
        for record in self._flows.values():
            for index, stored in enumerate(record.messages):
                if stored.body_bytes:
                    candidates.append((stored.order, "flow", record, index))
        candidates.extend(
            (stored.order, "standalone", self._standalone, index)
            for index, stored in enumerate(self._standalone)
            if stored.body_bytes
        )
        if not candidates:
            return None
        _, kind, owner, index = min(candidates, key=lambda item: (item[0], item[1]))
        return kind, owner, index

    def _oldest_weighted_message(self) -> tuple[str, object, int] | None:
        candidates: list[tuple[int, str, object, int]] = []
        for record in self._flows.values():
            for index, stored in enumerate(record.messages):
                candidates.append((stored.order, "flow", record, index))
        candidates.extend(
            (stored.order, "standalone", self._standalone, index)
            for index, stored in enumerate(self._standalone)
        )
        if not candidates:
            return None
        _, kind, owner, index = min(candidates, key=lambda item: (item[0], item[1]))
        return kind, owner, index

    def _remove_flow(self, flow_id: str) -> None:
        record = self._flows.pop(flow_id)
        self._message_count -= len(record.messages)
        self._body_bytes -= sum(stored.body_bytes for stored in record.messages)
        self._memory_bytes -= sum(stored.weight for stored in record.messages)

    def _remove_flow_message(
        self, record: _FlowRecord, index: int, *, keep_empty: bool = False
    ) -> None:
        stored = record.messages.pop(index)
        self._message_count -= 1
        self._body_bytes -= stored.body_bytes
        self._memory_bytes -= stored.weight
        record.indexes = {
            item.key: item_index
            for item_index, item in enumerate(record.messages)
            if item.key is not None
        }
        if not record.messages and not keep_empty:
            self._flows.pop(record.flow_id, None)

    def _assert_invariants(self) -> None:
        visible_messages = len(self._standalone) + sum(
            len(record.messages) for record in self._flows.values()
        )
        visible_body_bytes = sum(
            stored.body_bytes
            for stored in self._standalone
        ) + sum(
            stored.body_bytes
            for record in self._flows.values()
            for stored in record.messages
        )
        assert self._message_count == visible_messages
        assert self._body_bytes == visible_body_bytes
        visible_memory_bytes = sum(
            stored.weight for stored in self._standalone
        ) + sum(
            stored.weight
            for record in self._flows.values()
            for stored in record.messages
        )
        assert self._memory_bytes == visible_memory_bytes
        assert self._message_count <= self.max_messages
        assert self._body_bytes <= self.max_body_bytes
        assert self._memory_bytes <= self.max_memory_bytes
        assert all(record.messages for record in self._flows.values())

    def _remove_standalone(self, index: int) -> None:
        stored = self._standalone[index]
        del self._standalone[index]
        self._message_count -= 1
        self._body_bytes -= stored.body_bytes
        self._memory_bytes -= stored.weight


def cast_flow(value: object) -> _FlowRecord:
    if not isinstance(value, _FlowRecord):
        raise AssertionError("flow owner must be a flow record")
    return value


def _flow_id_for(payload: Mapping[str, object]) -> str | None:
    message_type = payload.get("type")
    if message_type == "flow.metadata":
        metadata = payload.get("metadata")
        if isinstance(metadata, Mapping) and isinstance(metadata.get("flow_id"), str):
            return str(metadata["flow_id"])
    if message_type in {"flow.lifecycle", "body.chunk", "body.end"}:
        flow_id = payload.get("flow_id")
        if isinstance(flow_id, str):
            return flow_id
    return None


def _coalescing_key(
    payload: Mapping[str, object], message_type: object
) -> tuple[object, ...] | None:
    flow_id = _flow_id_for(payload)
    if flow_id is None:
        return None
    if message_type == "flow.metadata":
        return ("metadata",)
    if message_type == "body.end":
        return ("body.end", payload.get("body_side"))
    if message_type == "body.chunk":
        return ("body.chunk", payload.get("body_side"), payload.get("chunk_index"))
    if message_type == "flow.lifecycle":
        return ("flow.lifecycle", payload.get("state"))
    return None


def _is_terminal(message: ParsedMessageResult) -> bool:
    payload = message.message if isinstance(message, KnownParsedMessage) else message.payload
    return payload.get("type") == "body.end" or (
        payload.get("type") == "flow.lifecycle"
        and payload.get("state")
        in {"request_end", "response_end", "error", "flow_completed"}
    )


def _message_body_bytes(payload: Mapping[str, object]) -> int:
    message_type = payload.get("type")
    if message_type == "body.chunk":
        return _decoded_length(payload.get("data_base64"))
    if message_type == "body.end":
        return _descriptor_bytes(payload.get("body"))
    if message_type == "flow.metadata":
        return _flow_metadata_body_bytes(payload.get("metadata"))
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


def _message_weight(payload: Mapping[str, object]) -> int:
    """Count all canonical retained data, including unknown nested fields."""

    return _canonical_weight(payload)


def _canonical_weight(value: object) -> int:
    if value is None or isinstance(value, bool):
        return 1
    if isinstance(value, int | float):
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

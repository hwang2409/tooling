"""Deterministic bounded in-memory retention for project-owned messages."""

from __future__ import annotations

import base64
import time
from collections import OrderedDict, deque
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


@dataclass
class _FlowRecord:
    flow_id: str
    messages: list[ParsedMessageResult] = field(default_factory=list)
    indexes: dict[tuple[object, ...], int] = field(default_factory=dict)
    last_order: int = 0
    completed: bool = False
    completed_at: float | None = None
    body_bytes: int = 0


@dataclass
class _Standalone:
    order: int
    message: ParsedMessageResult


class MemoryStore:
    """Retain the newest completed flows under count, age, and byte bounds.

    Messages are revalidated and copied on ingress.  Flow records contain only
    parsed protocol wrappers; mitmproxy objects are not accepted or retained.
    Metadata and terminal body messages are coalesced by flow/side, which keeps
    repeated lifecycle updates from consuming unbounded memory.
    """

    def __init__(
        self,
        max_items: int = DEFAULT_MAX_COMPLETED_FLOWS,
        *,
        max_age_seconds: float = DEFAULT_MAX_AGE_SECONDS,
        max_body_bytes: int = DEFAULT_MAX_BODY_BYTES,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_items < 1:
            raise ValueError("max_items must be positive")
        if max_age_seconds < 0:
            raise ValueError("max_age_seconds must not be negative")
        if max_body_bytes < 0:
            raise ValueError("max_body_bytes must not be negative")
        self.max_items = max_items
        self.max_age_seconds = max_age_seconds
        self.max_body_bytes = max_body_bytes
        self._clock = clock
        self._records: OrderedDict[str, _FlowRecord] = OrderedDict()
        self._standalone: deque[_Standalone] = deque(maxlen=max_items)
        self._order = 0
        self._evicted_flows = 0
        self._expired_flows = 0
        self._dropped_messages = 0
        self._body_budget_drops = 0

    def append(self, message: ParsedMessage) -> None:
        """Revalidate and retain a protocol message without retaining aliases."""

        self._purge_expired()
        retained = require_parsed_message(message)
        payload = retained.message if isinstance(retained, KnownParsedMessage) else retained.payload
        message_type = payload.get("type")
        flow_id = _flow_id_for(payload)
        self._order += 1
        if flow_id is None:
            if len(self._standalone) == self._standalone.maxlen:
                self._dropped_messages += 1
            self._standalone.append(_Standalone(self._order, retained))
            return

        record = self._records.get(flow_id)
        if record is None:
            record = _FlowRecord(flow_id=flow_id, last_order=self._order)
            self._records[flow_id] = record
        record.last_order = self._order
        key = _coalescing_key(payload, message_type)
        if key is not None and key in record.indexes:
            index = record.indexes[key]
            record.messages[index] = retained
        else:
            if key is not None:
                record.indexes[key] = len(record.messages)
            record.messages.append(retained)
        record.body_bytes = _record_body_bytes(record.messages)

        if message_type == "flow.lifecycle" and payload.get("state") == "flow_completed":
            if not record.completed:
                record.completed = True
                record.completed_at = self._clock()
            self._enforce_completed_limit()
        self._enforce_body_budget()

    def newest_first(self) -> Iterator[ParsedMessageResult]:
        """Yield independent, newest-first copies of retained messages."""

        self._purge_expired()
        entries: list[tuple[int, ParsedMessageResult]] = [
            (item.order, item.message) for item in self._standalone
        ]
        for record in self._records.values():
            entries.extend((record.last_order, message) for message in record.messages)
        for _, message in sorted(entries, key=lambda item: item[0], reverse=True):
            yield require_parsed_message(message)

    @property
    def counters(self) -> dict[str, int]:
        """Return observable bounded-retention counters."""

        self._purge_expired()
        return {
            "completed_flows": sum(record.completed for record in self._records.values()),
            "retained_flows": len(self._records),
            "body_bytes": sum(record.body_bytes for record in self._records.values()),
            "evicted_flows": self._evicted_flows,
            "expired_flows": self._expired_flows,
            "dropped_messages": self._dropped_messages,
            "body_budget_drops": self._body_budget_drops,
        }

    def clear(self) -> None:
        """Drop all retained state while preserving observable counters."""

        self._records.clear()
        self._standalone.clear()

    def _purge_expired(self) -> None:
        now = self._clock()
        expired = [
            flow_id
            for flow_id, record in self._records.items()
            if record.completed
            and record.completed_at is not None
            and now - record.completed_at >= self.max_age_seconds
        ]
        for flow_id in expired:
            del self._records[flow_id]
            self._expired_flows += 1

    def _enforce_completed_limit(self) -> None:
        while sum(record.completed for record in self._records.values()) > self.max_items:
            candidate = next(
                (
                    flow_id
                    for flow_id, record in self._records.items()
                    if record.completed
                ),
                None,
            )
            if candidate is None:
                return
            del self._records[candidate]
            self._evicted_flows += 1

    def _enforce_body_budget(self) -> None:
        while self._body_total() > self.max_body_bytes and self._records:
            candidate = min(
                self._records.values(), key=lambda record: (record.last_order, record.flow_id)
            )
            del self._records[candidate.flow_id]
            self._evicted_flows += 1
            if candidate.body_bytes > self.max_body_bytes:
                self._body_budget_drops += 1

    def _body_total(self) -> int:
        return sum(record.body_bytes for record in self._records.values())


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
    return None


def _record_body_bytes(messages: list[ParsedMessageResult]) -> int:
    total = 0
    for message in messages:
        payload = message.message if isinstance(message, KnownParsedMessage) else message.payload
        message_type = payload.get("type")
        if message_type == "body.chunk":
            data = payload.get("data_base64")
            total += _decoded_length(data)
        elif message_type == "body.end":
            total += _body_descriptor_bytes(payload.get("body"))
        elif message_type == "flow.metadata":
            metadata = payload.get("metadata")
            if isinstance(metadata, Mapping):
                total += _body_descriptor_bytes(metadata.get("request_body"))
                total += _body_descriptor_bytes(metadata.get("response_body"))
    return total


def _body_descriptor_bytes(value: object) -> int:
    if not isinstance(value, Mapping):
        return 0
    return _decoded_length(value.get("data"))


def _decoded_length(value: object) -> int:
    if not isinstance(value, str):
        return 0
    return len(base64.b64decode(value, validate=True))

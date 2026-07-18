"""Transport-agnostic protocol-v1 session, projection, and broadcast logic.

The application owns the bounded store, the shared browser cursor, and the
subscriber set.  Every emitted wire message is round-tripped through full
protocol validation and serialized to an immutable JSON text, so no mapping
proxy, tuple, or mutable alias can cross onto a transport.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime

from mitm_inspector.api.limits import MAX_INGEST_BODY_PREFIX_BYTES
from mitm_inspector.api.projection import collect_grid_flows, diff_grid_changes, grid_flow
from mitm_inspector.json_boundary import PlainJsonObject
from mitm_inspector.protocol import (
    MAX_U64,
    KnownParsedMessage,
    ParsedMessageResult,
    ProtocolError,
    parse_message,
    parsed_message_to_plain_json,
)
from mitm_inspector.store.memory import MemoryStore
from mitm_inspector.store.sqlite import SQLiteFlowStorage

_RELAYED_KNOWN_TYPES = frozenset({"source.hello", "flow.lifecycle", "stream.gap"})
_EVICTION_COUNTER_NAMES = (
    "evicted_flows",
    "expired_flows",
    "message_evictions",
    "dropped_messages",
    "per_flow_drops",
    "flow_message_evictions",
    "body_budget_drops",
    "memory_budget_drops",
)


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(slots=True)
class Subscriber:
    """One connected browser stream; ``deliver`` accepts one JSON text."""

    deliver: Callable[[str], bool]
    on_drop: Callable[[], None] | None = None
    closed: bool = False


@dataclass(frozen=True, slots=True)
class IngestResult:
    parsed: ParsedMessageResult
    relayed: bool
    delta_emitted: bool


@dataclass(slots=True)
class _Counters:
    ingested_messages: int = 0
    relayed_messages: int = 0
    emitted_deltas: int = 0
    emitted_snapshots: int = 0
    dropped_subscribers: int = 0
    resync_responses: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "ingested_messages": self.ingested_messages,
            "relayed_messages": self.relayed_messages,
            "emitted_deltas": self.emitted_deltas,
            "emitted_snapshots": self.emitted_snapshots,
            "dropped_subscribers": self.dropped_subscribers,
            "resync_responses": self.resync_responses,
        }


@dataclass(slots=True)
class _State:
    published: dict[str, PlainJsonObject] = field(default_factory=dict)
    eviction_marks: tuple[int, ...] | None = None
    cursor: int = 0
    cursor_exhausted: bool = False
    snapshot_sequence: int = 0


class ApiApplication:
    """Protocol-v1 session logic shared by the WebSocket and HTTP surfaces."""

    def __init__(
        self,
        store: MemoryStore,
        *,
        source_id: str = "mitm-inspector",
        max_body_prefix_bytes: int = MAX_INGEST_BODY_PREFIX_BYTES,
        max_in_memory_bytes: int = 128 * 1024 * 1024,
        wall_clock: Callable[[], str] = _utc_now_iso,
        cursor_start: int = 0,
        storage: SQLiteFlowStorage | None = None,
    ) -> None:
        if type(source_id) is not str or not source_id:
            raise ValueError("source_id must be a non-empty string")
        for name, value in (
            ("max_body_prefix_bytes", max_body_prefix_bytes),
            ("max_in_memory_bytes", max_in_memory_bytes),
        ):
            if type(value) is not int or value < 0 or value > MAX_U64:
                raise ValueError(f"{name} must be a bounded unsigned 64-bit integer")
        if type(cursor_start) is not int or cursor_start < 0 or cursor_start > MAX_U64:
            raise ValueError("cursor_start must be a bounded unsigned 64-bit integer")
        self._store = store
        self._source_id = source_id
        self._max_body_prefix_bytes = max_body_prefix_bytes
        self._max_in_memory_bytes = max_in_memory_bytes
        self._wall_clock = wall_clock
        self._storage = storage
        self._subscribers: list[Subscriber] = []
        self._state = _State(cursor=cursor_start)
        self._counters = _Counters()

    @property
    def store(self) -> MemoryStore:
        return self._store

    @property
    def storage(self) -> SQLiteFlowStorage | None:
        return self._storage

    @property
    def cursor(self) -> str:
        return str(self._state.cursor)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    @property
    def counters(self) -> dict[str, object]:
        counters: dict[str, object] = dict(self._counters.as_dict())
        counters["cursor"] = str(self._state.cursor)
        counters["cursor_exhausted"] = self._state.cursor_exhausted
        counters["subscribers"] = len(self._subscribers)
        counters["published_flows"] = len(self._state.published)
        counters["store"] = dict(self._store.counters)
        if self._storage is not None:
            counters["storage"] = self._storage.counters
        return counters

    def ingest(self, value: object) -> IngestResult:
        """Validate, retain, relay, and project one capture-side message.

        Raises ``ProtocolError``/``ValueError`` for rejected input; the store
        and broadcast state are untouched in that case.
        """

        parsed = parse_message(value)
        self._store.append(parsed)
        if self._storage is not None:
            self._storage.offer(parsed)
        self._counters.ingested_messages += 1
        payload = self._payload_of(parsed)
        relayed = False
        if isinstance(parsed, KnownParsedMessage):
            if payload.get("type") in _RELAYED_KNOWN_TYPES:
                relayed = True
        else:
            relayed = True
        if relayed:
            self._broadcast(self._wire_text(parsed_message_to_plain_json(parsed)))
            self._counters.relayed_messages += 1
        metadata: Mapping[str, object] | None = None
        if payload.get("type") == "flow.metadata":
            candidate = payload.get("metadata")
            if isinstance(candidate, Mapping):
                metadata = candidate
        delta_emitted = self._reconcile(force=metadata is not None, metadata=metadata)
        return IngestResult(parsed=parsed, relayed=relayed, delta_emitted=delta_emitted)

    def sweep(self) -> bool:
        """Publish removals caused by age expiry while traffic is idle."""

        return self._reconcile(force=False)

    def subscribe(
        self,
        deliver: Callable[[str], bool],
        *,
        on_drop: Callable[[], None] | None = None,
    ) -> Subscriber:
        """Send the connect sequence and register a live subscriber."""

        self._reconcile(force=False)
        frames = (
            self._wire_text(self._hello_message()),
            self._wire_text(self._initial_resync_message()),
            self._wire_text(self._snapshot_message()),
        )
        subscriber = Subscriber(deliver=deliver, on_drop=on_drop)
        for frame in frames:
            if not self._safe_deliver(subscriber, frame):
                self._drop(subscriber)
                return subscriber
        self._subscribers.append(subscriber)
        return subscriber

    def unsubscribe(self, subscriber: Subscriber) -> None:
        subscriber.closed = True
        if subscriber in self._subscribers:
            self._subscribers.remove(subscriber)

    def handle_client_message(self, value: object) -> tuple[str, ...]:
        """Answer one client ``browser.resync`` request with a fresh snapshot."""

        parsed = parse_message(value)
        if (
            not isinstance(parsed, KnownParsedMessage)
            or parsed.message.get("type") != "browser.resync"
        ):
            raise ProtocolError("client messages must be browser.resync requests")
        request = parsed_message_to_plain_json(parsed)
        self._reconcile(force=False)
        echo: PlainJsonObject = {
            "protocol_version": "1",
            "type": "browser.resync",
            "reason": request["reason"],
            "requested_cursor": request["requested_cursor"],
        }
        self._counters.resync_responses += 1
        return (self._wire_text(echo), self._wire_text(self._snapshot_message()))

    def snapshot_text(self) -> str:
        """Return one validated ``browser.snapshot`` JSON text."""

        self._reconcile(force=False)
        return self._wire_text(self._snapshot_message())

    def flow_detail_text(self, flow_id: str) -> str | None:
        """Return every retained message for one selected flow, oldest first."""

        if type(flow_id) is not str or not flow_id:
            return None
        messages: list[PlainJsonObject] = []
        for parsed in self._store.newest_first():
            payload = self._payload_of(parsed)
            if self._message_flow_id(payload) != flow_id:
                continue
            messages.append(parsed_message_to_plain_json(parsed))
        if not messages:
            return None
        messages.reverse()
        detail: PlainJsonObject = {
            "protocol_version": "1",
            "flow_id": flow_id,
            "messages": list(messages),
        }
        return json.dumps(detail, separators=(",", ":"))

    def _reconcile(
        self, *, force: bool, metadata: Mapping[str, object] | None = None
    ) -> bool:
        counters = self._store.counters
        marks = tuple(counters[name] for name in _EVICTION_COUNTER_NAMES)
        if marks == self._state.eviction_marks:
            if not force:
                return False
            if metadata is not None:
                # Nothing was evicted, so this newest metadata message is the
                # only possible change; project just its flow instead of
                # re-scanning and re-copying the entire retained store.
                return self._reconcile_single_flow(metadata)
        self._state.eviction_marks = marks
        current = collect_grid_flows(self._store)
        changes = diff_grid_changes(self._state.published, current)
        self._state.published = current
        if not changes:
            return False
        return self._emit_delta(changes)

    def _reconcile_single_flow(self, metadata: Mapping[str, object]) -> bool:
        flow_id = metadata.get("flow_id")
        if not isinstance(flow_id, str) or not flow_id:
            return False
        flow = grid_flow(metadata)
        published = self._state.published
        unchanged = published.get(flow_id) == flow
        # The just-ingested metadata is now the flow's newest message, which
        # moves the flow to the end of the oldest-first projection order even
        # when its content is identical, exactly like a full re-projection.
        current = dict(published)
        current.pop(flow_id, None)
        current[flow_id] = flow
        self._state.published = current
        if unchanged:
            return False
        return self._emit_delta([{"op": "upsert", "flow": flow}])

    def _emit_delta(self, changes: list[PlainJsonObject]) -> bool:
        if self._state.cursor >= MAX_U64:
            # The shared cursor is a bounded u64.  Exhaustion is a stable
            # terminal condition: connected streams are dropped instead of
            # emitting an unrepresentable successor cursor, and reconnects
            # keep receiving coherent snapshots at the final cursor.
            self._state.cursor_exhausted = True
            for subscriber in list(self._subscribers):
                self._drop(subscriber)
            return False
        self._state.cursor += 1
        delta: PlainJsonObject = {
            "protocol_version": "1",
            "type": "browser.delta",
            "cursor": str(self._state.cursor),
            "changes": list(changes),
        }
        self._broadcast(self._wire_text(delta))
        self._counters.emitted_deltas += 1
        return True

    def _broadcast(self, frame: str) -> None:
        for subscriber in list(self._subscribers):
            if not self._safe_deliver(subscriber, frame):
                self._drop(subscriber)

    def _safe_deliver(self, subscriber: Subscriber, frame: str) -> bool:
        if subscriber.closed:
            return False
        try:
            return bool(subscriber.deliver(frame))
        except Exception:
            return False

    def _drop(self, subscriber: Subscriber) -> None:
        already_closed = subscriber.closed
        self.unsubscribe(subscriber)
        if already_closed:
            return
        self._counters.dropped_subscribers += 1
        if subscriber.on_drop is not None:
            try:
                subscriber.on_drop()
            except Exception:
                # A subscriber teardown callback cannot poison the broadcast.
                pass

    def _hello_message(self) -> PlainJsonObject:
        occurred_at = self._wall_clock()
        if type(occurred_at) is not str or not occurred_at:
            raise ProtocolError("wall clock must produce a non-empty string")
        return {
            "protocol_version": "1",
            "type": "source.hello",
            "source_id": self._source_id,
            "occurred_at": occurred_at,
            "capabilities": {"body_chunks": True, "redaction": "headers-and-query"},
            "limits": {
                "max_body_prefix_bytes": str(self._max_body_prefix_bytes),
                "max_in_memory_bytes": str(self._max_in_memory_bytes),
            },
        }

    def _initial_resync_message(self) -> PlainJsonObject:
        return {
            "protocol_version": "1",
            "type": "browser.resync",
            "reason": "initial_connect",
            "requested_cursor": str(self._state.cursor),
        }

    def _snapshot_message(self) -> PlainJsonObject:
        self._state.snapshot_sequence += 1
        self._counters.emitted_snapshots += 1
        return {
            "protocol_version": "1",
            "type": "browser.snapshot",
            "snapshot_id": f"{self._source_id}-snapshot-{self._state.snapshot_sequence}",
            "cursor": str(self._state.cursor),
            "flows": list(self._state.published.values()),
        }

    @staticmethod
    def _wire_text(message: PlainJsonObject) -> str:
        """Fully revalidate and serialize a message for the wire."""

        wire = parsed_message_to_plain_json(parse_message(message))
        return json.dumps(wire, separators=(",", ":"))

    @staticmethod
    def _payload_of(parsed: ParsedMessageResult) -> Mapping[str, object]:
        return parsed.message if isinstance(parsed, KnownParsedMessage) else parsed.payload

    @staticmethod
    def _message_flow_id(payload: Mapping[str, object]) -> str | None:
        message_type = payload.get("type")
        if message_type == "flow.metadata":
            metadata = payload.get("metadata")
            if isinstance(metadata, Mapping):
                flow_id = metadata.get("flow_id")
                return flow_id if isinstance(flow_id, str) else None
            return None
        flow_id = payload.get("flow_id")
        return flow_id if isinstance(flow_id, str) else None


__all__ = [
    "ApiApplication",
    "IngestResult",
    "Subscriber",
]

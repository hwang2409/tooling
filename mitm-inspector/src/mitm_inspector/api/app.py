"""Transport-agnostic protocol-v1 session, projection, and broadcast logic.

The application owns the bounded store, the shared browser cursor, and the
subscriber set.  Every emitted wire message is round-tripped through full
protocol validation and serialized to an immutable JSON text, so no mapping
proxy, tuple, or mutable alias can cross onto a transport.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

from mitm_inspector.api.bodies import body_content_encoding, decoded_body_descriptor
from mitm_inspector.api.limits import MAX_INGEST_BODY_PREFIX_BYTES
from mitm_inspector.api.projection import (
    LifecycleTimingReducer,
    canonical_grid_flow,
    collect_grid_flows,
    diff_grid_changes,
    enriched_flow,
    grid_flow_with_timing,
)
from mitm_inspector.detail_limits import MAX_DURABLE_DETAIL_OUTPUT_BYTES
from mitm_inspector.json_boundary import PlainJsonObject
from mitm_inspector.protocol import (
    MAX_U64,
    KnownParsedMessage,
    ParsedMessage,
    ParsedMessageResult,
    ProtocolError,
    parse_message,
    parsed_message_to_plain_json,
    require_parsed_message,
    trusted_parsed_message_to_plain_json,
)
from mitm_inspector.store.memory import MemoryStore
from mitm_inspector.store.sqlite import SQLiteFlowStorage

_RELAYED_KNOWN_TYPES = frozenset({"source.hello", "flow.lifecycle", "stream.gap"})
SUBSCRIBER_QUEUE_FRAMES = 256
_INITIAL_SUBSCRIBER_FRAMES = 3
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


class FlowDetailTooLarge(RuntimeError):
    """Raised when bounded durable detail still exceeds its wire ceiling."""


@dataclass(slots=True)
class _Counters:
    ingested_messages: int = 0
    relayed_messages: int = 0
    emitted_deltas: int = 0
    emitted_snapshots: int = 0
    dropped_subscribers: int = 0
    subscribers_partial_history: int = 0
    resync_responses: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "ingested_messages": self.ingested_messages,
            "relayed_messages": self.relayed_messages,
            "emitted_deltas": self.emitted_deltas,
            "emitted_snapshots": self.emitted_snapshots,
            "dropped_subscribers": self.dropped_subscribers,
            "subscribers_partial_history": self.subscribers_partial_history,
            "resync_responses": self.resync_responses,
        }


@dataclass(slots=True)
class _State:
    published: dict[str, PlainJsonObject] = field(default_factory=dict)
    eviction_marks: tuple[int, ...] | None = None
    cursor: int = 0
    cursor_exhausted: bool = False
    snapshot_sequence: int = 0


@dataclass(slots=True)
class _GridProjectionCache:
    """Retain only the small, redacted projection produced for the grid."""

    base_key: tuple[object, ...] | None = None
    key: tuple[object, ...] | None = None
    flow: PlainJsonObject | None = None

    def project(
        self,
        metadata: Mapping[str, object],
        *,
        started_at: str | None,
        ended_at: str | None,
    ) -> PlainJsonObject:
        base_key = _metadata_projection_key(metadata)
        key = (*base_key, started_at, ended_at)
        if self.key == key and self.flow is not None:
            return dict(self.flow)
        if self.base_key == base_key and self.flow is not None:
            self.flow = _retime_grid_flow(self.flow, started_at, ended_at)
            self.key = key
            return dict(self.flow)
        flow = grid_flow_with_timing(metadata, started_at, ended_at)
        self.base_key = base_key
        self.key = key
        self.flow = flow
        return dict(flow)

    def seed(
        self,
        metadata: Mapping[str, object],
        flow: PlainJsonObject,
        *,
        started_at: str | None,
        ended_at: str | None,
    ) -> None:
        self.base_key = _metadata_projection_key(metadata)
        self.key = (*self.base_key, started_at, ended_at)
        self.flow = flow


@dataclass(slots=True)
class _LiveFlow:
    metadata: Mapping[str, object] | None = None
    timing: LifecycleTimingReducer = field(default_factory=LifecycleTimingReducer)
    projection: _GridProjectionCache = field(default_factory=_GridProjectionCache)


def _body_version(descriptor: object) -> tuple[object, ...]:
    """Identify every descriptor field without comparing retained body data."""

    if not isinstance(descriptor, Mapping):
        return (type(descriptor), id(descriptor))
    fields = tuple(
        sorted(
            (str(key), type(value), id(value))
            for key, value in descriptor.items()
            if key != "data"
        )
    )
    return (id(descriptor), id(descriptor.get("data")), fields)


def _metadata_projection_key(metadata: Mapping[str, object]) -> tuple[object, ...]:
    return (
        id(metadata),
        _body_version(metadata.get("request_body")),
        _body_version(metadata.get("response_body")),
    )


def _retime_grid_flow(
    flow: PlainJsonObject,
    started_at: str | None,
    ended_at: str | None,
) -> PlainJsonObject:
    """Update timing in the same canonical position without touching bodies."""

    values = dict(flow)
    if started_at is not None:
        values["started_at"] = started_at
    if ended_at is not None:
        values["ended_at"] = ended_at
    return canonical_grid_flow(values)


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
        self._live_flows: dict[str, _LiveFlow] = {}

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

        parsed = (
            require_parsed_message(value)
            if isinstance(value, ParsedMessage)
            else parse_message(value)
        )
        payload = self._payload_of(parsed)
        self._store.append(parsed)
        self._track_live_flow(payload)
        if self._storage is not None:
            self._storage.offer(parsed)
        self._counters.ingested_messages += 1
        relayed = False
        if isinstance(parsed, KnownParsedMessage):
            if payload.get("type") in _RELAYED_KNOWN_TYPES:
                relayed = True
        else:
            relayed = True
        if relayed:
            self._broadcast(self._wire_text(parsed_message_to_plain_json(parsed)))
            self._counters.relayed_messages += 1
        changed_flow_id = self._message_flow_id(payload)
        delta_emitted = self._reconcile(
            force=changed_flow_id is not None,
            flow_id=changed_flow_id,
        )
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
        initial_frames = [
            self._wire_text(self._hello_message()),
            self._wire_text(self._initial_resync_message()),
            self._wire_text(self._snapshot_message()),
        ]
        subscriber = Subscriber(deliver=deliver, on_drop=on_drop)
        for frame in initial_frames:
            if not self._safe_deliver(subscriber, frame):
                self._drop(subscriber)
                return subscriber
        self._subscribers.append(subscriber)

        historical = self._historical_lifecycle_frames()
        history_capacity = SUBSCRIBER_QUEUE_FRAMES - _INITIAL_SUBSCRIBER_FRAMES
        partial_history = len(historical) > history_capacity
        if partial_history:
            historical = historical[-history_capacity:]
        for frame in historical:
            if not self._safe_deliver(subscriber, frame):
                partial_history = True
                break
        if partial_history:
            self._counters.subscribers_partial_history += 1
        return subscriber

    def _historical_lifecycle_frames(self) -> list[str]:
        """Deliver retained lifecycle history after the initial snapshot."""

        lifecycle_messages: list[tuple[str, str, str, str, str]] = []
        for parsed in self._store.newest_first():
            payload = self._payload_of(parsed)
            if payload.get("type") != "flow.lifecycle":
                continue
            source_id = payload.get("source_id")
            sequence = payload.get("sequence")
            event_id = payload.get("event_id")
            flow_id = payload.get("flow_id")
            if not (
                isinstance(source_id, str)
                and isinstance(sequence, str)
                and isinstance(event_id, str)
                and isinstance(flow_id, str)
            ):
                continue
            historical = parsed_message_to_plain_json(parsed)
            historical["historical"] = True
            lifecycle_messages.append(
                (source_id, sequence, event_id, flow_id,
                 self._wire_text(historical))
            )
        lifecycle_messages.sort(key=lambda item: (item[0], len(item[1]), item[1], item[2]))
        return [item[4] for item in lifecycle_messages]

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
        # The store already indexes messages by flow. Scanning and sorting the
        # complete retained history here made every detail request O(all
        # retained messages), then revalidated each selected body a second
        # time while serializing it. This is an internal store read, so copy
        # the immutable parsed messages without repeating protocol validation.
        for parsed in self._store.trusted_flow_messages(flow_id):
            payload = self._payload_of(parsed)
            if self._message_flow_id(payload) != flow_id:
                continue
            messages.append(trusted_parsed_message_to_plain_json(parsed))
        if not messages:
            return None
        return self._flow_detail_text_from_plain_messages(flow_id, messages)

    def flow_detail_text_from_messages(
        self,
        flow_id: str,
        parsed_messages: Sequence[ParsedMessageResult],
    ) -> str | None:
        """Render validated oldest-first durable messages as flow detail."""

        if type(flow_id) is not str or not flow_id:
            return None
        messages = [
            trusted_parsed_message_to_plain_json(parsed)
            for parsed in parsed_messages
            if self._message_flow_id(self._payload_of(parsed)) == flow_id
        ]
        if not messages:
            return None
        return self._flow_detail_text_from_plain_messages(flow_id, messages)

    def durable_flow_detail_bytes(self, flow_id: str) -> bytes | None:
        """Read, render, and encode one bounded durable detail synchronously."""

        if self._storage is None:
            return None
        messages = self._storage.flow_messages(flow_id)
        detail = self.flow_detail_text_from_messages(flow_id, messages)
        if detail is None:
            return None
        encoded = detail.encode("utf-8")
        if len(encoded) > MAX_DURABLE_DETAIL_OUTPUT_BYTES:
            raise FlowDetailTooLarge
        return encoded

    def _flow_detail_text_from_plain_messages(
        self,
        flow_id: str,
        messages: list[PlainJsonObject],
    ) -> str:
        messages = self._decoded_detail_messages(messages)
        detail: PlainJsonObject = {
            "protocol_version": "1",
            "flow_id": flow_id,
            "messages": list(messages),
        }
        return json.dumps(detail, separators=(",", ":"))

    def _reconcile(
        self,
        *,
        force: bool,
        flow_id: str | None = None,
    ) -> bool:
        counters = self._store.counters
        marks = tuple(counters[name] for name in _EVICTION_COUNTER_NAMES)
        if marks == self._state.eviction_marks:
            if not force:
                return False
            if flow_id is not None:
                return self._reconcile_single_flow(flow_id)
        self._state.eviction_marks = marks
        current = collect_grid_flows(self._store)
        self._sync_live_flows(current)
        changes = diff_grid_changes(self._state.published, current)
        self._state.published = current
        if not changes:
            return False
        return self._emit_delta(changes)

    def _reconcile_single_flow(self, flow_id: str) -> bool:
        live_flow = self._live_flows.get(flow_id)
        if live_flow is None or live_flow.metadata is None:
            return False
        started_at, ended_at = live_flow.timing.values(flow_id)
        flow = live_flow.projection.project(
            live_flow.metadata,
            started_at=started_at,
            ended_at=ended_at,
        )
        published = self._state.published
        unchanged = published.get(flow_id) == flow
        if flow_id not in published:
            current = {flow_id: flow}
            current.update(published)
            self._state.published = current
        else:
            published[flow_id] = flow
        if unchanged:
            return False
        return self._emit_delta([{"op": "upsert", "flow": flow}])

    def _track_live_flow(self, payload: Mapping[str, object]) -> None:
        flow_id = self._message_flow_id(payload)
        if flow_id is None:
            return
        live_flow = self._live_flows.setdefault(flow_id, _LiveFlow())
        if payload.get("type") == "flow.metadata":
            live_flow.metadata = self._store.latest_flow_metadata(flow_id)
        elif payload.get("type") == "flow.lifecycle":
            timing = LifecycleTimingReducer()
            for retained in self._store.trusted_flow_messages(flow_id):
                timing.add(self._payload_of(retained))
            live_flow.timing = timing

    def _sync_live_flows(self, current: Mapping[str, PlainJsonObject]) -> None:
        synced: dict[str, _LiveFlow] = {}
        for flow_id, flow in current.items():
            metadata = self._store.latest_flow_metadata(flow_id)
            if metadata is None:
                continue
            timing = LifecycleTimingReducer()
            for parsed in self._store.trusted_flow_messages(flow_id):
                timing.add(self._payload_of(parsed))
            started_at, ended_at = timing.values(flow_id)
            live_flow = _LiveFlow(metadata=metadata, timing=timing)
            live_flow.projection.seed(
                metadata,
                flow,
                started_at=started_at,
                ended_at=ended_at,
            )
            synced[flow_id] = live_flow
        self._live_flows = synced

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
    def _decoded_detail_messages(messages: list[PlainJsonObject]) -> list[PlainJsonObject]:
        metadata_values = [
            message.get("metadata")
            for message in messages
            if message.get("type") == "flow.metadata"
            and isinstance(message.get("metadata"), Mapping)
        ]
        latest = metadata_values[-1] if metadata_values else None
        if not isinstance(latest, Mapping):
            return messages
        timing = LifecycleTimingReducer()
        for message in messages:
            timing.add(message)
        flow_id = latest.get("flow_id")
        started_at, ended_at = (
            timing.values(flow_id) if isinstance(flow_id, str) else (None, None)
        )
        encodings = {
            side: body_content_encoding(latest, side) for side in ("request", "response")
        }
        decoded_sides: set[str] = set()
        for side, encoding in encodings.items():
            descriptor = latest.get(f"{side}_body")
            if descriptor is None:
                continue
            _decoded, was_decoded = decoded_body_descriptor(descriptor, encoding)
            if was_decoded:
                decoded_sides.add(side)
        result: list[PlainJsonObject] = []
        for message in messages:
            message_type = message.get("type")
            if message_type == "flow.metadata":
                metadata = message.get("metadata")
                if isinstance(metadata, Mapping):
                    message["metadata"] = enriched_flow(
                        metadata, started_at=started_at, ended_at=ended_at
                    )
            elif message_type == "body.chunk":
                message_side = message.get("body_side")
                if isinstance(message_side, str) and message_side in decoded_sides:
                    continue
            elif message_type == "body.end":
                message_side = message.get("body_side")
                body = message.get("body")
                if isinstance(message_side, str) and body is not None:
                    decoded, was_decoded = decoded_body_descriptor(
                        body, encodings.get(message_side)
                    )
                    if was_decoded and isinstance(decoded, dict):
                        message["body"] = decoded
                        size = decoded.get("size_bytes")
                        if isinstance(size, str):
                            message["total_bytes"] = size
            result.append(message)
        return result

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
    "FlowDetailTooLarge",
    "IngestResult",
    "Subscriber",
]

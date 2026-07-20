"""Body-redacted grid projection of retained flow metadata.

The browser grid stream (``browser.snapshot``/``browser.delta``) must never
carry captured body bytes; bodies are delivered only through the explicit
per-flow selection endpoint.  The projection keeps every descriptor
schema-valid while stripping the base64 prefix data.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import cast

from mitm_inspector.api.bodies import (
    body_content_encoding,
    decoded_body_parts,
)
from mitm_inspector.api.summary import flow_summary
from mitm_inspector.json_boundary import (
    PlainJsonObject,
    PlainJsonValue,
    canonicalize_json,
)
from mitm_inspector.protocol import KnownParsedMessage, is_rfc3339_utc
from mitm_inspector.store.memory import MemoryStore

_DATA_BEARING_STATES = frozenset({"captured", "truncated"})
_KNOWN_FLOW_ORDER = (
    "flow_id",
    "session_id",
    "method",
    "scheme",
    "host",
    "port",
    "path",
    "request_headers",
    "response_headers",
    "response_status",
    "request_body",
    "response_body",
    "started_at",
    "ended_at",
    "request_body_size",
    "request_content_type",
    "response_body_size",
    "response_content_type",
    "content_encoding",
    "summary",
)
_KNOWN_FLOW_FIELDS = frozenset(_KNOWN_FLOW_ORDER)


def _timing_value(metadata: Mapping[str, object], key: str) -> str | None:
    value = metadata.get(key)
    return value if isinstance(value, str) and is_rfc3339_utc(value) else None


@dataclass(slots=True)
class LifecycleTimingReducer:
    """Sequence-aware lifecycle timing shared by every API projection."""

    _started: dict[str, tuple[int, str, str, str]] = field(default_factory=dict)
    _ended: dict[str, tuple[int, str, str, str]] = field(default_factory=dict)

    def add(self, payload: Mapping[str, object]) -> None:
        if payload.get("type") != "flow.lifecycle":
            return
        flow_id = payload.get("flow_id")
        sequence = payload.get("sequence")
        occurred_at = payload.get("occurred_at")
        source_id = payload.get("source_id")
        event_id = payload.get("event_id")
        state = payload.get("state")
        if not (
            isinstance(flow_id, str)
            and isinstance(sequence, str)
            and isinstance(occurred_at, str)
            and isinstance(source_id, str)
            and isinstance(event_id, str)
            and is_rfc3339_utc(occurred_at)
        ):
            return
        try:
            candidate = (int(sequence), occurred_at, source_id, event_id)
        except ValueError:
            return
        if state == "request_started" and (
            flow_id not in self._started
            or candidate[0] < self._started[flow_id][0]
            or (
                candidate[0] == self._started[flow_id][0]
                and candidate[1:] > self._started[flow_id][1:]
            )
        ):
            self._started[flow_id] = candidate
        if state in {"flow_completed", "error"} and (
            flow_id not in self._ended
            or candidate > self._ended[flow_id]
        ):
            self._ended[flow_id] = candidate

    def values(self, flow_id: str) -> tuple[str | None, str | None]:
        started = self._started.get(flow_id)
        ended = self._ended.get(flow_id)
        return (
            started[1] if started is not None else None,
            ended[1] if ended is not None else None,
        )


def redacted_body_descriptor(descriptor: PlainJsonValue) -> PlainJsonValue:
    """Return a schema-valid copy of a body descriptor without body data."""

    if not isinstance(descriptor, dict):
        return descriptor
    if descriptor.get("state") not in _DATA_BEARING_STATES:
        return descriptor
    redacted: PlainJsonObject = {
        "state": "truncated",
        "size_bytes": descriptor.get("size_bytes", "0"),
        "captured_bytes": "0",
        "encoding": "base64",
        "data": "",
    }
    if "content_type" in descriptor:
        redacted["content_type"] = descriptor["content_type"]
    return redacted


def canonical_grid_flow(flow: Mapping[str, object]) -> PlainJsonObject:
    """Assemble projection fields in one stable top-level order."""

    result: PlainJsonObject = {}
    for key in _KNOWN_FLOW_ORDER:
        if key in flow:
            result[key] = cast(PlainJsonValue, flow[key])
    for key in sorted(flow):
        if key not in _KNOWN_FLOW_FIELDS:
            result[key] = cast(PlainJsonValue, flow[key])
    return result


def grid_flow(metadata: Mapping[str, object]) -> PlainJsonObject:
    """Copy validated flow metadata into an independent body-redacted flow."""

    copied = enriched_flow(metadata)
    for side in ("request_body", "response_body"):
        if side in copied:
            copied[side] = redacted_body_descriptor(copied[side])
    return copied


def enriched_flow(
    metadata: Mapping[str, object],
    *,
    started_at: str | None = None,
    ended_at: str | None = None,
) -> PlainJsonObject:
    """Copy metadata and add decoded bodies, wire facts, and a summary."""

    copied = canonicalize_json(metadata, label="metadata")
    if not isinstance(copied, dict):
        raise ValueError("flow metadata must be an object")
    session_id = copied.get("session_id")
    copied["session_id"] = session_id if isinstance(session_id, str) else None
    resolved_started_at = started_at or _timing_value(copied, "started_at")
    resolved_ended_at = ended_at or _timing_value(copied, "ended_at")
    if resolved_started_at is None:
        copied.pop("started_at", None)
    else:
        copied["started_at"] = resolved_started_at
    if resolved_ended_at is None:
        copied.pop("ended_at", None)
    else:
        copied["ended_at"] = resolved_ended_at
    decoded_bodies: dict[str, bytes | None] = {}
    decoded_encodings: PlainJsonObject = {}
    for side in ("request", "response"):
        body_key = f"{side}_body"
        descriptor = copied.get(body_key)
        if isinstance(descriptor, dict):
            size = descriptor.get("size_bytes")
            if isinstance(size, str):
                copied[f"{side}_body_size"] = size
            content_type = descriptor.get("content_type")
            if isinstance(content_type, str):
                copied[f"{side}_content_type"] = content_type
        encoding = body_content_encoding(copied, side)
        if descriptor is not None:
            parts = decoded_body_parts(descriptor, encoding)
            copied[body_key] = parts.descriptor
            # Record failed attempts too. Otherwise summary extraction would
            # retry the same expensive invalid compressed body.
            decoded_bodies[side] = parts.data
            was_decoded = parts.was_decoded
            if was_decoded and encoding is not None:
                decoded_encodings[side] = encoding
    if decoded_encodings:
        copied["content_encoding"] = decoded_encodings
    else:
        copied.pop("content_encoding", None)
    copied["summary"] = flow_summary(copied, decoded_bodies=decoded_bodies)
    return canonical_grid_flow(copied)


def flow_id_of(metadata: Mapping[str, object]) -> str | None:
    flow_id = metadata.get("flow_id")
    return flow_id if isinstance(flow_id, str) and flow_id else None


def collect_grid_flows(
    store: MemoryStore,
) -> dict[str, PlainJsonObject]:
    """Project newest metadata in stable newest-flow-first insertion order."""

    newest: dict[str, Mapping[str, object]] = {}
    timing = LifecycleTimingReducer()
    for parsed in store.newest_first():
        if not isinstance(parsed, KnownParsedMessage):
            continue
        payload = parsed.message
        message_type = payload.get("type")
        if message_type == "flow.lifecycle":
            timing.add(payload)
            continue
        if message_type != "flow.metadata":
            continue
        candidate_metadata = payload.get("metadata")
        if not isinstance(candidate_metadata, Mapping):
            continue
        flow_id = flow_id_of(candidate_metadata)
        if flow_id is None or flow_id in newest:
            continue
        newest[flow_id] = candidate_metadata
    flow_order = store.flow_ids_newest_first()
    return {
        flow_id: grid_flow_with_timing(newest[flow_id], *timing.values(flow_id))
        for flow_id in flow_order
        if flow_id in newest
    }


def collect_grid_flow(
    store: MemoryStore,
    target_flow_id: str,
) -> PlainJsonObject | None:
    """Project one retained flow without rebuilding the full grid."""

    metadata: Mapping[str, object] | None = None
    timing = LifecycleTimingReducer()
    for parsed in store.newest_first():
        if not isinstance(parsed, KnownParsedMessage):
            continue
        payload = parsed.message
        if payload.get("type") == "flow.metadata":
            candidate = payload.get("metadata")
            if (
                metadata is None
                and isinstance(candidate, Mapping)
                and flow_id_of(candidate) == target_flow_id
            ):
                metadata = candidate
            continue
        if payload.get("type") != "flow.lifecycle" or payload.get("flow_id") != target_flow_id:
            continue
        timing.add(payload)
    if metadata is None:
        return None
    return grid_flow_with_timing(metadata, *timing.values(target_flow_id))


def grid_flow_with_timing(
    metadata: Mapping[str, object],
    started_at: str | None,
    ended_at: str | None,
) -> PlainJsonObject:
    """Enrich and redact one flow while preserving supplied lifecycle timing."""

    copied = enriched_flow(metadata, started_at=started_at, ended_at=ended_at)
    for side in ("request_body", "response_body"):
        if side in copied:
            copied[side] = redacted_body_descriptor(copied[side])
    return copied


def diff_grid_changes(
    published: Mapping[str, PlainJsonObject],
    current: Mapping[str, PlainJsonObject],
) -> list[PlainJsonObject]:
    """Return protocol delta changes that move ``published`` to ``current``."""

    changes: list[PlainJsonObject] = []
    for flow_id in published:
        if flow_id not in current:
            changes.append({"op": "remove", "flow_id": flow_id})
    for flow_id, flow in current.items():
        if published.get(flow_id) != flow:
            changes.append({"op": "upsert", "flow": flow})
    return changes

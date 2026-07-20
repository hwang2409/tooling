"""Body-redacted grid projection of retained flow metadata.

The browser grid stream (``browser.snapshot``/``browser.delta``) must never
carry captured body bytes; bodies are delivered only through the explicit
per-flow selection endpoint.  The projection keeps every descriptor
schema-valid while stripping the base64 prefix data.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, field

from mitm_inspector.api.bodies import (
    DecodedBodyParts,
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


@dataclass(slots=True)
class FlowProjectionCache:
    """Memoize one flow's projection while its retained metadata is unchanged."""

    metadata: Mapping[str, object] | None = None
    flow: PlainJsonObject | None = None
    _body_parts: dict[
        str,
        tuple[tuple[object, object, object, object, object], str | None, DecodedBodyParts],
    ] = field(default_factory=dict)

    def project(
        self,
        metadata: Mapping[str, object],
        *,
        started_at: str | None,
        ended_at: str | None,
    ) -> PlainJsonObject:
        if self.metadata is not None and self.flow is not None and self._matches(metadata):
            copied = dict(self.flow)
            if started_at is not None:
                copied["started_at"] = started_at
            else:
                copied.pop("started_at", None)
            if ended_at is not None:
                copied["ended_at"] = ended_at
            else:
                copied.pop("ended_at", None)
            return copied
        self.metadata = metadata
        self.flow = enriched_flow(
            metadata,
            started_at=started_at,
            ended_at=ended_at,
            body_cache=self,
        )
        return dict(self.flow)

    def body_parts(
        self, side: str, descriptor: object, encoding: str | None
    ) -> DecodedBodyParts:
        cached = self._body_parts.get(side)
        if (
            cached is not None
            and _same_body_version(cached[0], _body_version(descriptor))
            and cached[1] == encoding
        ):
            return cached[2]
        parts = decoded_body_parts(descriptor, encoding)
        self._body_parts[side] = (_body_version(descriptor), encoding, parts)
        return parts

    def _matches(self, metadata: Mapping[str, object]) -> bool:
        if self.metadata is None or self.metadata.keys() != metadata.keys():
            return False
        for key, value in metadata.items():
            previous = self.metadata[key]
            if key in {"request_body", "response_body"}:
                if not _same_body_version(_body_version(previous), _body_version(value)):
                    return False
            elif previous != value:
                return False
        return True


def _body_version(descriptor: object) -> tuple[object, object, object, object, object]:
    if not isinstance(descriptor, Mapping):
        return (None, None, None, descriptor, None)
    return (
        descriptor.get("state"),
        descriptor.get("size_bytes"),
        descriptor.get("captured_bytes"),
        descriptor.get("data"),
        descriptor.get("content_type"),
    )


def _same_body_version(
    left: tuple[object, object, object, object, object],
    right: tuple[object, object, object, object, object],
) -> bool:
    return (
        left[0] == right[0]
        and left[1] == right[1]
        and left[2] == right[2]
        and left[3] is right[3]
        and left[4] == right[4]
    )


@dataclass(slots=True)
class LifecycleTimingReducer:
    """Sequence-aware lifecycle timing shared by every API projection."""

    _started: dict[str, tuple[int, str]] = field(default_factory=dict)
    _ended: dict[str, tuple[int, str]] = field(default_factory=dict)

    def add(self, payload: Mapping[str, object]) -> None:
        if payload.get("type") != "flow.lifecycle":
            return
        flow_id = payload.get("flow_id")
        sequence = payload.get("sequence")
        occurred_at = payload.get("occurred_at")
        state = payload.get("state")
        if not (
            isinstance(flow_id, str)
            and isinstance(sequence, str)
            and isinstance(occurred_at, str)
            and is_rfc3339_utc(occurred_at)
        ):
            return
        candidate = (int(sequence), occurred_at)
        if state == "request_started" and (
            flow_id not in self._started or candidate[0] < self._started[flow_id][0]
        ):
            self._started[flow_id] = candidate
        if state in {"flow_completed", "error"} and (
            flow_id not in self._ended or candidate[0] > self._ended[flow_id][0]
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
    body_cache: FlowProjectionCache | None = None,
) -> PlainJsonObject:
    """Copy metadata and add decoded bodies, wire facts, and a summary."""

    copied = canonicalize_json(metadata, label="metadata")
    if not isinstance(copied, dict):
        raise ValueError("flow metadata must be an object")
    session_id = copied.get("session_id")
    copied["session_id"] = session_id if isinstance(session_id, str) else None
    if started_at is not None:
        copied["started_at"] = started_at
    if ended_at is not None:
        copied["ended_at"] = ended_at
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
            parts = (
                body_cache.body_parts(side, descriptor, encoding)
                if body_cache is not None
                else decoded_body_parts(descriptor, encoding)
            )
            copied[body_key] = parts.descriptor
            if parts.data is not None:
                decoded_bodies[side] = parts.data
            was_decoded = parts.was_decoded
            if was_decoded and encoding is not None:
                decoded_encodings[side] = encoding
    if decoded_encodings:
        copied["content_encoding"] = decoded_encodings
    else:
        copied.pop("content_encoding", None)
    copied["summary"] = flow_summary(copied, decoded_bodies=decoded_bodies)
    return copied


def flow_id_of(metadata: Mapping[str, object]) -> str | None:
    flow_id = metadata.get("flow_id")
    return flow_id if isinstance(flow_id, str) and flow_id else None


def collect_grid_flows(
    store: MemoryStore,
    *,
    projection_caches: MutableMapping[str, FlowProjectionCache] | None = None,
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
    current: dict[str, PlainJsonObject] = {}
    for flow_id in flow_order:
        selected_metadata = newest.get(flow_id)
        if selected_metadata is None:
            continue
        cache = (
            projection_caches.setdefault(flow_id, FlowProjectionCache())
            if projection_caches is not None
            else None
        )
        current[flow_id] = grid_flow_with_timing(
            selected_metadata,
            *timing.values(flow_id),
            projection_cache=cache,
        )
    return current


def collect_grid_flow(
    store: MemoryStore,
    target_flow_id: str,
    *,
    projection_cache: FlowProjectionCache | None = None,
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
    return grid_flow_with_timing(
        metadata,
        *timing.values(target_flow_id),
        projection_cache=projection_cache,
    )


def grid_flow_with_timing(
    metadata: Mapping[str, object],
    started_at: str | None,
    ended_at: str | None,
    *,
    projection_cache: FlowProjectionCache | None = None,
) -> PlainJsonObject:
    """Enrich and redact one flow while preserving supplied lifecycle timing."""

    copied = (
        projection_cache.project(
            metadata,
            started_at=started_at,
            ended_at=ended_at,
        )
        if projection_cache is not None
        else enriched_flow(metadata, started_at=started_at, ended_at=ended_at)
    )
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

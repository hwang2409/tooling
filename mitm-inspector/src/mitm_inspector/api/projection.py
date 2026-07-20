"""Body-redacted grid projection of retained flow metadata.

The browser grid stream (``browser.snapshot``/``browser.delta``) must never
carry captured body bytes; bodies are delivered only through the explicit
per-flow selection endpoint.  The projection keeps every descriptor
schema-valid while stripping the base64 prefix data.
"""

from __future__ import annotations

from collections.abc import Mapping

from mitm_inspector.api.bodies import body_content_encoding, decoded_body_descriptor
from mitm_inspector.api.summary import flow_summary
from mitm_inspector.json_boundary import (
    PlainJsonObject,
    PlainJsonValue,
    canonicalize_json,
)
from mitm_inspector.protocol import KnownParsedMessage, is_rfc3339_utc
from mitm_inspector.store.memory import MemoryStore

_DATA_BEARING_STATES = frozenset({"captured", "truncated"})


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
    summary = flow_summary(copied)
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
            decoded, was_decoded = decoded_body_descriptor(descriptor, encoding)
            copied[body_key] = decoded
            if was_decoded and encoding is not None:
                decoded_encodings[side] = encoding
    if decoded_encodings:
        copied["content_encoding"] = decoded_encodings
    else:
        copied.pop("content_encoding", None)
    copied["summary"] = summary
    return copied


def flow_id_of(metadata: Mapping[str, object]) -> str | None:
    flow_id = metadata.get("flow_id")
    return flow_id if isinstance(flow_id, str) and flow_id else None


def collect_grid_flows(store: MemoryStore) -> dict[str, PlainJsonObject]:
    """Project newest metadata plus retained lifecycle timing per flow."""

    newest: dict[str, Mapping[str, object]] = {}
    started: dict[str, tuple[int, str]] = {}
    ended: dict[str, tuple[int, str]] = {}
    for parsed in store.newest_first():
        if not isinstance(parsed, KnownParsedMessage):
            continue
        payload = parsed.message
        message_type = payload.get("type")
        if message_type == "flow.lifecycle":
            flow_id = payload.get("flow_id")
            sequence = payload.get("sequence")
            occurred_at = payload.get("occurred_at")
            state = payload.get("state")
            if not (
                isinstance(flow_id, str)
                and isinstance(sequence, str)
                and isinstance(occurred_at, str)
            ):
                continue
            if not is_rfc3339_utc(occurred_at):
                continue
            candidate = (int(sequence), occurred_at)
            if state == "request_started" and (
                flow_id not in started or candidate[0] < started[flow_id][0]
            ):
                started[flow_id] = candidate
            if state in {"flow_completed", "error"} and (
                flow_id not in ended or candidate[0] > ended[flow_id][0]
            ):
                ended[flow_id] = candidate
            continue
        if message_type != "flow.metadata":
            continue
        metadata = payload.get("metadata")
        if not isinstance(metadata, Mapping):
            continue
        flow_id = flow_id_of(metadata)
        if flow_id is None or flow_id in newest:
            continue
        newest[flow_id] = metadata
    return {
        flow_id: grid_flow_with_timing(
            metadata,
            started_at=started[flow_id][1] if flow_id in started else None,
            ended_at=ended[flow_id][1] if flow_id in ended else None,
        )
        for flow_id, metadata in newest.items()
    }


def collect_grid_flow(store: MemoryStore, target_flow_id: str) -> PlainJsonObject | None:
    """Project one retained flow without rebuilding the full grid."""

    metadata: Mapping[str, object] | None = None
    started: tuple[int, str] | None = None
    ended: tuple[int, str] | None = None
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
        sequence = payload.get("sequence")
        occurred_at = payload.get("occurred_at")
        state = payload.get("state")
        if not isinstance(sequence, str) or not isinstance(occurred_at, str):
            continue
        if not is_rfc3339_utc(occurred_at):
            continue
        candidate_time = (int(sequence), occurred_at)
        if state == "request_started" and (
            started is None or candidate_time[0] < started[0]
        ):
            started = candidate_time
        if state in {"flow_completed", "error"} and (
            ended is None or candidate_time[0] > ended[0]
        ):
            ended = candidate_time
    if metadata is None:
        return None
    return grid_flow_with_timing(
        metadata,
        started_at=started[1] if started is not None else None,
        ended_at=ended[1] if ended is not None else None,
    )


def grid_flow_with_timing(
    metadata: Mapping[str, object],
    *,
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

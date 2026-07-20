"""Body-redacted grid projection of retained flow metadata.

The browser grid stream (``browser.snapshot``/``browser.delta``) must never
carry captured body bytes; bodies are delivered only through the explicit
per-flow selection endpoint.  The projection keeps every descriptor
schema-valid while stripping the base64 prefix data.
"""

from __future__ import annotations

from collections.abc import Mapping

from mitm_inspector.json_boundary import (
    PlainJsonObject,
    PlainJsonValue,
    canonicalize_json,
)
from mitm_inspector.protocol import KnownParsedMessage
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

    copied = canonicalize_json(metadata, label="metadata")
    if not isinstance(copied, dict):
        raise ValueError("flow metadata must be an object")
    session_id = copied.get("session_id")
    copied["session_id"] = session_id if isinstance(session_id, str) else None
    for side in ("request_body", "response_body"):
        if side in copied:
            copied[side] = redacted_body_descriptor(copied[side])
    return copied


def flow_id_of(metadata: Mapping[str, object]) -> str | None:
    flow_id = metadata.get("flow_id")
    return flow_id if isinstance(flow_id, str) and flow_id else None


def collect_grid_flows(store: MemoryStore) -> dict[str, PlainJsonObject]:
    """Project the store's newest metadata per flow, newest flow first."""

    newest: dict[str, PlainJsonObject] = {}
    for parsed in store.newest_first():
        if not isinstance(parsed, KnownParsedMessage):
            continue
        payload = parsed.message
        if payload.get("type") != "flow.metadata":
            continue
        metadata = payload.get("metadata")
        if not isinstance(metadata, Mapping):
            continue
        flow_id = flow_id_of(metadata)
        if flow_id is None or flow_id in newest:
            continue
        newest[flow_id] = grid_flow(metadata)
    return newest


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

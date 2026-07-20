"""Shared bounded message metrics for capture sinks and memory retention."""

from __future__ import annotations

import base64
from collections.abc import Mapping

from mitm_inspector.protocol import MAX_U64


def validate_bounded_numbers(value: object) -> None:
    """Reject arbitrary Python integers before copying or retaining input."""

    if type(value) is int:
        if value < -(1 << 63) or value > MAX_U64:
            raise ValueError("integer exceeds the bounded protocol numeric range")
        return
    if isinstance(value, Mapping):
        for item in value.values():
            validate_bounded_numbers(item)
        return
    if isinstance(value, list | tuple):
        for item in value:
            validate_bounded_numbers(item)


def canonical_weight(value: object) -> int:
    """Count canonical retained data, including additive nested fields."""

    if value is None or isinstance(value, bool):
        return 1
    if isinstance(value, int | float):
        if type(value) is int and (value < -(1 << 63) or value > MAX_U64):
            raise ValueError("integer exceeds the bounded protocol numeric range")
        return 8
    if isinstance(value, str):
        return len(value)
    if isinstance(value, Mapping):
        return 8 + sum(len(key) + canonical_weight(item) for key, item in value.items())
    if isinstance(value, list | tuple):
        return 8 + sum(canonical_weight(item) for item in value)
    return 0


def body_bytes(payload: Mapping[str, object]) -> int:
    """Count decoded body-prefix bytes in any supported message envelope."""

    message_type = payload.get("type")
    if message_type == "body.chunk":
        return _decoded_length(payload.get("data_base64"))
    if message_type == "body.end":
        return _descriptor_bytes(payload.get("body"))
    if message_type == "flow.metadata":
        return _flow_metadata_body_bytes(payload.get("metadata"))
    if message_type == "browser.snapshot":
        return sum(_flow_metadata_body_bytes(flow) for flow in _mappings(payload.get("flows")))
    if message_type == "browser.delta":
        return sum(
            _flow_metadata_body_bytes(change.get("flow"))
            for change in _mappings(payload.get("changes"))
            if change.get("op") == "upsert"
        )
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

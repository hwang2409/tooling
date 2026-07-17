"""Runtime validation for protocol-v1 messages.

This module deliberately uses project-owned dictionaries and TypedDicts rather
than mitmproxy objects. Unknown message types and additive fields are retained
for forward compatibility; known required fields are checked at the boundary.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal, NotRequired, TypedDict, cast

PROTOCOL_VERSION = "1"
_DECIMAL_STRING = re.compile(r"^(0|[1-9][0-9]*)$")


class ProtocolError(ValueError):
    """Raised when a message cannot be accepted at the protocol boundary."""


class Header(TypedDict):
    name: str
    value: str


class BodyDescriptor(TypedDict):
    state: Literal["missing", "empty", "captured", "truncated"]
    size_bytes: NotRequired[str]
    captured_bytes: NotRequired[str]
    content_type: NotRequired[str]
    encoding: NotRequired[Literal["base64"]]
    data: NotRequired[str]


class FlowMetadata(TypedDict):
    flow_id: str
    method: str
    scheme: str
    host: str
    port: str
    path: str
    request_headers: list[Header]
    response_headers: NotRequired[list[Header]]
    request_body: BodyDescriptor
    response_body: NotRequired[BodyDescriptor]


class ProtocolMessage(TypedDict):
    protocol_version: str
    type: str


def _object(value: object, *, label: str = "message") -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{label} must be an object")
    return dict(value)


def _string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProtocolError(f"{label} must be a non-empty string")
    return value


def _decimal_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProtocolError(f"{label} must be an unsigned decimal string")
    candidate = value
    if not _DECIMAL_STRING.fullmatch(candidate):
        raise ProtocolError(f"{label} must be an unsigned decimal string")
    return candidate


def _headers(value: object, *, label: str) -> list[Header]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise ProtocolError(f"{label} must be an ordered list")
    result: list[Header] = []
    for index, item in enumerate(value):
        header = _object(item, label=f"{label}[{index}]")
        result.append(
            Header(
                name=_string(header.get("name"), label=f"{label}[{index}].name"),
                value=_string(header.get("value"), label=f"{label}[{index}].value"),
            )
        )
    return result


def _body(value: object, *, label: str) -> BodyDescriptor:
    body = _object(value, label=label)
    state = body.get("state")
    if state not in {"missing", "empty", "captured", "truncated"}:
        raise ProtocolError(f"{label}.state is not a supported body state")
    if state in {"empty", "captured", "truncated"}:
        _decimal_string(body.get("size_bytes"), label=f"{label}.size_bytes")
    if state == "truncated":
        _decimal_string(body.get("captured_bytes"), label=f"{label}.captured_bytes")
    if state == "captured" or state == "truncated":
        _string(body.get("encoding"), label=f"{label}.encoding")
        _string(body.get("data"), label=f"{label}.data")
    return cast(BodyDescriptor, body)


def _flow_metadata(value: object, *, label: str = "metadata") -> FlowMetadata:
    metadata = _object(value, label=label)
    result = cast(FlowMetadata, metadata)
    for key in ("flow_id", "method", "scheme", "host", "path"):
        _string(metadata.get(key), label=f"{label}.{key}")
    _decimal_string(metadata.get("port"), label=f"{label}.port")
    result["request_headers"] = _headers(
        metadata.get("request_headers"), label=f"{label}.request_headers"
    )
    result["request_body"] = _body(metadata.get("request_body"), label=f"{label}.request_body")
    if "response_headers" in metadata:
        result["response_headers"] = _headers(
            metadata["response_headers"], label=f"{label}.response_headers"
        )
    if "response_body" in metadata:
        result["response_body"] = _body(metadata["response_body"], label=f"{label}.response_body")
    return result


def _base(message: dict[str, Any]) -> None:
    if message.get("protocol_version") != PROTOCOL_VERSION:
        raise ProtocolError(f"protocol_version must be {PROTOCOL_VERSION!r}")
    _string(message.get("type"), label="type")


def parse_message(value: object) -> ProtocolMessage:
    """Validate and return a protocol-v1 message while retaining extra fields.

    The returned object is a shallow copy, so callers can safely retain
    additive fields they do not yet understand.
    """

    message = _object(value)
    _base(message)
    message_type = cast(str, message["type"])

    if message_type == "source.hello":
        _string(message.get("source_id"), label="source_id")
        _string(message.get("occurred_at"), label="occurred_at")
    elif message_type == "flow.metadata":
        _flow_metadata(message.get("metadata"))
    elif message_type == "flow.lifecycle":
        _string(message.get("source_id"), label="source_id")
        _string(message.get("flow_id"), label="flow_id")
        _string(message.get("event_id"), label="event_id")
        _string(message.get("occurred_at"), label="occurred_at")
        _decimal_string(message.get("sequence"), label="sequence")
        _string(message.get("state"), label="state")
    elif message_type == "body.chunk":
        _string(message.get("flow_id"), label="flow_id")
        _string(message.get("body_side"), label="body_side")
        _decimal_string(message.get("chunk_index"), label="chunk_index")
        _decimal_string(message.get("offset_bytes"), label="offset_bytes")
        _string(message.get("data_base64"), label="data_base64")
    elif message_type == "body.end":
        _string(message.get("flow_id"), label="flow_id")
        _string(message.get("body_side"), label="body_side")
        _decimal_string(message.get("total_bytes"), label="total_bytes")
        _body(message.get("body"), label="body")
    elif message_type == "stream.gap":
        _decimal_string(message.get("expected_sequence"), label="expected_sequence")
        _decimal_string(message.get("actual_sequence"), label="actual_sequence")
    elif message_type == "browser.snapshot":
        _string(message.get("snapshot_id"), label="snapshot_id")
        _decimal_string(message.get("cursor"), label="cursor")
        flows = message.get("flows")
        if not isinstance(flows, Sequence) or isinstance(flows, str | bytes | bytearray):
            raise ProtocolError("flows must be an ordered list")
        for index, flow in enumerate(flows):
            _flow_metadata(flow, label=f"flows[{index}]")
    elif message_type == "browser.delta":
        _decimal_string(message.get("cursor"), label="cursor")
        changes = message.get("changes")
        if not isinstance(changes, Sequence) or isinstance(changes, str | bytes | bytearray):
            raise ProtocolError("changes must be an ordered list")
        for index, change in enumerate(changes):
            item = _object(change, label=f"changes[{index}]")
            operation = _string(item.get("op"), label=f"changes[{index}].op")
            if operation == "upsert":
                _flow_metadata(item.get("flow"), label=f"changes[{index}].flow")
            elif operation == "remove":
                _string(item.get("flow_id"), label=f"changes[{index}].flow_id")
            else:
                raise ProtocolError(f"changes[{index}].op is not supported")
    elif message_type == "browser.resync":
        _string(message.get("reason"), label="reason")
        _decimal_string(message.get("requested_cursor"), label="requested_cursor")

    return cast(ProtocolMessage, message)

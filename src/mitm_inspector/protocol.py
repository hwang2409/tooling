"""Authoritative runtime validation for protocol-v1 messages.

Known messages have discriminated TypedDict models and explicit invariants.
Unknown types and additive fields are retained, but malformed known messages
are rejected before they reach the capture/store/API boundaries.
"""

from __future__ import annotations

import base64
import math
import re
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Literal, NotRequired, TypedDict, cast, final

PROTOCOL_VERSION = "1"
MAX_U64 = 18_446_744_073_709_551_615
_U64_PATTERN = re.compile(r"^(0|[1-9][0-9]*)$")
_BASE64_PATTERN = re.compile(r"^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$")
BODY_SIDES = ("request", "response")
LIFECYCLE_STATES = (
    "request_started",
    "request_headers",
    "request_body",
    "request_end",
    "response_started",
    "response_headers",
    "response_body",
    "response_end",
    "error",
    "flow_completed",
)
RESYNC_REASONS = ("cursor_gap", "history_evicted", "initial_connect")
KNOWN_MESSAGE_TYPES = frozenset(
    {
        "source.hello",
        "flow.metadata",
        "flow.lifecycle",
        "body.chunk",
        "body.end",
        "stream.gap",
        "browser.snapshot",
        "browser.delta",
        "browser.resync",
    }
)


class ProtocolError(ValueError):
    """Raised when a message cannot be accepted at the protocol boundary."""


class Header(TypedDict):
    name: str
    value: str


LifecycleState = Literal[
    "request_started",
    "request_headers",
    "request_body",
    "request_end",
    "response_started",
    "response_headers",
    "response_body",
    "response_end",
    "error",
    "flow_completed",
]
BodySide = Literal["request", "response"]


class MissingBody(TypedDict):
    state: Literal["missing"]
    content_type: NotRequired[str]


class EmptyBody(TypedDict):
    state: Literal["empty"]
    size_bytes: Literal["0"]
    content_type: NotRequired[str]


class CapturedBody(TypedDict):
    state: Literal["captured"]
    size_bytes: str
    content_type: NotRequired[str]
    encoding: Literal["base64"]
    data: str


class TruncatedBody(TypedDict):
    state: Literal["truncated"]
    size_bytes: str
    captured_bytes: str
    content_type: NotRequired[str]
    encoding: Literal["base64"]
    data: str


BodyDescriptor = MissingBody | EmptyBody | CapturedBody | TruncatedBody


class FlowMetadata(TypedDict):
    flow_id: str
    method: str
    scheme: Literal["http", "https"]
    host: str
    port: str
    path: str
    request_headers: list[Header]
    response_headers: NotRequired[list[Header]]
    request_body: BodyDescriptor
    response_body: NotRequired[BodyDescriptor]


class SourceCapabilities(TypedDict):
    body_chunks: bool
    redaction: Literal["headers-and-query"]


class SourceLimits(TypedDict):
    max_body_prefix_bytes: str
    max_in_memory_bytes: str


class SourceHello(TypedDict):
    protocol_version: Literal["1"]
    type: Literal["source.hello"]
    source_id: str
    occurred_at: str
    capabilities: SourceCapabilities
    limits: SourceLimits


class FlowMetadataMessage(TypedDict):
    protocol_version: Literal["1"]
    type: Literal["flow.metadata"]
    metadata: FlowMetadata


class FlowLifecycle(TypedDict):
    protocol_version: Literal["1"]
    type: Literal["flow.lifecycle"]
    source_id: str
    flow_id: str
    event_id: str
    occurred_at: str
    sequence: str
    state: LifecycleState


class BodyChunk(TypedDict):
    protocol_version: Literal["1"]
    type: Literal["body.chunk"]
    flow_id: str
    body_side: BodySide
    chunk_index: str
    offset_bytes: str
    data_base64: str


class BodyEnd(TypedDict):
    protocol_version: Literal["1"]
    type: Literal["body.end"]
    flow_id: str
    body_side: BodySide
    total_bytes: str
    body: BodyDescriptor


class StreamGap(TypedDict):
    protocol_version: Literal["1"]
    type: Literal["stream.gap"]
    expected_sequence: str
    actual_sequence: str
    dropped_count: NotRequired[str]


class BrowserSnapshot(TypedDict):
    protocol_version: Literal["1"]
    type: Literal["browser.snapshot"]
    snapshot_id: str
    cursor: str
    flows: list[FlowMetadata]


class UpsertChange(TypedDict):
    op: Literal["upsert"]
    flow: FlowMetadata


class RemoveChange(TypedDict):
    op: Literal["remove"]
    flow_id: str


DeltaChange = UpsertChange | RemoveChange


class BrowserDelta(TypedDict):
    protocol_version: Literal["1"]
    type: Literal["browser.delta"]
    cursor: str
    changes: list[DeltaChange]


class BrowserResync(TypedDict):
    protocol_version: Literal["1"]
    type: Literal["browser.resync"]
    reason: Literal["cursor_gap", "history_evicted", "initial_connect"]
    requested_cursor: str


KnownMessage = (
    SourceHello
    | FlowMetadataMessage
    | FlowLifecycle
    | BodyChunk
    | BodyEnd
    | StreamGap
    | BrowserSnapshot
    | BrowserDelta
    | BrowserResync
)


type FrozenJsonValue = (
    None
    | bool
    | int
    | float
    | str
    | tuple[FrozenJsonValue, ...]
    | Mapping[str, FrozenJsonValue]
)
type FrozenJsonObject = Mapping[str, FrozenJsonValue]
type PlainJsonValue = (
    None | bool | int | float | str | list[PlainJsonValue] | dict[str, PlainJsonValue]
)
type PlainJsonObject = dict[str, PlainJsonValue]
_PARSE_TOKEN = object()


class ParsedMessage:
    """Nominal base for immutable values created by :func:`parse_message`."""

    __slots__ = ()

    def __new__(cls, *_args: object, **_kwargs: object) -> ParsedMessage:
        if cls is ParsedMessage:
            raise TypeError("ParsedMessage values must be created by parse_message")
        return super().__new__(cls)


@final
class KnownParsedMessage(ParsedMessage):
    """Recursively immutable validated message from the known vocabulary."""

    __slots__ = ("_message",)
    _message: FrozenJsonObject
    kind: Literal["known"] = "known"

    def __init__(self, message: FrozenJsonObject, *, _token: object) -> None:
        if _token is not _PARSE_TOKEN:
            raise TypeError("KnownParsedMessage values must be created by parse_message")
        message_type = message.get("type")
        if not isinstance(message_type, str) or message_type not in KNOWN_MESSAGE_TYPES:
            raise ProtocolError("known parsed message must use a known type")
        object.__setattr__(self, "_message", message)

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("parsed messages are immutable")

    @property
    def message(self) -> FrozenJsonObject:
        return self._message


@final
class OpaqueParsedMessage(ParsedMessage):
    """Recursively immutable validated message outside the known vocabulary."""

    __slots__ = ("_original_type", "_payload")
    _original_type: str
    _payload: FrozenJsonObject
    kind: Literal["unknown"] = "unknown"

    def __init__(
        self,
        original_type: str,
        payload: FrozenJsonObject,
        *,
        _token: object,
    ) -> None:
        if _token is not _PARSE_TOKEN:
            raise TypeError("OpaqueParsedMessage values must be created by parse_message")
        if original_type in KNOWN_MESSAGE_TYPES:
            raise ProtocolError("opaque parsed message cannot use a known type")
        if payload.get("type") != original_type:
            raise ProtocolError("opaque original_type must equal payload.type")
        object.__setattr__(self, "_original_type", original_type)
        object.__setattr__(self, "_payload", payload)

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("parsed messages are immutable")

    @property
    def original_type(self) -> str:
        return self._original_type

    @property
    def payload(self) -> FrozenJsonObject:
        return self._payload


type ParsedMessageResult = KnownParsedMessage | OpaqueParsedMessage


def require_parsed_message(value: object) -> ParsedMessageResult:
    """Copy and fully revalidate an untrusted parsed wrapper at ingress."""

    if isinstance(value, KnownParsedMessage):
        try:
            supplied_payload = object.__getattribute__(value, "_message")
        except AttributeError as error:
            raise ProtocolError("known parsed wrapper has no message payload") from error
        payload = _copy_plain_object(supplied_payload, label="message")
        reparsed = parse_message(payload)
        if not isinstance(reparsed, KnownParsedMessage):
            raise ProtocolError("known parsed wrapper must contain a known message type")
        return reparsed

    if isinstance(value, OpaqueParsedMessage):
        try:
            original_type = object.__getattribute__(value, "_original_type")
            supplied_payload = object.__getattribute__(value, "_payload")
        except AttributeError as error:
            raise ProtocolError("opaque parsed wrapper is incomplete") from error
        if not isinstance(original_type, str) or not original_type:
            raise ProtocolError("opaque original_type must be a non-empty string")
        payload = _copy_plain_object(supplied_payload, label="payload")
        if original_type in KNOWN_MESSAGE_TYPES:
            raise ProtocolError("opaque parsed wrapper cannot use a known type")
        if payload.get("type") != original_type:
            raise ProtocolError("opaque original_type must equal payload.type")
        reparsed = parse_message(payload)
        if not isinstance(reparsed, OpaqueParsedMessage):
            raise ProtocolError("opaque parsed wrapper must contain an unknown message type")
        return reparsed

    raise ProtocolError("message must be a parsed wrapper")


def parsed_message_to_plain_json(value: object) -> PlainJsonObject:
    """Revalidate a wrapper and return an independent JSON wire message."""

    canonical = require_parsed_message(value)
    if isinstance(canonical, KnownParsedMessage):
        return _copy_plain_object(canonical.message, label="message")
    return _copy_plain_object(canonical.payload, label="payload")


def _copy_plain_json(
    value: object,
    *,
    label: str,
    ancestors: set[int] | None = None,
) -> PlainJsonValue:
    """Copy JSON-shaped containers to plain mutable dict/list wire values."""

    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProtocolError(f"{label} must contain finite JSON numbers")
        return value
    if ancestors is None:
        ancestors = set()
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in ancestors:
            raise ProtocolError(f"{label} must not contain cycles")
        ancestors.add(identity)
        try:
            copied: dict[str, PlainJsonValue] = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    raise ProtocolError(f"{label} object keys must be strings")
                copied[key] = _copy_plain_json(
                    item,
                    label=f"{label}.{key}",
                    ancestors=ancestors,
                )
            return copied
        finally:
            ancestors.remove(identity)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        identity = id(value)
        if identity in ancestors:
            raise ProtocolError(f"{label} must not contain cycles")
        ancestors.add(identity)
        try:
            return [
                _copy_plain_json(item, label=f"{label}[{index}]", ancestors=ancestors)
                for index, item in enumerate(value)
            ]
        finally:
            ancestors.remove(identity)
    raise ProtocolError(f"{label} must contain only JSON values")


def _copy_plain_object(value: object, *, label: str) -> PlainJsonObject:
    copied = _copy_plain_json(value, label=label)
    if not isinstance(copied, dict):
        raise ProtocolError(f"{label} must be an object")
    return copied


def _freeze_json(
    value: object,
    *,
    label: str,
    ancestors: set[int] | None = None,
) -> FrozenJsonValue:
    """Copy JSON-shaped input into recursively immutable canonical containers."""

    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProtocolError(f"{label} must contain finite JSON numbers")
        return value
    if ancestors is None:
        ancestors = set()
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in ancestors:
            raise ProtocolError(f"{label} must not contain cycles")
        ancestors.add(identity)
        try:
            frozen: dict[str, FrozenJsonValue] = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    raise ProtocolError(f"{label} object keys must be strings")
                frozen[key] = _freeze_json(item, label=f"{label}.{key}", ancestors=ancestors)
            return MappingProxyType(frozen)
        finally:
            ancestors.remove(identity)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        identity = id(value)
        if identity in ancestors:
            raise ProtocolError(f"{label} must not contain cycles")
        ancestors.add(identity)
        try:
            return tuple(
                _freeze_json(item, label=f"{label}[{index}]", ancestors=ancestors)
                for index, item in enumerate(value)
            )
        finally:
            ancestors.remove(identity)
    raise ProtocolError(f"{label} must contain only JSON values")


def _freeze_object(value: Mapping[str, object], *, label: str = "message") -> FrozenJsonObject:
    frozen = _freeze_json(value, label=label)
    if not isinstance(frozen, Mapping):
        raise AssertionError("object canonicalization returned a non-object")
    return frozen


def _object(value: object, *, label: str = "message") -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{label} must be an object")
    return dict(value)


def _string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProtocolError(f"{label} must be a non-empty string")
    return value


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise ProtocolError(f"{label} must be a string")
    return value


def _u64(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not _U64_PATTERN.fullmatch(value):
        raise ProtocolError(f"{label} must be a uint64 decimal string")
    if int(value) > MAX_U64:
        raise ProtocolError(f"{label} exceeds uint64")
    return value


def _enum(value: object, choices: Sequence[str], *, label: str) -> str:
    candidate = _string(value, label=label)
    if candidate not in choices:
        raise ProtocolError(f"{label} is not supported")
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
                value=_text(header.get("value"), label=f"{label}[{index}].value"),
            )
        )
    return result


def _base64_bytes(value: object, *, label: str) -> bytes:
    data = _text(value, label=label)
    if not _BASE64_PATTERN.fullmatch(data):
        raise ProtocolError(f"{label} must be valid base64")
    try:
        return base64.b64decode(data, validate=True)
    except ValueError as error:
        raise ProtocolError(f"{label} must be valid base64") from error


def _body(value: object, *, label: str) -> BodyDescriptor:
    body = _object(value, label=label)
    state = _enum(
        body.get("state"),
        ("missing", "empty", "captured", "truncated"),
        label=f"{label}.state",
    )
    if "content_type" in body:
        _text(body["content_type"], label=f"{label}.content_type")
    if state == "missing":
        if any(key in body for key in ("size_bytes", "captured_bytes", "encoding", "data")):
            raise ProtocolError(f"{label} missing state cannot carry body counts or data")
        return cast(MissingBody, body)
    size = _u64(body.get("size_bytes"), label=f"{label}.size_bytes")
    if state == "empty":
        if size != "0" or any(key in body for key in ("captured_bytes", "encoding", "data")):
            raise ProtocolError(f"{label} empty state must have only size_bytes=0")
        return cast(EmptyBody, body)
    if "captured_bytes" in body and state == "captured":
        raise ProtocolError(f"{label} captured state cannot carry captured_bytes")
    if body.get("encoding") != "base64":
        raise ProtocolError(f"{label}.encoding must be base64")
    decoded = _base64_bytes(body.get("data"), label=f"{label}.data")
    if state == "captured":
        if len(decoded) != int(size):
            raise ProtocolError(f"{label}.data length does not equal size_bytes")
        return cast(CapturedBody, body)
    captured = _u64(body.get("captured_bytes"), label=f"{label}.captured_bytes")
    if int(captured) > int(size) or len(decoded) > int(captured):
        raise ProtocolError(f"{label} truncated prefix exceeds declared counts")
    return cast(TruncatedBody, body)


def _flow_metadata(value: object, *, label: str = "metadata") -> FlowMetadata:
    metadata = _object(value, label=label)
    scheme = _enum(metadata.get("scheme"), ("http", "https"), label=f"{label}.scheme")
    for key in ("flow_id", "method", "host", "path"):
        _string(metadata.get(key), label=f"{label}.{key}")
    port = _u64(metadata.get("port"), label=f"{label}.port")
    request_headers = _headers(metadata.get("request_headers"), label=f"{label}.request_headers")
    request_body = _body(metadata.get("request_body"), label=f"{label}.request_body")
    result = cast(FlowMetadata, metadata)
    result["scheme"] = cast(Literal["http", "https"], scheme)
    result["port"] = port
    result["request_headers"] = request_headers
    result["request_body"] = request_body
    if "response_headers" in metadata:
        result["response_headers"] = _headers(
            metadata["response_headers"], label=f"{label}.response_headers"
        )
    if "response_body" in metadata:
        result["response_body"] = _body(metadata["response_body"], label=f"{label}.response_body")
    return result


def _validate_source_hello(message: dict[str, object]) -> None:
    _string(message.get("source_id"), label="source_id")
    _string(message.get("occurred_at"), label="occurred_at")
    capabilities = _object(message.get("capabilities"), label="capabilities")
    if not isinstance(capabilities.get("body_chunks"), bool):
        raise ProtocolError("capabilities.body_chunks must be boolean")
    _enum(capabilities.get("redaction"), ("headers-and-query",), label="capabilities.redaction")
    limits = _object(message.get("limits"), label="limits")
    _u64(limits.get("max_body_prefix_bytes"), label="limits.max_body_prefix_bytes")
    _u64(limits.get("max_in_memory_bytes"), label="limits.max_in_memory_bytes")


def _validate_body_end(message: dict[str, object]) -> None:
    total = _u64(message.get("total_bytes"), label="total_bytes")
    body = _body(message.get("body"), label="body")
    if body["state"] == "missing":
        if total != "0":
            raise ProtocolError("missing body must have total_bytes=0")
    elif body["size_bytes"] != total:
        raise ProtocolError("body.size_bytes must equal total_bytes")


def _validate_gap(message: dict[str, object]) -> None:
    expected = int(_u64(message.get("expected_sequence"), label="expected_sequence"))
    actual = int(_u64(message.get("actual_sequence"), label="actual_sequence"))
    if actual <= expected:
        raise ProtocolError("actual_sequence must be greater than expected_sequence")
    if "dropped_count" in message:
        dropped = int(_u64(message["dropped_count"], label="dropped_count"))
        if dropped != actual - expected - 1:
            raise ProtocolError("dropped_count does not match the sequence gap")


def _base(message: dict[str, object]) -> str:
    if message.get("protocol_version") != PROTOCOL_VERSION:
        raise ProtocolError(f"protocol_version must be {PROTOCOL_VERSION!r}")
    return _string(message.get("type"), label="type")


def parse_message(value: object) -> ParsedMessageResult:
    """Validate, deep-copy, and recursively freeze a raw protocol message."""

    message = _object(value)
    message_type = _base(message)

    if message_type == "source.hello":
        _validate_source_hello(message)
    elif message_type == "flow.metadata":
        _flow_metadata(message.get("metadata"))
    elif message_type == "flow.lifecycle":
        for key in ("source_id", "flow_id", "event_id", "occurred_at"):
            _string(message.get(key), label=key)
        _u64(message.get("sequence"), label="sequence")
        _enum(message.get("state"), LIFECYCLE_STATES, label="state")
    elif message_type == "body.chunk":
        _string(message.get("flow_id"), label="flow_id")
        _enum(message.get("body_side"), BODY_SIDES, label="body_side")
        _u64(message.get("chunk_index"), label="chunk_index")
        _u64(message.get("offset_bytes"), label="offset_bytes")
        _base64_bytes(message.get("data_base64"), label="data_base64")
    elif message_type == "body.end":
        _string(message.get("flow_id"), label="flow_id")
        _enum(message.get("body_side"), BODY_SIDES, label="body_side")
        _validate_body_end(message)
    elif message_type == "stream.gap":
        _validate_gap(message)
    elif message_type == "browser.snapshot":
        _string(message.get("snapshot_id"), label="snapshot_id")
        _u64(message.get("cursor"), label="cursor")
        flows = message.get("flows")
        if not isinstance(flows, Sequence) or isinstance(flows, str | bytes | bytearray):
            raise ProtocolError("flows must be an ordered list")
        for index, flow in enumerate(flows):
            _flow_metadata(flow, label=f"flows[{index}]")
    elif message_type == "browser.delta":
        _u64(message.get("cursor"), label="cursor")
        changes = message.get("changes")
        if not isinstance(changes, Sequence) or isinstance(changes, str | bytes | bytearray):
            raise ProtocolError("changes must be an ordered list")
        for index, change in enumerate(changes):
            item = _object(change, label=f"changes[{index}]")
            operation = _enum(item.get("op"), ("upsert", "remove"), label=f"changes[{index}].op")
            if operation == "upsert":
                _flow_metadata(item.get("flow"), label=f"changes[{index}].flow")
            else:
                _string(item.get("flow_id"), label=f"changes[{index}].flow_id")
    elif message_type == "browser.resync":
        _enum(message.get("reason"), RESYNC_REASONS, label="reason")
        _u64(message.get("requested_cursor"), label="requested_cursor")

    frozen_message = _freeze_object(message)
    if message_type in KNOWN_MESSAGE_TYPES:
        return KnownParsedMessage(frozen_message, _token=_PARSE_TOKEN)
    return OpaqueParsedMessage(message_type, frozen_message, _token=_PARSE_TOKEN)

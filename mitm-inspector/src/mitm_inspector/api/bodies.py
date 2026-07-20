"""Serve-time decoding helpers for retained HTTP body prefixes."""

from __future__ import annotations

import base64
import binascii
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, cast

from mitm_inspector.api.limits import MAX_INGEST_BODY_PREFIX_BYTES
from mitm_inspector.json_boundary import PlainJsonObject, PlainJsonValue

_BODY_STATES_WITH_DATA = frozenset({"captured", "truncated"})
_SUPPORTED_ENCODINGS = frozenset({"gzip", "x-gzip", "deflate"})
_DECOMPRESS_INPUT_CHUNK_BYTES = 64 * 1024
_MAX_CONCATENATED_MEMBERS = 256
MAX_DECODED_BODY_BYTES = MAX_INGEST_BODY_PREFIX_BYTES


@dataclass(frozen=True, slots=True)
class ContentDecodeResult:
    """One bounded content-decoding attempt."""

    data: bytes
    status: Literal["complete", "incomplete", "limit", "invalid"]


@dataclass(frozen=True, slots=True)
class DecodedBodyParts:
    """The served descriptor and decoded bytes from one decode attempt."""

    descriptor: PlainJsonValue
    data: bytes | None
    was_decoded: bool
    data_was_decoded: bool


def header_value(headers: object, name: str) -> str | None:
    """Return the last matching sanitized header value, case-insensitively."""

    if not isinstance(headers, Sequence) or isinstance(headers, str | bytes | bytearray):
        return None
    result: str | None = None
    for header in headers:
        if not isinstance(header, Mapping):
            continue
        header_name = header.get("name")
        value = header.get("value")
        if (
            isinstance(header_name, str)
            and header_name.casefold() == name.casefold()
            and isinstance(value, str)
        ):
            result = value
    return result


def body_content_encoding(metadata: Mapping[str, object], side: str) -> str | None:
    """Return one supported original content encoding for a flow body."""

    headers = metadata.get(f"{side}_headers")
    value = header_value(headers, "content-encoding")
    if value is None:
        return None
    normalized = value.strip().casefold()
    return normalized if normalized in _SUPPORTED_ENCODINGS else None


def decoded_body_bytes(descriptor: object, encoding: str | None) -> tuple[bytes | None, bool]:
    """Decode a descriptor's retained bytes and optional HTTP content coding."""

    parts = decoded_body_parts(descriptor, encoding)
    return parts.data, parts.data_was_decoded


def decoded_body_descriptor(
    descriptor: PlainJsonValue, encoding: str | None
) -> tuple[PlainJsonValue, bool]:
    """Return a descriptor whose prefix is decoded when its coding is supported."""

    parts = decoded_body_parts(descriptor, encoding)
    return parts.descriptor, parts.was_decoded


def decoded_body_parts(descriptor: object, encoding: str | None) -> DecodedBodyParts:
    """Decode a body once and return both projections that need its result."""

    if not isinstance(descriptor, Mapping):
        return DecodedBodyParts(cast(PlainJsonValue, descriptor), None, False, False)
    if descriptor.get("state") not in _BODY_STATES_WITH_DATA:
        return DecodedBodyParts(cast(PlainJsonValue, descriptor), None, False, False)
    data = descriptor.get("data")
    if not isinstance(data, str):
        return DecodedBodyParts(cast(PlainJsonValue, descriptor), None, False, False)
    try:
        raw = base64.b64decode(data, validate=True)
    except (ValueError, binascii.Error):
        return DecodedBodyParts(cast(PlainJsonValue, descriptor), None, False, False)
    if encoding is None:
        return DecodedBodyParts(cast(PlainJsonValue, descriptor), raw, False, False)
    decoded_result = _decode_content(raw, encoding)
    if decoded_result.status == "invalid":
        return DecodedBodyParts(cast(PlainJsonValue, descriptor), None, False, False)
    # A decoded descriptor needs an exact decoded total. Incomplete streams
    # and output-ceiling hits retain their encoded representation instead of
    # inventing size_bytes/total_bytes for an unknowable body.
    if descriptor.get("state") != "captured" or decoded_result.status != "complete":
        return DecodedBodyParts(cast(PlainJsonValue, descriptor), decoded_result.data, False, True)
    decoded = decoded_result.data
    result: PlainJsonObject = {
        "state": "captured",
        "size_bytes": str(len(decoded)),
        "encoding": "base64",
        "data": base64.b64encode(decoded).decode("ascii"),
    }
    content_type = descriptor.get("content_type")
    if isinstance(content_type, str):
        result["content_type"] = content_type
    return DecodedBodyParts(result, decoded, True, True)


def _decode_content(data: bytes, encoding: str) -> ContentDecodeResult:
    if encoding not in _SUPPORTED_ENCODINGS:
        return ContentDecodeResult(b"", "invalid")
    window_bits = 16 + zlib.MAX_WBITS if encoding in {"gzip", "x-gzip"} else zlib.MAX_WBITS
    attempts = (window_bits, -zlib.MAX_WBITS) if encoding == "deflate" else (window_bits,)
    for attempt in attempts:
        result = _decode_members(data, attempt, concatenated=encoding in {"gzip", "x-gzip"})
        if result.status != "invalid":
            return result
    return ContentDecodeResult(b"", "invalid")


def _decode_members(data: bytes, window_bits: int, *, concatenated: bool) -> ContentDecodeResult:
    if not data:
        return ContentDecodeResult(b"", "incomplete")
    output = bytearray()
    remaining_input = data
    member_count = 0
    while remaining_input:
        member_count += 1
        if member_count > _MAX_CONCATENATED_MEMBERS:
            return ContentDecodeResult(bytes(output), "limit")
        member = _decode_member(
            remaining_input,
            window_bits,
            output_limit=MAX_DECODED_BODY_BYTES - len(output),
        )
        output.extend(member.data)
        if member.status != "complete":
            return ContentDecodeResult(bytes(output), member.status)
        remaining_input = member.trailing
        if not concatenated and remaining_input:
            return ContentDecodeResult(bytes(output), "invalid")
    return ContentDecodeResult(bytes(output), "complete")


@dataclass(frozen=True, slots=True)
class _MemberDecodeResult:
    data: bytes
    trailing: bytes
    status: Literal["complete", "incomplete", "limit", "invalid"]


def _decode_member(data: bytes, window_bits: int, *, output_limit: int) -> _MemberDecodeResult:
    decoder = zlib.decompressobj(window_bits)
    output = bytearray()
    offset = 0
    pending = b""
    try:
        while offset < len(data) or pending:
            if not pending:
                end = min(len(data), offset + _DECOMPRESS_INPUT_CHUNK_BYTES)
                pending = data[offset:end]
                offset = end
            remaining = output_limit - len(output)
            decoded = decoder.decompress(pending, remaining + 1)
            if len(decoded) > remaining:
                output.extend(decoded[:remaining])
                return _MemberDecodeResult(bytes(output), b"", "limit")
            output.extend(decoded)
            pending = decoder.unconsumed_tail
            if decoder.eof:
                trailing = decoder.unused_data + pending + data[offset:]
                return _MemberDecodeResult(bytes(output), trailing, "complete")
            if remaining == 0:
                return _MemberDecodeResult(bytes(output), b"", "limit")
    except zlib.error:
        return _MemberDecodeResult(bytes(output), b"", "invalid")
    return _MemberDecodeResult(bytes(output), b"", "incomplete")


__all__ = [
    "body_content_encoding",
    "ContentDecodeResult",
    "DecodedBodyParts",
    "MAX_DECODED_BODY_BYTES",
    "decoded_body_bytes",
    "decoded_body_descriptor",
    "decoded_body_parts",
    "header_value",
]

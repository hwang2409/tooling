"""Serve-time decoding helpers for retained HTTP body prefixes."""

from __future__ import annotations

import base64
import binascii
import zlib
from collections.abc import Mapping, Sequence

from mitm_inspector.json_boundary import PlainJsonObject, PlainJsonValue

_BODY_STATES_WITH_DATA = frozenset({"captured", "truncated"})
_SUPPORTED_ENCODINGS = frozenset({"gzip", "x-gzip", "deflate"})


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

    if not isinstance(descriptor, Mapping):
        return None, False
    if descriptor.get("state") not in _BODY_STATES_WITH_DATA:
        return None, False
    data = descriptor.get("data")
    if not isinstance(data, str):
        return None, False
    try:
        raw = base64.b64decode(data, validate=True)
    except (ValueError, binascii.Error):
        return None, False
    if encoding is None:
        return raw, False
    decoded = _decode_content(raw, encoding)
    if decoded is None:
        return None, False
    return decoded[0], True


def decoded_body_descriptor(
    descriptor: PlainJsonValue, encoding: str | None
) -> tuple[PlainJsonValue, bool]:
    """Return a descriptor whose prefix is decoded when its coding is supported."""

    if not isinstance(descriptor, dict) or descriptor.get("state") not in _BODY_STATES_WITH_DATA:
        return descriptor, False
    if encoding is None:
        return descriptor, False
    data = descriptor.get("data")
    if not isinstance(data, str):
        return descriptor, False
    try:
        raw = base64.b64decode(data, validate=True)
    except (ValueError, binascii.Error):
        return descriptor, False
    decoded_result = _decode_content(raw, encoding)
    if decoded_result is None:
        return descriptor, False
    decoded, complete = decoded_result
    state = "captured" if descriptor.get("state") == "captured" and complete else "truncated"
    result: PlainJsonObject = {
        "state": state,
        "size_bytes": str(len(decoded)),
        "encoding": "base64",
        "data": base64.b64encode(decoded).decode("ascii"),
    }
    if state == "truncated":
        result["captured_bytes"] = str(len(decoded))
    content_type = descriptor.get("content_type")
    if isinstance(content_type, str):
        result["content_type"] = content_type
    return result, True


def _decode_content(data: bytes, encoding: str) -> tuple[bytes, bool] | None:
    if encoding not in _SUPPORTED_ENCODINGS:
        return None
    window_bits = 16 + zlib.MAX_WBITS if encoding in {"gzip", "x-gzip"} else zlib.MAX_WBITS
    attempts = (window_bits, -zlib.MAX_WBITS) if encoding == "deflate" else (window_bits,)
    for attempt in attempts:
        try:
            decoder = zlib.decompressobj(attempt)
            decoded = decoder.decompress(data)
            decoded += decoder.flush()
        except zlib.error:
            continue
        return decoded, decoder.eof
    return None


__all__ = [
    "body_content_encoding",
    "decoded_body_bytes",
    "decoded_body_descriptor",
    "header_value",
]

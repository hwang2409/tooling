"""Bounded HTTP/1.1 request-head parsing and RFC 6455 frame codec.

These are pure byte-level functions with explicit limits so the loopback
server never trusts transport input.  No sockets are opened here.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import ipaddress
import re
from dataclasses import dataclass
from urllib.parse import urlsplit

MAX_REQUEST_HEAD_BYTES = 32 * 1024
MAX_HEADER_COUNT = 100
WEBSOCKET_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
WEBSOCKET_VERSION = "13"

_TOKEN = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")
_TARGET = re.compile(r"^[!-~]+$")
_HEADER_VALUE = re.compile(r"^[\t\x20-\x7e\x80-\xff]*$")
_LOOPBACK_NAMES = frozenset({"localhost"})


class HttpWireError(ValueError):
    """Raised when a request head cannot be accepted."""


class WebSocketWireError(ValueError):
    """Raised when a WebSocket frame stream violates RFC 6455 or a bound."""


@dataclass(frozen=True, slots=True)
class RequestHead:
    method: str
    target: str
    version: str
    headers: tuple[tuple[str, str], ...]

    def header_values(self, name: str) -> tuple[str, ...]:
        lowered = name.lower()
        return tuple(value for key, value in self.headers if key == lowered)

    def single_header(self, name: str) -> str | None:
        values = self.header_values(name)
        if len(values) > 1:
            raise HttpWireError(f"duplicate {name} header")
        return values[0] if values else None

    def token_list(self, name: str) -> tuple[str, ...]:
        tokens: list[str] = []
        for value in self.header_values(name):
            tokens.extend(
                token.strip().lower() for token in value.split(",") if token.strip()
            )
        return tuple(tokens)


def parse_request_head(raw: bytes) -> RequestHead:
    """Parse one bounded request head ending in CRLF CRLF."""

    if type(raw) is not bytes:
        raise HttpWireError("request head must be bytes")
    if len(raw) > MAX_REQUEST_HEAD_BYTES:
        raise HttpWireError("request head exceeds the size limit")
    if not raw.endswith(b"\r\n\r\n"):
        raise HttpWireError("request head must end with CRLF CRLF")
    try:
        text = raw[:-4].decode("ascii")
    except UnicodeDecodeError as error:
        raise HttpWireError("request head must be ASCII") from error
    lines = text.split("\r\n")
    if "\n" in text.replace("\r\n", "") or "\r" in text.replace("\r\n", ""):
        raise HttpWireError("request head contains a bare CR or LF")
    request_line = lines[0]
    parts = request_line.split(" ")
    if len(parts) != 3:
        raise HttpWireError("malformed request line")
    method, target, version = parts
    if not _TOKEN.fullmatch(method):
        raise HttpWireError("malformed request method")
    if not _TARGET.fullmatch(target):
        raise HttpWireError("malformed request target")
    if version not in {"HTTP/1.1", "HTTP/1.0"}:
        raise HttpWireError("unsupported HTTP version")
    headers: list[tuple[str, str]] = []
    for line in lines[1:]:
        if not line:
            raise HttpWireError("empty header line inside request head")
        if line[0] in " \t":
            raise HttpWireError("obsolete header folding is not accepted")
        name, separator, value = line.partition(":")
        if not separator or not _TOKEN.fullmatch(name):
            raise HttpWireError("malformed header name")
        value = value.strip(" \t")
        if not _HEADER_VALUE.fullmatch(value):
            raise HttpWireError("malformed header value")
        headers.append((name.lower(), value))
        if len(headers) > MAX_HEADER_COUNT:
            raise HttpWireError("too many request headers")
    return RequestHead(
        method=method,
        target=target,
        version=version,
        headers=tuple(headers),
    )


def _is_loopback_hostname(hostname: str) -> bool:
    if not hostname:
        return False
    if hostname.lower() in _LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def is_loopback_host_header(value: str) -> bool:
    """Accept only ``loopback-host[:port]`` authority forms."""

    if type(value) is not str or not value or "/" in value or "@" in value:
        return False
    try:
        parsed = urlsplit(f"//{value}")
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        return False
    if parsed.path or parsed.query or parsed.fragment:
        return False
    if hostname is None or not _is_loopback_hostname(hostname):
        return False
    if value.rstrip().endswith(":") or (port is not None and not 1 <= port <= 65535):
        return False
    return True


def is_loopback_origin(value: str) -> bool:
    """Accept only ``http(s)://loopback-host[:port]`` origins."""

    if type(value) is not str or not value:
        return False
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        return False
    if parsed.scheme not in {"http", "https"}:
        return False
    if parsed.path or parsed.query or parsed.fragment:
        return False
    if parsed.username is not None or parsed.password is not None:
        return False
    if hostname is None or not _is_loopback_hostname(hostname):
        return False
    if port is not None and not 1 <= port <= 65535:
        return False
    return True


def websocket_accept_key(key: str) -> str:
    """Derive the Sec-WebSocket-Accept value from a validated client key."""

    if type(key) is not str:
        raise WebSocketWireError("websocket key must be a string")
    try:
        decoded = base64.b64decode(key, validate=True)
    except (binascii.Error, ValueError) as error:
        raise WebSocketWireError("websocket key must be valid base64") from error
    if len(decoded) != 16:
        raise WebSocketWireError("websocket key must decode to 16 bytes")
    digest = hashlib.sha1((key + WEBSOCKET_GUID).encode("ascii")).digest()
    return base64.b64encode(digest).decode("ascii")


def encode_frame(opcode: int, payload: bytes, *, fin: bool = True) -> bytes:
    """Encode one unmasked server-to-client frame."""

    if type(opcode) is not int or not 0 <= opcode <= 0xF:
        raise WebSocketWireError("opcode must be a 4-bit integer")
    if type(payload) is not bytes:
        raise WebSocketWireError("payload must be bytes")
    header = bytearray([(0x80 if fin else 0x00) | opcode])
    length = len(payload)
    if length <= 125:
        header.append(length)
    elif length <= 0xFFFF:
        header.append(126)
        header.extend(length.to_bytes(2, "big"))
    else:
        header.append(127)
        header.extend(length.to_bytes(8, "big"))
    return bytes(header) + payload


def encode_text_frame(text: str) -> bytes:
    return encode_frame(0x1, text.encode("utf-8"))


def encode_close_frame(code: int = 1000, reason: str = "") -> bytes:
    payload = code.to_bytes(2, "big") + reason.encode("utf-8")
    if len(payload) > 125:
        raise WebSocketWireError("close payload exceeds the control frame limit")
    return encode_frame(0x8, payload)


def encode_pong_frame(payload: bytes) -> bytes:
    if len(payload) > 125:
        raise WebSocketWireError("pong payload exceeds the control frame limit")
    return encode_frame(0xA, payload)


@dataclass(frozen=True, slots=True)
class TextMessage:
    text: str


@dataclass(frozen=True, slots=True)
class Ping:
    payload: bytes


@dataclass(frozen=True, slots=True)
class Pong:
    payload: bytes


@dataclass(frozen=True, slots=True)
class Close:
    code: int | None


FrameEvent = TextMessage | Ping | Pong | Close


class FrameDecoder:
    """Incremental, bounded decoder for masked client-to-server frames."""

    def __init__(self, *, max_message_bytes: int = 64 * 1024) -> None:
        if type(max_message_bytes) is not int or max_message_bytes < 1:
            raise WebSocketWireError("max_message_bytes must be a positive integer")
        self._max_message_bytes = max_message_bytes
        self._buffer = bytearray()
        self._fragments = bytearray()
        self._fragment_open = False

    def feed(self, data: bytes) -> list[FrameEvent]:
        """Consume bytes and return completed events; raise on violations."""

        if type(data) is not bytes:
            raise WebSocketWireError("frame data must be bytes")
        self._buffer.extend(data)
        if len(self._buffer) > self._max_message_bytes + 14:
            raise WebSocketWireError("frame buffer exceeds the message size limit")
        events: list[FrameEvent] = []
        while True:
            frame = self._decode_one()
            if frame is None:
                return events
            event = self._apply(*frame)
            if event is not None:
                events.append(event)

    def _decode_one(self) -> tuple[bool, int, bytes] | None:
        buffer = self._buffer
        if len(buffer) < 2:
            return None
        first, second = buffer[0], buffer[1]
        if first & 0x70:
            raise WebSocketWireError("reserved frame bits must be zero")
        fin = bool(first & 0x80)
        opcode = first & 0x0F
        if not second & 0x80:
            raise WebSocketWireError("client frames must be masked")
        length = second & 0x7F
        offset = 2
        if length == 126:
            if len(buffer) < offset + 2:
                return None
            length = int.from_bytes(buffer[offset : offset + 2], "big")
            offset += 2
        elif length == 127:
            if len(buffer) < offset + 8:
                return None
            length = int.from_bytes(buffer[offset : offset + 8], "big")
            offset += 8
            if length > 0x7FFFFFFFFFFFFFFF:
                raise WebSocketWireError("frame length high bit must be zero")
        if length > self._max_message_bytes:
            raise WebSocketWireError("frame exceeds the message size limit")
        if len(buffer) < offset + 4 + length:
            return None
        mask = bytes(buffer[offset : offset + 4])
        masked = bytes(buffer[offset + 4 : offset + 4 + length])
        del buffer[: offset + 4 + length]
        payload = bytes(byte ^ mask[index % 4] for index, byte in enumerate(masked))
        return fin, opcode, payload

    def _apply(self, fin: bool, opcode: int, payload: bytes) -> FrameEvent | None:
        if opcode in {0x8, 0x9, 0xA}:
            if not fin:
                raise WebSocketWireError("control frames must not be fragmented")
            if len(payload) > 125:
                raise WebSocketWireError("control frame payload exceeds 125 bytes")
            if opcode == 0x8:
                if len(payload) == 1:
                    raise WebSocketWireError("close frame payload must not be one byte")
                code = int.from_bytes(payload[:2], "big") if payload else None
                return Close(code)
            if opcode == 0x9:
                return Ping(payload)
            return Pong(payload)
        if opcode == 0x1:
            if self._fragment_open:
                raise WebSocketWireError("data frame interleaved with an open fragment")
            if fin:
                return TextMessage(self._decode_text(payload))
            self._fragment_open = True
            self._fragments = bytearray(payload)
            return None
        if opcode == 0x0:
            if not self._fragment_open:
                raise WebSocketWireError("continuation frame without an open fragment")
            self._fragments.extend(payload)
            if len(self._fragments) > self._max_message_bytes:
                raise WebSocketWireError("fragmented message exceeds the size limit")
            if not fin:
                return None
            self._fragment_open = False
            text = self._decode_text(bytes(self._fragments))
            self._fragments = bytearray()
            return TextMessage(text)
        raise WebSocketWireError(f"unsupported frame opcode {opcode}")

    @staticmethod
    def _decode_text(payload: bytes) -> str:
        try:
            return payload.decode("utf-8")
        except UnicodeDecodeError as error:
            raise WebSocketWireError("text frames must be valid UTF-8") from error

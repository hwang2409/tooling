"""Bounded HTTP/1.1 request-head parsing for the loopback API."""

from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass
from urllib.parse import urlsplit

MAX_REQUEST_HEAD_BYTES = 32 * 1024
MAX_HEADER_COUNT = 100
_TOKEN = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")
_TARGET = re.compile(r"^[!-~]+$")
_HEADER_VALUE = re.compile(r"^[\t\x20-\x7e\x80-\xff]*$")
_LOOPBACK_NAMES = frozenset({"localhost"})


class HttpWireError(ValueError):
    """Raised when a request head cannot be accepted."""


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
        return tuple(
            token.strip().lower()
            for value in self.header_values(name)
            for token in value.split(",")
            if token.strip()
        )


def parse_request_head(raw: bytes) -> RequestHead:
    if type(raw) is not bytes or len(raw) > MAX_REQUEST_HEAD_BYTES:
        raise HttpWireError("request head is invalid or too large")
    if not raw.endswith(b"\r\n\r\n"):
        raise HttpWireError("request head must end with CRLF CRLF")
    try:
        text = raw[:-4].decode("ascii")
    except UnicodeDecodeError as error:
        raise HttpWireError("request head must be ASCII") from error
    lines = text.split("\r\n")
    parts = lines[0].split(" ")
    if len(parts) != 3:
        raise HttpWireError("malformed request line")
    method, target, version = parts
    if not _TOKEN.fullmatch(method) or not _TARGET.fullmatch(target):
        raise HttpWireError("malformed request line")
    if version not in {"HTTP/1.1", "HTTP/1.0"}:
        raise HttpWireError("unsupported HTTP version")
    headers: list[tuple[str, str]] = []
    for line in lines[1:]:
        if not line or line[0] in " \t":
            raise HttpWireError("malformed header line")
        name, separator, value = line.partition(":")
        if not separator or not _TOKEN.fullmatch(name):
            raise HttpWireError("malformed header name")
        value = value.strip(" \t")
        if not _HEADER_VALUE.fullmatch(value):
            raise HttpWireError("malformed header value")
        headers.append((name.lower(), value))
        if len(headers) > MAX_HEADER_COUNT:
            raise HttpWireError("too many request headers")
    return RequestHead(method, target, version, tuple(headers))


def is_loopback_host_header(value: str) -> bool:
    if type(value) is not str or not value or "/" in value or "@" in value:
        return False
    try:
        parsed = urlsplit(f"//{value}")
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        return False
    if parsed.path or parsed.query or parsed.fragment or hostname is None:
        return False
    if hostname.lower() not in _LOOPBACK_NAMES:
        try:
            if not ipaddress.ip_address(hostname).is_loopback:
                return False
        except ValueError:
            return False
    return not value.rstrip().endswith(":") and (port is None or 1 <= port <= 65535)

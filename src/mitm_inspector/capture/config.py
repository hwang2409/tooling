"""Strict, side-effect-free environment configuration for capture."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass

DEFAULT_SOURCE_ID = "mitm-inspector"
DEFAULT_MAX_BODY_PREFIX_BYTES = 1 * 1024 * 1024
DEFAULT_MAX_IN_MEMORY_BYTES = 128 * 1024 * 1024
DEFAULT_MAX_PENDING_MESSAGES = 4_096
_DECIMAL = re.compile(r"(?:0|[1-9][0-9]*)\Z")
_MAX_UINT64 = 18_446_744_073_709_551_615


@dataclass(frozen=True)
class CaptureConfig:
    """Parsed capture settings; no transport or thread is created here."""

    capture_socket: str | None = None
    source_id: str = DEFAULT_SOURCE_ID
    max_body_prefix_bytes: int = DEFAULT_MAX_BODY_PREFIX_BYTES
    max_in_memory_bytes: int = DEFAULT_MAX_IN_MEMORY_BYTES
    max_pending_messages: int = DEFAULT_MAX_PENDING_MESSAGES

    def __post_init__(self) -> None:
        if not self.source_id or "\x00" in self.source_id:
            raise ValueError("source_id must be non-empty and contain no NUL bytes")
        if self.capture_socket is not None and (
            not self.capture_socket.startswith("/") or "\x00" in self.capture_socket
        ):
            raise ValueError("capture_socket must be an absolute POSIX Unix-socket path")
        if self.max_body_prefix_bytes < 0:
            raise ValueError("max_body_prefix_bytes must not be negative")
        if self.max_in_memory_bytes < 1:
            raise ValueError("max_in_memory_bytes must be greater than zero")
        if self.max_pending_messages < 1:
            raise ValueError("max_pending_messages must be greater than zero")
        if self.max_body_prefix_bytes > self.max_in_memory_bytes:
            raise ValueError("max_body_prefix_bytes must not exceed max_in_memory_bytes")

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> CaptureConfig:
        values = os.environ if environ is None else environ
        socket_value = _optional_socket(values, "MITM_INSPECTOR_CAPTURE_SOCKET")
        source_id = _optional_text(values, "MITM_INSPECTOR_SOURCE_ID", DEFAULT_SOURCE_ID)
        max_body = _optional_uint(
            values,
            "MITM_INSPECTOR_MAX_BODY_PREFIX_BYTES",
            DEFAULT_MAX_BODY_PREFIX_BYTES,
            allow_zero=True,
        )
        max_memory = _optional_uint(
            values,
            "MITM_INSPECTOR_MAX_IN_MEMORY_BYTES",
            DEFAULT_MAX_IN_MEMORY_BYTES,
            allow_zero=False,
        )
        max_pending = _optional_uint(
            values,
            "MITM_INSPECTOR_MAX_PENDING_MESSAGES",
            DEFAULT_MAX_PENDING_MESSAGES,
            allow_zero=False,
        )
        if max_body > max_memory:
            raise ValueError(
                "MITM_INSPECTOR_MAX_BODY_PREFIX_BYTES must not exceed "
                "MITM_INSPECTOR_MAX_IN_MEMORY_BYTES"
            )
        return cls(socket_value, source_id, max_body, max_memory, max_pending)


def _optional_text(environ: Mapping[str, str], name: str, default: str) -> str:
    value = environ.get(name)
    if value is None:
        return default
    if not value or "\x00" in value:
        raise ValueError(f"{name} must be a non-empty string without NUL bytes")
    return value


def _optional_socket(environ: Mapping[str, str], name: str) -> str | None:
    value = environ.get(name)
    if value is None:
        return None
    if not value or not value.startswith("/") or "\x00" in value:
        raise ValueError(f"{name} must be an absolute POSIX Unix-socket path")
    return value


def _optional_uint(
    environ: Mapping[str, str], name: str, default: int, *, allow_zero: bool
) -> int:
    value = environ.get(name)
    if value is None:
        return default
    if not _DECIMAL.fullmatch(value):
        raise ValueError(f"{name} must be a decimal integer")
    parsed = int(value)
    if parsed > _MAX_UINT64:
        raise ValueError(f"{name} exceeds uint64")
    if not allow_zero and parsed == 0:
        raise ValueError(f"{name} must be greater than zero")
    return parsed

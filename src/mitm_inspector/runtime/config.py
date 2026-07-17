"""Validated, immutable configuration for the local runtime."""

from __future__ import annotations

import ipaddress
import os
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit


class RuntimeConfigError(ValueError):
    """Raised when a runtime setting would make the local boundary unsafe."""


_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_CONTROL_OR_SPACE = re.compile(r"[\x00-\x20\x7f]")


def _positive_int(value: int, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise RuntimeConfigError(f"{name} must be a positive integer")
    return value


def _port(value: int, name: str) -> int:
    if type(value) is not int or not 1 <= value <= 65535:
        raise RuntimeConfigError(f"{name} must be an integer from 1 through 65535")
    return value


def _loopback_host(value: str, name: str) -> str:
    if type(value) is not str or not value or _CONTROL_OR_SPACE.search(value):
        raise RuntimeConfigError(f"{name} must be a loopback host")
    if value.lower() in _LOOPBACK_HOSTS:
        return value
    try:
        address = ipaddress.ip_address(value)
    except ValueError as exc:
        raise RuntimeConfigError(f"{name} must be a loopback host") from exc
    if not address.is_loopback:
        raise RuntimeConfigError(f"{name} must be a loopback host")
    return value


def _upstream_url(value: str) -> str:
    if type(value) is not str or not value or _CONTROL_OR_SPACE.search(value):
        raise RuntimeConfigError("reverse_upstream must be an absolute HTTP(S) URL")
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        upstream_port = parsed.port
    except ValueError as exc:
        if "port" in str(exc).lower():
            raise RuntimeConfigError("reverse_upstream must use a valid port") from exc
        raise RuntimeConfigError("reverse_upstream must be an absolute HTTP(S) URL") from exc
    if parsed.scheme.lower() not in {"http", "https"} or not hostname:
        raise RuntimeConfigError("reverse_upstream must be an absolute HTTP(S) URL")
    if upstream_port is not None and not 1 <= upstream_port <= 65535:
        raise RuntimeConfigError("reverse_upstream must use a valid port")
    if parsed.username is not None or parsed.password is not None:
        raise RuntimeConfigError("reverse_upstream must not contain credentials")
    if "?" in value or "#" in value:
        raise RuntimeConfigError("reverse_upstream must not contain a query or fragment")
    return value


def _executable(value: Path | str, name: str) -> Path:
    try:
        raw = os.fspath(value)
        path = Path(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeConfigError(f"{name} must be a non-empty executable path") from exc
    if not raw or "\x00" in raw:
        raise RuntimeConfigError(f"{name} must be a non-empty executable path")
    return path


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """All settings needed to compose and supervise the local two-process runtime.

    Paths are intentionally not resolved or checked for existence.  This keeps
    ``plan`` useful in a clean checkout and lets callers supply a PATH command
    such as ``mitmdump`` or an absolute executable in deployment code.
    """

    reverse_upstream: str = "https://api.anthropic.com"
    app_host: str = "127.0.0.1"
    proxy_host: str = "127.0.0.1"
    app_port: int = 8000
    proxy_port: int = 8080
    retention_max_flows: int = 2_000
    retention_max_age_seconds: int = 1_800
    max_body_bytes: int = 128 * 1024 * 1024
    max_body_prefix_bytes: int = 1024 * 1024
    mitmdump_executable: Path = Path("mitmdump")
    app_executable: Path = Path("python")
    open_browser: bool = False
    readiness_timeout_seconds: float = 15.0
    graceful_shutdown_seconds: float = 5.0
    kill_wait_seconds: float = 1.0
    poll_interval_seconds: float = 0.1

    def __post_init__(self) -> None:
        object.__setattr__(self, "reverse_upstream", _upstream_url(self.reverse_upstream))
        object.__setattr__(self, "app_host", _loopback_host(self.app_host, "app_host"))
        object.__setattr__(self, "proxy_host", _loopback_host(self.proxy_host, "proxy_host"))
        object.__setattr__(self, "app_port", _port(self.app_port, "app_port"))
        object.__setattr__(self, "proxy_port", _port(self.proxy_port, "proxy_port"))
        if self.app_port == self.proxy_port:
            raise RuntimeConfigError("app_port and proxy_port must be different")
        object.__setattr__(
            self,
            "retention_max_flows",
            _positive_int(self.retention_max_flows, "retention_max_flows"),
        )
        object.__setattr__(
            self,
            "retention_max_age_seconds",
            _positive_int(self.retention_max_age_seconds, "retention_max_age_seconds"),
        )
        object.__setattr__(
            self,
            "max_body_bytes",
            _positive_int(self.max_body_bytes, "max_body_bytes"),
        )
        object.__setattr__(
            self,
            "max_body_prefix_bytes",
            _positive_int(self.max_body_prefix_bytes, "max_body_prefix_bytes"),
        )
        if self.max_body_prefix_bytes > self.max_body_bytes:
            raise RuntimeConfigError("max_body_prefix_bytes cannot exceed max_body_bytes")
        object.__setattr__(
            self,
            "mitmdump_executable",
            _executable(self.mitmdump_executable, "mitmdump_executable"),
        )
        object.__setattr__(
            self,
            "app_executable",
            _executable(self.app_executable, "app_executable"),
        )
        if type(self.open_browser) is not bool:
            raise RuntimeConfigError("open_browser must be a boolean")
        for name in (
            "readiness_timeout_seconds",
            "graceful_shutdown_seconds",
            "kill_wait_seconds",
            "poll_interval_seconds",
        ):
            value = getattr(self, name)
            if type(value) not in {int, float} or value <= 0:
                raise RuntimeConfigError(f"{name} must be a positive number")

    @property
    def max_flows(self) -> int:
        """Short alias used by runtime integrations."""

        return self.retention_max_flows

    @property
    def max_age_seconds(self) -> int:
        """Short alias used by runtime integrations."""

        return self.retention_max_age_seconds

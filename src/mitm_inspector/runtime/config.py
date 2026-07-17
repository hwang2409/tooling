"""Validated immutable configuration for the local runtime boundary."""

from __future__ import annotations

import ipaddress
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


class RuntimeConfigError(ValueError):
    """Raised when a runtime setting would make the local boundary unsafe."""


class RuntimePreflightError(RuntimeError):
    """Raised when live runtime executable/addon paths are not usable."""

    def __init__(self, issues: tuple[str, ...]) -> None:
        self.issues = issues
        super().__init__("runtime preflight failed: " + "; ".join(issues))


_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_CONTROL_OR_SPACE = re.compile(r"[\x00-\x20\x7f]")
_HOST_LABEL = re.compile(r"^[A-Za-z0-9_-]{1,63}$")
_DEFAULT_SOCKET_PATH = Path(tempfile.gettempdir()).resolve() / "mitm-inspector.sock"
DEFAULT_APP_EXECUTABLE = Path(sys.executable)
DEFAULT_MITMDUMP_EXECUTABLE = DEFAULT_APP_EXECUTABLE.with_name("mitmdump")
DEFAULT_ADDON_PATH = Path(__file__).resolve().parents[1] / "capture" / "addon.py"

CAPTURE_SOCKET_ENV = "MITM_INSPECTOR_CAPTURE_SOCKET"
CAPTURE_SOURCE_ID_ENV = "MITM_INSPECTOR_SOURCE_ID"
CAPTURE_MAX_BODY_PREFIX_ENV = "MITM_INSPECTOR_MAX_BODY_PREFIX_BYTES"
CAPTURE_MAX_MEMORY_ENV = "MITM_INSPECTOR_MAX_IN_MEMORY_BYTES"


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


def _valid_authority_host(host: str) -> bool:
    if not host:
        return False
    try:
        if len(host.encode("idna")) > 255:
            return False
    except UnicodeError:
        return False
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        pass
    try:
        encoded = host.encode("idna")
    except UnicodeError:
        return False
    if encoded.endswith(b"."):
        encoded = encoded[:-1]
    return bool(encoded) and all(
        _HOST_LABEL.fullmatch(label) for label in encoded.decode().split(".")
    )


def _upstream_url(value: str) -> str:
    """Validate the ``reverse:<spec>`` authority grammar used by mitmdump 12.2.3."""

    if type(value) is not str or not value or _CONTROL_OR_SPACE.search(value):
        raise RuntimeConfigError("reverse_upstream must be an absolute HTTP(S) authority URL")
    try:
        parsed = urlsplit(value)
        scheme = parsed.scheme.lower()
        hostname = parsed.hostname
        upstream_port = parsed.port
    except ValueError as exc:
        if "port" in str(exc).lower():
            raise RuntimeConfigError("reverse_upstream must use a valid port") from exc
        raise RuntimeConfigError(
            "reverse_upstream must be an absolute HTTP(S) authority URL"
        ) from exc
    if scheme not in {"http", "https"} or not hostname:
        raise RuntimeConfigError("reverse_upstream must use lowercase http or https authority")
    if parsed.username is not None or parsed.password is not None:
        raise RuntimeConfigError("reverse_upstream must not contain credentials")
    if parsed.path or parsed.query or parsed.fragment or value.endswith(":"):
        raise RuntimeConfigError(
            "reverse_upstream must not contain a path, query, fragment, or empty port"
        )
    if not _valid_authority_host(hostname):
        raise RuntimeConfigError("reverse_upstream must contain a valid hostname or IP")
    if upstream_port is not None and not 1 <= upstream_port <= 65535:
        raise RuntimeConfigError("reverse_upstream must use a valid port")
    return urlunsplit((scheme, parsed.netloc, "", "", ""))


def _absolute_path(value: Path | str, name: str) -> Path:
    try:
        raw = os.fspath(value)
        path = Path(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeConfigError(f"{name} must be a non-empty absolute path") from exc
    if not raw or "\x00" in raw or not path.is_absolute():
        raise RuntimeConfigError(f"{name} must be a non-empty absolute path")
    return path


@dataclass(frozen=True, slots=True)
class CaptureIPCConfig:
    """Durable local capture transport settings shared by both children.

    The proxy receives these values through the explicit environment overlay
    because stock mitmdump cannot accept project-specific command-line
    options.  The future app receives the identical overlay and may also
    expose equivalent flags without changing this contract.
    """

    socket_path: Path = _DEFAULT_SOCKET_PATH
    source_id: str = "mitm-inspector"
    max_body_prefix_bytes: int = 1024 * 1024
    max_in_memory_bytes: int = 128 * 1024 * 1024

    def __post_init__(self) -> None:
        path = _absolute_path(self.socket_path, "capture socket path")
        if os.name == "posix" and len(os.fsencode(str(path))) >= 104:
            raise RuntimeConfigError("capture socket path is too long for a Unix socket")
        if (
            type(self.source_id) is not str
            or not self.source_id
            or _CONTROL_OR_SPACE.search(self.source_id)
        ):
            raise RuntimeConfigError("capture source_id must be a non-empty token")
        object.__setattr__(self, "socket_path", path)
        object.__setattr__(
            self,
            "max_body_prefix_bytes",
            _positive_int(self.max_body_prefix_bytes, "capture max_body_prefix_bytes"),
        )
        object.__setattr__(
            self,
            "max_in_memory_bytes",
            _positive_int(self.max_in_memory_bytes, "capture max_in_memory_bytes"),
        )
        if self.max_body_prefix_bytes > self.max_in_memory_bytes:
            raise RuntimeConfigError(
                "capture max_body_prefix_bytes cannot exceed max_in_memory_bytes"
            )

    @property
    def unix_socket_path(self) -> Path:
        """Compatibility spelling for the POSIX transport path."""

        return self.socket_path

    def environment(self) -> dict[str, str]:
        """Return the exact environment overlay used by proxy and app children."""

        return {
            CAPTURE_SOCKET_ENV: str(self.socket_path),
            CAPTURE_SOURCE_ID_ENV: self.source_id,
            CAPTURE_MAX_BODY_PREFIX_ENV: str(self.max_body_prefix_bytes),
            CAPTURE_MAX_MEMORY_ENV: str(self.max_in_memory_bytes),
        }


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """All settings needed to compose and supervise the local two-process runtime."""

    reverse_upstream: str = "https://api.anthropic.com"
    app_host: str = "127.0.0.1"
    proxy_host: str = "127.0.0.1"
    app_port: int = 8000
    proxy_port: int = 8080
    retention_max_flows: int = 2_000
    retention_max_age_seconds: int = 1_800
    max_body_bytes: int = 128 * 1024 * 1024
    max_body_prefix_bytes: int = 1024 * 1024
    mitmdump_executable: Path = DEFAULT_MITMDUMP_EXECUTABLE
    app_executable: Path = DEFAULT_APP_EXECUTABLE
    addon_path: Path = DEFAULT_ADDON_PATH
    capture_ipc: CaptureIPCConfig | None = None
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
            _absolute_path(self.mitmdump_executable, "mitmdump_executable"),
        )
        object.__setattr__(
            self,
            "app_executable",
            _absolute_path(self.app_executable, "app_executable"),
        )
        object.__setattr__(self, "addon_path", _absolute_path(self.addon_path, "addon_path"))
        ipc = self.capture_ipc or CaptureIPCConfig(
            socket_path=_DEFAULT_SOCKET_PATH,
            max_body_prefix_bytes=self.max_body_prefix_bytes,
            max_in_memory_bytes=self.max_body_bytes,
        )
        if (
            ipc.max_body_prefix_bytes != self.max_body_prefix_bytes
            or ipc.max_in_memory_bytes != self.max_body_bytes
        ):
            raise RuntimeConfigError(
                "capture_ipc limits must match max_body_prefix_bytes and max_body_bytes"
            )
        object.__setattr__(self, "capture_ipc", ipc)
        if type(self.open_browser) is not bool:
            raise RuntimeConfigError("open_browser must be a boolean")
        for name in (
            "readiness_timeout_seconds",
            "graceful_shutdown_seconds",
            "kill_wait_seconds",
            "poll_interval_seconds",
        ):
            value = getattr(self, name)
            if type(value) not in {int, float} or not math.isfinite(float(value)) or value <= 0:
                raise RuntimeConfigError(f"{name} must be a finite positive number")

    @property
    def capture(self) -> CaptureIPCConfig:
        """Return the normalized shared capture configuration."""

        if self.capture_ipc is None:  # pragma: no cover - __post_init__ always fills it.
            raise RuntimeConfigError("capture_ipc was not normalized")
        return self.capture_ipc

    @property
    def max_flows(self) -> int:
        return self.retention_max_flows

    @property
    def max_age_seconds(self) -> int:
        return self.retention_max_age_seconds


def preflight_issues(config: RuntimeConfig) -> tuple[str, ...]:
    """Return live-startup path issues without raising, for CLI plan output."""

    issues: list[str] = []
    for name, path in (
        ("app_executable", config.app_executable),
        ("mitmdump_executable", config.mitmdump_executable),
    ):
        if not path.is_file():
            issues.append(f"{name} is not a file: {path}")
        elif not os.access(path, os.X_OK):
            issues.append(f"{name} is not executable: {path}")
    if not config.addon_path.is_file():
        issues.append(f"addon_path is not a file: {config.addon_path}")
    if not config.capture.socket_path.parent.is_dir():
        issues.append(
            f"capture socket parent is not a directory: {config.capture.socket_path.parent}"
        )
    elif not os.access(config.capture.socket_path.parent, os.W_OK):
        issues.append(
            f"capture socket parent is not writable: {config.capture.socket_path.parent}"
        )
    return tuple(issues)


def validate_preflight(config: RuntimeConfig) -> None:
    issues = preflight_issues(config)
    if issues:
        raise RuntimePreflightError(issues)

"""Validated immutable configuration for the local runtime boundary."""

from __future__ import annotations

import ipaddress
import math
import os
import re
import stat
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from mitm_inspector.api.limits import MAX_INGEST_BODY_PREFIX_BYTES
from mitm_inspector.store.sqlite import (
    DEFAULT_STORAGE_MAX_BYTES,
    DEFAULT_STORAGE_MAX_FLOWS,
    DEFAULT_STORAGE_REPLAY,
    default_storage_path,
)


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
_DEFAULT_RUNTIME_BASE_DIR = Path(tempfile.gettempdir()).resolve()
DEFAULT_APP_EXECUTABLE = Path(sys.executable)
DEFAULT_MITMDUMP_EXECUTABLE = DEFAULT_APP_EXECUTABLE.with_name("mitmdump")
DEFAULT_ADDON_PATH = Path(__file__).resolve().parents[1] / "capture" / "addon.py"
MAX_UINT64 = 18_446_744_073_709_551_615

CAPTURE_SOCKET_ENV = "MITM_INSPECTOR_CAPTURE_SOCKET"
CAPTURE_SOURCE_ID_ENV = "MITM_INSPECTOR_SOURCE_ID"
CAPTURE_MAX_BODY_PREFIX_ENV = "MITM_INSPECTOR_MAX_BODY_PREFIX_BYTES"
CAPTURE_MAX_MEMORY_ENV = "MITM_INSPECTOR_MAX_IN_MEMORY_BYTES"
CAPTURE_MAX_PENDING_ENV = "MITM_INSPECTOR_MAX_PENDING_MESSAGES"


def _uint64(value: int, name: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if type(value) is not int or not minimum <= value <= MAX_UINT64:
        range_text = "0" if allow_zero else "1"
        raise RuntimeConfigError(f"{name} must be an unsigned 64-bit integer from {range_text}")
    return value


def _finite_positive(value: int | float, name: str) -> float:
    if type(value) not in {int, float}:
        raise RuntimeConfigError(f"{name} must be a finite positive number")
    try:
        numeric = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise RuntimeConfigError(f"{name} must be a finite positive number") from exc
    if not math.isfinite(numeric) or numeric <= 0:
        raise RuntimeConfigError(f"{name} must be a finite positive number")
    return numeric


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
        scheme_match = re.match(r"^([A-Za-z][A-Za-z0-9+.-]*):", value)
        scheme = scheme_match.group(1) if scheme_match else ""
        hostname = parsed.hostname
        upstream_port = parsed.port
    except ValueError as exc:
        if "port" in str(exc).lower():
            raise RuntimeConfigError("reverse_upstream must use a valid port") from exc
        raise RuntimeConfigError(
            "reverse_upstream must be an absolute HTTP(S) authority URL"
        ) from exc
    if scheme not in {"http", "https"} or scheme != scheme.lower() or not hostname:
        raise RuntimeConfigError("reverse_upstream must use lowercase http or https authority")
    if parsed.username is not None or parsed.password is not None:
        raise RuntimeConfigError("reverse_upstream must not contain credentials")
    if (
        parsed.path
        or parsed.query
        or parsed.fragment
        or any(delimiter in value for delimiter in "?#")
        or value.endswith(":")
    ):
        raise RuntimeConfigError(
            "reverse_upstream must not contain a path, query, fragment, or empty port"
        )
    if not _valid_authority_host(hostname):
        raise RuntimeConfigError("reverse_upstream must contain a valid hostname or IP")
    if upstream_port is not None and not 1 <= upstream_port <= 65535:
        raise RuntimeConfigError("reverse_upstream must use a valid port")
    return value


def _absolute_path(value: Path | str, name: str) -> Path:
    try:
        raw = os.fspath(value)
        path = Path(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeConfigError(f"{name} must be a non-empty absolute path") from exc
    if not raw or "\x00" in raw or not path.is_absolute():
        raise RuntimeConfigError(f"{name} must be a non-empty absolute path")
    return path


def validate_private_runtime_dir(path: Path) -> None:
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise RuntimeConfigError(f"runtime directory is unavailable: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise RuntimeConfigError(f"runtime directory must be a real directory: {path}")
    if os.name == "posix" and info.st_uid != os.getuid():
        raise RuntimeConfigError(f"runtime directory is not owned by the current user: {path}")
    if stat.S_IMODE(info.st_mode) != 0o700:
        raise RuntimeConfigError(f"runtime directory must have mode 0700: {path}")


def _validate_runtime_base_dir(path: Path) -> None:
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise RuntimeConfigError(f"runtime base directory is unavailable: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise RuntimeConfigError(f"runtime base directory must be a real directory: {path}")
    if not os.access(path, os.W_OK):
        raise RuntimeConfigError(f"runtime base directory is not writable: {path}")


def cleanup_private_runtime_dir(capture: CaptureIPCConfig) -> tuple[str, ...]:
    """Best-effort cleanup that refuses to unlink adversarial paths."""

    runtime_dir = capture.runtime_dir
    if runtime_dir is None:
        return ()
    issues: list[str] = []
    try:
        validate_private_runtime_dir(runtime_dir)
    except RuntimeConfigError as exc:
        return (str(exc),)
    endpoint = capture.socket_path
    try:
        endpoint_info = os.lstat(endpoint)
        if stat.S_ISLNK(endpoint_info.st_mode):
            issues.append(f"refusing to remove symlink capture endpoint: {endpoint}")
        elif not stat.S_ISSOCK(endpoint_info.st_mode):
            issues.append(f"refusing to remove non-socket capture endpoint: {endpoint}")
        else:
            endpoint.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        issues.append(f"capture endpoint cleanup failed: {exc}")
    if not issues:
        try:
            runtime_dir.rmdir()
        except FileNotFoundError:
            pass
        except OSError as exc:
            issues.append(f"runtime directory cleanup failed: {exc}")
    return tuple(issues)


@dataclass(frozen=True, slots=True)
class CaptureIPCConfig:
    """Durable local capture transport settings shared by both children.

    The proxy receives these values through the explicit environment overlay
    because stock mitmdump cannot accept project-specific command-line
    options.  The app receives the identical overlay and may also
    expose equivalent flags without changing this contract.
    """

    socket_path: Path = _DEFAULT_SOCKET_PATH
    runtime_base_dir: Path = _DEFAULT_RUNTIME_BASE_DIR
    runtime_dir: Path | None = None
    source_id: str = "mitm-inspector"
    max_body_prefix_bytes: int = 1024 * 1024
    max_in_memory_bytes: int = 128 * 1024 * 1024
    max_pending_messages: int = 4096

    def __post_init__(self) -> None:
        path = _absolute_path(self.socket_path, "capture socket path")
        base_dir = _absolute_path(self.runtime_base_dir, "runtime base directory")
        _validate_runtime_base_dir(base_dir)
        runtime_dir = self.runtime_dir
        if runtime_dir is not None:
            runtime_dir = _absolute_path(runtime_dir, "runtime directory")
            validate_private_runtime_dir(runtime_dir)
            if path.parent != runtime_dir:
                raise RuntimeConfigError("capture socket must be directly inside runtime directory")
        if os.name == "posix" and len(os.fsencode(str(path))) >= 104:
            raise RuntimeConfigError("capture socket path is too long for a Unix socket")
        if (
            type(self.source_id) is not str
            or not self.source_id
            or _CONTROL_OR_SPACE.search(self.source_id)
        ):
            raise RuntimeConfigError("capture source_id must be a non-empty token")
        object.__setattr__(self, "socket_path", path)
        object.__setattr__(self, "runtime_base_dir", base_dir)
        object.__setattr__(self, "runtime_dir", runtime_dir)
        object.__setattr__(
            self,
            "max_body_prefix_bytes",
            _uint64(
                self.max_body_prefix_bytes,
                "capture max_body_prefix_bytes",
                allow_zero=True,
            ),
        )
        object.__setattr__(
            self,
            "max_in_memory_bytes",
            _uint64(
                self.max_in_memory_bytes,
                "capture max_in_memory_bytes",
                allow_zero=False,
            ),
        )
        object.__setattr__(
            self,
            "max_pending_messages",
            _uint64(
                self.max_pending_messages,
                "capture max_pending_messages",
                allow_zero=False,
            ),
        )
        if self.max_body_prefix_bytes > self.max_in_memory_bytes:
            raise RuntimeConfigError(
                "capture max_body_prefix_bytes cannot exceed max_in_memory_bytes"
            )
        if self.max_body_prefix_bytes > MAX_INGEST_BODY_PREFIX_BYTES:
            # Two base64 body prefixes plus the metadata envelope must fit
            # one bounded ingest line, or every large flow would drop the
            # producer connection.
            raise RuntimeConfigError(
                "capture max_body_prefix_bytes exceeds the bounded ingest line capacity"
            )

    @property
    def unix_socket_path(self) -> Path:
        """Compatibility spelling for the POSIX transport path."""

        return self.socket_path

    def allocate(self) -> CaptureIPCConfig:
        """Allocate a fresh mode-0700 runtime directory and socket endpoint."""

        _validate_runtime_base_dir(self.runtime_base_dir)
        runtime_dir = Path(
            tempfile.mkdtemp(prefix=".mitm-inspector-", dir=self.runtime_base_dir)
        )
        try:
            validate_private_runtime_dir(runtime_dir)
            endpoint = runtime_dir / "capture.sock"
            if os.path.lexists(endpoint):
                raise RuntimeConfigError(f"capture endpoint unexpectedly exists: {endpoint}")
            return CaptureIPCConfig(
                socket_path=endpoint,
                runtime_base_dir=self.runtime_base_dir,
                runtime_dir=runtime_dir,
                source_id=self.source_id,
                max_body_prefix_bytes=self.max_body_prefix_bytes,
                max_in_memory_bytes=self.max_in_memory_bytes,
                max_pending_messages=self.max_pending_messages,
            )
        except BaseException:
            try:
                runtime_dir.rmdir()
            except OSError:
                # Never recursively remove a path after validation fails; an
                # unexpected entry is safer to report and leave for cleanup.
                pass
            raise

    def environment(self) -> dict[str, str]:
        """Return the exact environment overlay used by proxy and app children."""

        return {
            CAPTURE_SOCKET_ENV: str(self.socket_path),
            CAPTURE_SOURCE_ID_ENV: self.source_id,
            CAPTURE_MAX_BODY_PREFIX_ENV: str(self.max_body_prefix_bytes),
            CAPTURE_MAX_MEMORY_ENV: str(self.max_in_memory_bytes),
            CAPTURE_MAX_PENDING_ENV: str(self.max_pending_messages),
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
    max_pending_messages: int = 4096
    storage_path: Path | str = default_storage_path()
    no_storage: bool = False
    storage_max_flows: int = DEFAULT_STORAGE_MAX_FLOWS
    storage_max_bytes: int = DEFAULT_STORAGE_MAX_BYTES
    storage_replay: int = DEFAULT_STORAGE_REPLAY
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
            _uint64(self.retention_max_flows, "retention_max_flows", allow_zero=False),
        )
        object.__setattr__(
            self,
            "retention_max_age_seconds",
            _uint64(
                self.retention_max_age_seconds,
                "retention_max_age_seconds",
                allow_zero=False,
            ),
        )
        object.__setattr__(
            self,
            "max_body_bytes",
            _uint64(self.max_body_bytes, "max_body_bytes", allow_zero=False),
        )
        object.__setattr__(
            self,
            "max_body_prefix_bytes",
            _uint64(self.max_body_prefix_bytes, "max_body_prefix_bytes", allow_zero=True),
        )
        object.__setattr__(
            self,
            "max_pending_messages",
            _uint64(self.max_pending_messages, "max_pending_messages", allow_zero=False),
        )
        if os.fspath(self.storage_path) != ":memory:":
            object.__setattr__(
                self, "storage_path", _absolute_path(self.storage_path, "storage_path")
            )
        if type(self.no_storage) is not bool:
            raise RuntimeConfigError("no_storage must be a boolean")
        object.__setattr__(
            self,
            "storage_max_flows",
            _uint64(self.storage_max_flows, "storage_max_flows", allow_zero=False),
        )
        object.__setattr__(
            self,
            "storage_max_bytes",
            _uint64(self.storage_max_bytes, "storage_max_bytes", allow_zero=True),
        )
        object.__setattr__(
            self,
            "storage_replay",
            _uint64(self.storage_replay, "storage_replay", allow_zero=True),
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
            max_pending_messages=self.max_pending_messages,
        )
        if (
            ipc.max_body_prefix_bytes != self.max_body_prefix_bytes
            or ipc.max_in_memory_bytes != self.max_body_bytes
            or ipc.max_pending_messages != self.max_pending_messages
        ):
            raise RuntimeConfigError(
                "capture_ipc limits must match body and pending-message settings"
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
            _finite_positive(value, name)

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
    if not config.capture.runtime_base_dir.is_dir():
        issues.append(
            f"capture runtime base is not a directory: {config.capture.runtime_base_dir}"
        )
    elif not os.access(config.capture.runtime_base_dir, os.W_OK):
        issues.append(
            f"capture runtime base is not writable: {config.capture.runtime_base_dir}"
        )
    if config.capture.runtime_dir is not None:
        try:
            validate_private_runtime_dir(config.capture.runtime_dir)
        except RuntimeConfigError as exc:
            issues.append(str(exc))
        if config.capture.socket_path.parent != config.capture.runtime_dir:
            issues.append("capture socket must be directly inside runtime directory")
    return tuple(issues)


def validate_preflight(config: RuntimeConfig) -> None:
    issues = preflight_issues(config)
    if issues:
        raise RuntimePreflightError(issues)

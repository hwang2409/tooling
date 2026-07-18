"""Pure argv and explicit environment composition for runtime children."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

from mitm_inspector.runtime.config import RuntimeConfig

DEFAULT_APP_SERVER_MODULE: Final = "mitm_inspector.api.server"
CAPTURE_SOCKET_ARG: Final = "--capture-socket"
CAPTURE_SOURCE_ID_ARG: Final = "--capture-source-id"
CAPTURE_MAX_BODY_PREFIX_ARG: Final = "--capture-max-body-prefix-bytes"
CAPTURE_MAX_MEMORY_ARG: Final = "--capture-max-in-memory-bytes"
CAPTURE_MAX_PENDING_ARG: Final = "--capture-max-pending-messages"
STORAGE_PATH_ARG: Final = "--storage-path"
STORAGE_MAX_FLOWS_ARG: Final = "--storage-max-flows"
STORAGE_MAX_BYTES_ARG: Final = "--storage-max-bytes"
STORAGE_REPLAY_ARG: Final = "--storage-replay"


@dataclass(frozen=True, slots=True)
class ProcessSpec:
    """Immutable launch contract shared by real and fake process factories.

    ``env`` is an explicit overlay.  The production launcher copies the
    parent's environment and applies this overlay, so PATH and deployment
    variables remain available while the IPC contract is unambiguous.
    """

    name: str
    argv: tuple[str, ...]
    env: Mapping[str, str]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name or any(
            not isinstance(argument, str) for argument in self.argv
        ):
            raise ValueError("process spec requires a name and string argv")
        if not self.argv or any(
            not isinstance(key, str)
            or not isinstance(value, str)
            or not key
            or "\x00" in key
            or "\x00" in value
            for key, value in self.env.items()
        ):
            raise ValueError("process spec requires a non-empty argv and valid environment")
        object.__setattr__(self, "argv", tuple(self.argv))
        object.__setattr__(self, "env", MappingProxyType(dict(self.env)))


def _path_argument(value: Path | str, name: str) -> str:
    argument = os.fspath(value)
    if not isinstance(argument, str) or not argument or "\x00" in argument:
        raise ValueError(f"{name} must be a non-empty path")
    return argument


def build_proxy_argv(
    config: RuntimeConfig,
    addon_path: Path | str | None = None,
) -> tuple[str, ...]:
    """Build stock mitmdump argv with documented ``-s`` addon loading."""

    addon = _path_argument(addon_path or config.addon_path, "addon_path")
    return (
        _path_argument(config.mitmdump_executable, "mitmdump_executable"),
        "--mode",
        f"reverse:{config.reverse_upstream}",
        "--listen-host",
        config.proxy_host,
        "--listen-port",
        str(config.proxy_port),
        "-s",
        addon,
    )


def build_app_argv(
    config: RuntimeConfig,
    module: str = DEFAULT_APP_SERVER_MODULE,
) -> tuple[str, ...]:
    """Build the app-server argv, including the durable IPC flags."""

    if not module or any(character.isspace() for character in module) or "\x00" in module:
        raise ValueError("app server module must be a non-empty dotted module name")
    ipc = config.capture
    return (
        _path_argument(config.app_executable, "app_executable"),
        "-m",
        module,
        "--host",
        config.app_host,
        "--port",
        str(config.app_port),
        "--proxy-port",
        str(config.proxy_port),
        "--max-retained-flows",
        str(config.retention_max_flows),
        "--retention-seconds",
        str(config.retention_max_age_seconds),
        "--max-body-bytes",
        str(config.max_body_bytes),
        "--max-body-prefix-bytes",
        str(config.max_body_prefix_bytes),
        CAPTURE_SOCKET_ARG,
        str(ipc.socket_path),
        CAPTURE_SOURCE_ID_ARG,
        ipc.source_id,
        CAPTURE_MAX_BODY_PREFIX_ARG,
        str(ipc.max_body_prefix_bytes),
        CAPTURE_MAX_MEMORY_ARG,
        str(ipc.max_in_memory_bytes),
        CAPTURE_MAX_PENDING_ARG,
        str(ipc.max_pending_messages),
        STORAGE_PATH_ARG,
        str(config.storage_path),
        STORAGE_MAX_FLOWS_ARG,
        str(config.storage_max_flows),
        STORAGE_MAX_BYTES_ARG,
        str(config.storage_max_bytes),
        STORAGE_REPLAY_ARG,
        str(config.storage_replay),
        *(('--no-storage',) if config.no_storage else ()),
    )


def build_proxy_spec(config: RuntimeConfig) -> ProcessSpec:
    return ProcessSpec(
        name="proxy",
        argv=build_proxy_argv(config),
        env=config.capture.environment(),
    )


def build_app_spec(
    config: RuntimeConfig,
    module: str = DEFAULT_APP_SERVER_MODULE,
) -> ProcessSpec:
    return ProcessSpec(
        name="app",
        argv=build_app_argv(config, module),
        env=config.capture.environment(),
    )


def build_mitmdump_argv(
    config: RuntimeConfig,
    addon_path: Path | str | None = None,
) -> tuple[str, ...]:
    """Descriptive alias for callers that prefer the executable's full name."""

    return build_proxy_argv(config, addon_path)


def build_app_server_argv(
    config: RuntimeConfig,
    module: str = DEFAULT_APP_SERVER_MODULE,
) -> tuple[str, ...]:
    """Descriptive alias for the app-server command."""

    return build_app_argv(config, module)

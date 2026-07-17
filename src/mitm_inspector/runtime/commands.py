"""Pure argument-vector composition for the two runtime children."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

from mitm_inspector.runtime.config import RuntimeConfig

DEFAULT_APP_SERVER_MODULE: Final = "mitm_inspector.api.server"


def _path_argument(value: Path | str, name: str) -> str:
    argument = os.fspath(value)
    if not argument or "\x00" in argument:
        raise ValueError(f"{name} must be a non-empty path")
    return argument


def build_proxy_argv(config: RuntimeConfig, addon_path: Path | str) -> tuple[str, ...]:
    """Build the stock ``mitmdump`` argv, including only public addon loading."""

    addon = _path_argument(addon_path, "addon_path")
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
    """Build the future app-server argv without importing or launching the server."""

    if not module or any(character.isspace() for character in module) or "\x00" in module:
        raise ValueError("app server module must be a non-empty dotted module name")
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
    )


def build_mitmdump_argv(config: RuntimeConfig, addon_path: Path | str) -> tuple[str, ...]:
    """Descriptive alias for callers that prefer the executable's full name."""

    return build_proxy_argv(config, addon_path)


def build_app_server_argv(
    config: RuntimeConfig,
    module: str = DEFAULT_APP_SERVER_MODULE,
) -> tuple[str, ...]:
    """Descriptive alias for the future app-server command."""

    return build_app_argv(config, module)

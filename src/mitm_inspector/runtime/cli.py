"""Safe command-line planning surface for the local runtime."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol, cast

from mitm_inspector import __version__
from mitm_inspector.runtime.commands import build_app_argv, build_proxy_argv
from mitm_inspector.runtime.config import (
    DEFAULT_ADDON_PATH,
    DEFAULT_APP_EXECUTABLE,
    DEFAULT_MITMDUMP_EXECUTABLE,
    CaptureIPCConfig,
    RuntimeConfig,
    RuntimeConfigError,
    preflight_issues,
)
from mitm_inspector.runtime.readiness import HttpHealthReadinessProbe
from mitm_inspector.runtime.supervisor import RuntimeSupervisor, RuntimeSupervisorError
from mitm_inspector.store.sqlite import default_storage_path


class RunnableSupervisor(Protocol):
    def run(self) -> int: ...


def default_supervisor_factory(config: RuntimeConfig) -> RunnableSupervisor:
    return RuntimeSupervisor(
        config,
        readiness_probe=HttpHealthReadinessProbe(config),
    )


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--reverse-upstream", default="https://api.anthropic.com")
    parser.add_argument("--app-host", default="127.0.0.1")
    parser.add_argument("--proxy-host", default="127.0.0.1")
    parser.add_argument("--app-port", type=int, default=8000)
    parser.add_argument("--proxy-port", type=int, default=8080)
    parser.add_argument("--retention-max-flows", type=int, default=2_000)
    parser.add_argument("--retention-max-age-seconds", type=int, default=1_800)
    parser.add_argument("--max-body-bytes", type=int, default=128 * 1024 * 1024)
    parser.add_argument("--max-body-prefix-bytes", type=int, default=1024 * 1024)
    parser.add_argument("--max-pending-messages", type=int, default=4096)
    parser.add_argument("--storage-path", default=str(default_storage_path()))
    parser.add_argument("--no-storage", action="store_true")
    parser.add_argument("--storage-max-flows", type=int, default=10_000)
    parser.add_argument("--storage-max-bytes", type=int, default=512 * 1024 * 1024)
    parser.add_argument("--storage-replay", type=int, default=500)
    parser.add_argument("--mitmdump-executable", type=Path, default=DEFAULT_MITMDUMP_EXECUTABLE)
    parser.add_argument("--app-executable", type=Path, default=DEFAULT_APP_EXECUTABLE)
    parser.add_argument(
        "--addon-path",
        type=Path,
        default=DEFAULT_ADDON_PATH,
    )
    parser.add_argument("--capture-socket", type=Path, default=None)
    parser.add_argument("--source-id", default="mitm-inspector")
    parser.add_argument(
        "--open-browser",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="open the loopback app URL after both children are ready",
    )


def _config_from_args(args: argparse.Namespace) -> RuntimeConfig:
    return RuntimeConfig(
        reverse_upstream=args.reverse_upstream,
        app_host=args.app_host,
        proxy_host=args.proxy_host,
        app_port=args.app_port,
        proxy_port=args.proxy_port,
        retention_max_flows=args.retention_max_flows,
        retention_max_age_seconds=args.retention_max_age_seconds,
        max_body_bytes=args.max_body_bytes,
        max_body_prefix_bytes=args.max_body_prefix_bytes,
        max_pending_messages=args.max_pending_messages,
        storage_path=args.storage_path,
        no_storage=args.no_storage,
        storage_max_flows=args.storage_max_flows,
        storage_max_bytes=args.storage_max_bytes,
        storage_replay=args.storage_replay,
        mitmdump_executable=args.mitmdump_executable,
        app_executable=args.app_executable,
        addon_path=args.addon_path,
        capture_ipc=CaptureIPCConfig(
            socket_path=args.capture_socket or CaptureIPCConfig().socket_path,
            source_id=args.source_id,
            max_body_prefix_bytes=args.max_body_prefix_bytes,
            max_in_memory_bytes=args.max_body_bytes,
            max_pending_messages=args.max_pending_messages,
        ),
        open_browser=args.open_browser,
    )


def _plan(config: RuntimeConfig, addon_path: Path) -> dict[str, object]:
    host = f"[{config.app_host}]" if ":" in config.app_host else config.app_host
    issues = preflight_issues(config)
    return {
        "app": list(build_app_argv(config)),
        "proxy": list(build_proxy_argv(config, addon_path)),
        "shared_env": config.capture.environment(),
        "app_url": f"http://{host}:{config.app_port}/",
        "open_browser": config.open_browser,
        "start_order": ["app", "proxy"],
        "stop_order": ["proxy", "app"],
        "preflight": {
            "ok": not issues,
            "issues": list(issues),
        },
    }


def _print_plan(plan: dict[str, object], as_json: bool) -> None:
    if as_json:
        print(json.dumps(plan, indent=2))
        return
    app_argv = cast(list[str], plan["app"])
    proxy_argv = cast(list[str], plan["proxy"])
    start_order = cast(list[str], plan["start_order"])
    stop_order = cast(list[str], plan["stop_order"])
    shared_env = cast(dict[str, str], plan["shared_env"])
    print("app:", " ".join(app_argv))
    print("proxy:", " ".join(proxy_argv))
    print("app URL:", plan["app_url"])
    print("start order:", " -> ".join(start_order))
    print("stop order:", " -> ".join(stop_order))
    print("shared env:", "; ".join(f"{key}={value}" for key, value in shared_env.items()))
    print("browser:", "enabled" if plan["open_browser"] else "disabled")
    preflight = cast(dict[str, object], plan["preflight"])
    issues = cast(list[str], preflight["issues"])
    print("preflight:", "ok" if preflight["ok"] else "; ".join(issues))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mitm-inspector")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command")

    plan = subparsers.add_parser(
        "plan",
        help="print the child argv and topology without launching anything",
    )
    _add_runtime_arguments(plan)
    plan.add_argument("--json", action="store_true", help="emit a machine-readable plan")

    run = subparsers.add_parser(
        "run",
        help="start the local app server and stock mitmdump reverse proxy",
    )
    _add_runtime_arguments(run)
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan without launching anything",
    )
    run.add_argument("--json", action="store_true", help="emit a machine-readable plan")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    supervisor_factory: Callable[[RuntimeConfig], RunnableSupervisor] | None = None,
) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.error("choose 'plan' or 'run'")
    try:
        config = _config_from_args(args)
    except RuntimeConfigError as exc:
        parser.error(str(exc))
    plan = _plan(config, args.addon_path)
    if args.command == "plan" or args.dry_run:
        _print_plan(plan, args.json)
        return 0
    factory = supervisor_factory or default_supervisor_factory
    supervisor = factory(config)
    print("serving:", plan["app_url"])
    try:
        return supervisor.run()
    except RuntimeSupervisorError as exc:
        print(f"mitm-inspector: {exc}", file=sys.stderr)
        return 1

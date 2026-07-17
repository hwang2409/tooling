"""Safe command-line planning surface for the local runtime."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from mitm_inspector import __version__
from mitm_inspector.runtime.commands import build_app_argv, build_proxy_argv
from mitm_inspector.runtime.config import RuntimeConfig, RuntimeConfigError


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
    parser.add_argument("--mitmdump-executable", type=Path, default=Path("mitmdump"))
    parser.add_argument("--app-executable", type=Path, default=Path("python"))
    parser.add_argument(
        "--addon-path",
        type=Path,
        default=Path("src/mitm_inspector/capture/addon.py"),
    )
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
        mitmdump_executable=args.mitmdump_executable,
        app_executable=args.app_executable,
        open_browser=args.open_browser,
    )


def _plan(config: RuntimeConfig, addon_path: Path) -> dict[str, object]:
    host = f"[{config.app_host}]" if ":" in config.app_host else config.app_host
    return {
        "app": list(build_app_argv(config)),
        "proxy": list(build_proxy_argv(config, addon_path)),
        "app_url": f"http://{host}:{config.app_port}/",
        "open_browser": config.open_browser,
        "start_order": ["app", "proxy"],
        "stop_order": ["proxy", "app"],
    }


def _print_plan(plan: dict[str, object], as_json: bool) -> None:
    if as_json:
        print(json.dumps(plan, indent=2))
        return
    app_argv = cast(list[str], plan["app"])
    proxy_argv = cast(list[str], plan["proxy"])
    start_order = cast(list[str], plan["start_order"])
    stop_order = cast(list[str], plan["stop_order"])
    print("app:", " ".join(app_argv))
    print("proxy:", " ".join(proxy_argv))
    print("app URL:", plan["app_url"])
    print("start order:", " -> ".join(start_order))
    print("stop order:", " -> ".join(stop_order))
    print("browser:", "enabled" if plan["open_browser"] else "disabled")


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
        help="reserved for the B3 app server; dry-run is available now",
    )
    _add_runtime_arguments(run)
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan without launching anything",
    )
    run.add_argument("--json", action="store_true", help="emit a machine-readable plan")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.error("choose 'plan' or 'run --dry-run'; live run is deferred until B3")
    try:
        config = _config_from_args(args)
    except RuntimeConfigError as exc:
        parser.error(str(exc))
    plan = _plan(config, args.addon_path)
    if args.command == "plan" or args.dry_run:
        _print_plan(plan, args.json)
        return 0
    parser.error("live run is deferred until the B3 app server exists; use 'run --dry-run'")
    return 2

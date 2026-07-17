from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from mitm_inspector.runtime.cli import main
from mitm_inspector.runtime.commands import build_app_argv, build_proxy_argv
from mitm_inspector.runtime.config import RuntimeConfig, RuntimeConfigError
from mitm_inspector.runtime.supervisor import (
    ChildExitedError,
    ChildProcess,
    RuntimeState,
    RuntimeSupervisor,
)


class FakeChild:
    def __init__(self, pid: int, events: list[str]) -> None:
        self.pid = pid
        self.returncode: int | None = None
        self.events = events

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        if self.returncode is None:
            self.returncode = -9
        return self.returncode

    def terminate(self) -> None:
        self.events.append(f"terminate:{self.pid}")

    def kill(self) -> None:
        self.events.append(f"kill:{self.pid}")
        self.returncode = -9


class FakeFactory:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.children: list[FakeChild] = []
        self.argv: list[tuple[str, ...]] = []

    def spawn(self, argv: Sequence[str]) -> ChildProcess:
        self.argv.append(tuple(argv))
        child = FakeChild(len(self.children) + 1, self.events)
        self.children.append(child)
        self.events.append(f"spawn:{child.pid}")
        return child


class RecordingReadiness:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, float]] = []

    def wait_until_ready(
        self,
        component: str,
        child: ChildProcess,
        timeout_seconds: float,
    ) -> None:
        self.calls.append((component, child.pid, timeout_seconds))


def test_config_is_frozen_and_defaults_match_local_mvp() -> None:
    config = RuntimeConfig()

    assert config.reverse_upstream == "https://api.anthropic.com"
    assert config.app_host == config.proxy_host == "127.0.0.1"
    assert (config.app_port, config.proxy_port) == (8000, 8080)
    assert (config.max_flows, config.max_age_seconds) == (2_000, 1_800)
    assert (config.max_body_bytes, config.max_body_prefix_bytes) == (
        128 * 1024 * 1024,
        1024 * 1024,
    )
    with pytest.raises(AttributeError):
        config.app_port = 9000  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("reverse_upstream", "ftp://example.com", r"HTTP\(S\) URL"),
        ("reverse_upstream", "https://user:password@example.com", "credentials"),
        ("reverse_upstream", "https://example.com/path?token=secret", "query"),
        ("reverse_upstream", "https://example.com?", "query"),
        ("reverse_upstream", "https://example.com:99999", "valid port"),
        ("app_host", "0.0.0.0", "loopback"),
        ("proxy_host", "example.com", "loopback"),
        ("app_port", 0, "app_port"),
        ("proxy_port", 65536, "proxy_port"),
        ("app_port", 8080, "different"),
        ("max_body_prefix_bytes", 2, "max_body_prefix_bytes"),
    ],
)
def test_config_rejects_unsafe_or_conflicting_values(
    field: str,
    value: object,
    match: str,
) -> None:
    kwargs: dict[str, object] = {field: value}
    if field == "max_body_prefix_bytes":
        kwargs["max_body_bytes"] = 1
    if field == "app_port":
        kwargs["proxy_port"] = 8080
    with pytest.raises(RuntimeConfigError, match=match):
        RuntimeConfig(**kwargs)  # type: ignore[arg-type]


def test_command_builders_return_argv_without_shell_composition() -> None:
    config = RuntimeConfig(
        reverse_upstream="https://api.anthropic.com",
        mitmdump_executable=Path("/opt/bin/mitmdump"),
        app_executable=Path("/opt/bin/python"),
    )

    assert build_proxy_argv(config, Path("addon.py")) == (
        "/opt/bin/mitmdump",
        "--mode",
        "reverse:https://api.anthropic.com",
        "--listen-host",
        "127.0.0.1",
        "--listen-port",
        "8080",
        "-s",
        "addon.py",
    )
    assert build_app_argv(config)[:6] == (
        "/opt/bin/python",
        "-m",
        "mitm_inspector.api.server",
        "--host",
        "127.0.0.1",
        "--port",
    )
    assert all(isinstance(argument, str) for argument in build_proxy_argv(config, "addon.py"))


def test_supervisor_starts_app_before_proxy_and_stops_in_reverse_order() -> None:
    events: list[str] = []
    factory = FakeFactory(events)
    readiness = RecordingReadiness()
    supervisor = RuntimeSupervisor(
        RuntimeConfig(),
        process_factory=factory,
        readiness_probe=readiness,
        sleeper=lambda _seconds: None,
    )

    supervisor.start()
    assert supervisor.state is RuntimeState.RUNNING
    assert events == ["spawn:1", "spawn:2"]
    assert [call[0] for call in readiness.calls] == ["app", "proxy"]

    supervisor.stop()
    supervisor.stop()
    assert supervisor.state is RuntimeState.STOPPED
    assert events[-4:] == ["terminate:2", "terminate:1", "kill:2", "kill:1"]


def test_proxy_readiness_failure_cleans_up_app_and_proxy() -> None:
    events: list[str] = []
    factory = FakeFactory(events)

    class FailingReadiness(RecordingReadiness):
        def wait_until_ready(
            self,
            component: str,
            child: ChildProcess,
            timeout_seconds: float,
        ) -> None:
            super().wait_until_ready(component, child, timeout_seconds)
            if component == "proxy":
                raise RuntimeError("proxy did not become ready")

    supervisor = RuntimeSupervisor(
        RuntimeConfig(),
        process_factory=factory,
        readiness_probe=FailingReadiness(),
        sleeper=lambda _seconds: None,
    )

    with pytest.raises(RuntimeError, match="did not become ready"):
        supervisor.start()
    assert supervisor.state is RuntimeState.FAILED
    assert events[-4:] == ["terminate:2", "terminate:1", "kill:2", "kill:1"]


def test_unexpected_child_exit_is_propagated_after_sibling_cleanup() -> None:
    events: list[str] = []
    factory = FakeFactory(events)

    class ExitingReadiness(RecordingReadiness):
        def wait_until_ready(
            self,
            component: str,
            child: ChildProcess,
            timeout_seconds: float,
        ) -> None:
            super().wait_until_ready(component, child, timeout_seconds)
            if component == "proxy":
                factory.children[1].returncode = 23

    supervisor = RuntimeSupervisor(
        RuntimeConfig(),
        process_factory=factory,
        readiness_probe=ExitingReadiness(),
        sleeper=lambda _seconds: None,
    )

    with pytest.raises(ChildExitedError, match="proxy exited with status 23") as raised:
        supervisor.run()
    assert raised.value.component == "proxy"
    assert supervisor.state is RuntimeState.STOPPED
    assert "terminate:1" in events


def test_keyboard_cancellation_uses_cleanup_path() -> None:
    events: list[str] = []
    factory = FakeFactory(events)
    supervisor: RuntimeSupervisor

    def interrupt(_seconds: float) -> None:
        del _seconds
        raise KeyboardInterrupt

    supervisor = RuntimeSupervisor(
        RuntimeConfig(),
        process_factory=factory,
        readiness_probe=RecordingReadiness(),
        sleeper=interrupt,
    )

    assert supervisor.run() == 130
    assert supervisor.state is RuntimeState.STOPPED
    assert events[-4:] == ["terminate:2", "terminate:1", "kill:2", "kill:1"]


def test_browser_opener_is_injected_and_called_only_after_startup() -> None:
    events: list[str] = []
    factory = FakeFactory(events)
    opened: list[str] = []

    def open_browser(url: str) -> bool:
        opened.append(url)
        return True

    supervisor = RuntimeSupervisor(
        RuntimeConfig(
            open_browser=True,
            graceful_shutdown_seconds=0.001,
            kill_wait_seconds=0.001,
        ),
        process_factory=factory,
        readiness_probe=RecordingReadiness(),
        browser_opener=open_browser,
        sleeper=lambda _seconds: None,
    )

    supervisor.start()
    assert opened == ["http://127.0.0.1:8000/"]
    supervisor.stop()


def test_cli_plan_and_dry_run_do_not_launch_anything(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["plan", "--reverse-upstream", "https://api.anthropic.com", "--json"]) == 0
    output = capsys.readouterr().out
    assert '"reverse:https://api.anthropic.com"' in output
    assert '"8000"' in output

    assert main(["run", "--dry-run"]) == 0
    assert "start order: app -> proxy" in capsys.readouterr().out

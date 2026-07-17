from __future__ import annotations

import math
import os
import signal
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path

import pytest

from mitm_inspector.runtime.cli import main
from mitm_inspector.runtime.commands import (
    ProcessSpec,
    build_app_argv,
    build_app_spec,
    build_proxy_argv,
    build_proxy_spec,
)
from mitm_inspector.runtime.config import (
    CAPTURE_MAX_BODY_PREFIX_ENV,
    CAPTURE_MAX_MEMORY_ENV,
    CAPTURE_SOCKET_ENV,
    CAPTURE_SOURCE_ID_ENV,
    DEFAULT_ADDON_PATH,
    DEFAULT_APP_EXECUTABLE,
    DEFAULT_MITMDUMP_EXECUTABLE,
    CaptureIPCConfig,
    RuntimeConfig,
    RuntimeConfigError,
    RuntimePreflightError,
    preflight_issues,
)
from mitm_inspector.runtime.supervisor import (
    BrowserOpener,
    ChildExitedError,
    ChildProcess,
    CleanupError,
    RuntimeState,
    RuntimeSupervisor,
    SignalHandlingError,
    SignalPolicy,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class FakeChild:
    def __init__(
        self,
        pid: int,
        events: list[str],
        *,
        terminate_exits: bool = False,
        kill_exits: bool = True,
        leader_exits_on_terminate: bool = False,
    ) -> None:
        self.pid = pid
        self.process_group_id = pid + 1000
        self.returncode: int | None = None
        self._group_alive = True
        self.events = events
        self.terminate_exits = terminate_exits
        self.kill_exits = kill_exits
        self.leader_exits_on_terminate = leader_exits_on_terminate
        self.terminate_raises: BaseException | None = None
        self.kill_raises: BaseException | None = None
        self.on_terminate: Callable[[], None] | None = None

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is None:
            if timeout == 0:
                raise subprocess.TimeoutExpired("fake", timeout)
            self.returncode = -9
        return self.returncode

    def group_alive(self) -> bool:
        return self._group_alive

    def terminate(self) -> None:
        self.events.append(f"terminate:{self.pid}")
        if self.on_terminate is not None:
            self.on_terminate()
        if self.terminate_raises is not None:
            raise self.terminate_raises
        if self.leader_exits_on_terminate:
            self.returncode = -15
        if self.terminate_exits:
            self.returncode = -15
            self._group_alive = False

    def kill(self) -> None:
        self.events.append(f"kill:{self.pid}")
        if self.kill_raises is not None:
            raise self.kill_raises
        if self.kill_exits:
            self.returncode = -9
            self._group_alive = False


class FakeFactory:
    def __init__(
        self,
        events: list[str],
        child_factory: Callable[[int, list[str]], FakeChild] | None = None,
    ) -> None:
        self.events = events
        self.children: list[FakeChild] = []
        self.specs: list[ProcessSpec] = []
        self.child_factory = child_factory or (lambda pid, seen: FakeChild(pid, seen))
        self.on_spawn: Callable[[ProcessSpec], None] | None = None

    def spawn(self, spec: ProcessSpec) -> ChildProcess:
        self.specs.append(spec)
        if self.on_spawn is not None:
            self.on_spawn(spec)
        child = self.child_factory(len(self.children) + 1, self.events)
        self.children.append(child)
        self.events.append(f"spawn:{child.pid}")
        return child


class RecordingReadiness:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, float]] = []
        self.on_ready: Callable[[str], None] | None = None

    def wait_until_ready(
        self,
        component: str,
        child: ChildProcess,
        timeout_seconds: float,
    ) -> None:
        self.calls.append((component, child.pid, timeout_seconds))
        if self.on_ready is not None:
            self.on_ready(component)


def browser_returns_false(_url: str) -> bool:
    return False


def browser_raises(_url: str) -> bool:
    raise RuntimeError("no browser")


def runtime_config(tmp_path: Path, **overrides: object) -> RuntimeConfig:
    values: dict[str, object] = {
        "app_executable": tmp_path / "python",
        "mitmdump_executable": tmp_path / "mitmdump",
        "addon_path": tmp_path / "addon.py",
        "readiness_timeout_seconds": 1.0,
        "graceful_shutdown_seconds": 1.0,
        "kill_wait_seconds": 1.0,
        "poll_interval_seconds": 0.25,
    }
    values.update(overrides)
    return RuntimeConfig(**values)  # type: ignore[arg-type]


def fake_supervisor(
    tmp_path: Path,
    factory: FakeFactory,
    *,
    readiness: RecordingReadiness | None = None,
    config: RuntimeConfig | None = None,
    clock: FakeClock | None = None,
    **kwargs: object,
) -> RuntimeSupervisor:
    fake_clock = clock or FakeClock()
    return RuntimeSupervisor(
        config or runtime_config(tmp_path),
        process_factory=factory,
        readiness_probe=readiness or RecordingReadiness(),
        preflight_checker=lambda _config: None,
        clock=fake_clock.clock,
        sleeper=fake_clock.sleep,
        **kwargs,
    )


def test_config_is_frozen_relocatable_and_shares_ipc_limits() -> None:
    config = RuntimeConfig()

    assert config.reverse_upstream == "https://api.anthropic.com"
    assert config.app_executable == Path(sys.executable) == DEFAULT_APP_EXECUTABLE
    assert config.mitmdump_executable == DEFAULT_MITMDUMP_EXECUTABLE
    assert config.addon_path == DEFAULT_ADDON_PATH
    assert config.addon_path.is_absolute() and config.addon_path.is_file()
    assert config.capture.max_body_prefix_bytes == config.max_body_prefix_bytes
    assert config.capture.max_in_memory_bytes == config.max_body_bytes
    assert config.capture.socket_path.is_absolute()
    with pytest.raises(AttributeError):
        config.app_port = 9000  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("reverse_upstream", "ftp://example.com", "http or https"),
        ("reverse_upstream", "https://user:password@example.com", "credentials"),
        ("reverse_upstream", "https://example.com/path", "path"),
        ("reverse_upstream", "https://example.com?token=secret", "query"),
        ("reverse_upstream", "https://example.com:", "empty port"),
        ("reverse_upstream", "https://example.com:99999", "valid port"),
        ("reverse_upstream", "https://bad host", "authority"),
        ("app_host", "0.0.0.0", "loopback"),
        ("proxy_host", "example.com", "loopback"),
        ("app_port", 0, "app_port"),
        ("proxy_port", 65536, "proxy_port"),
        ("app_port", 8080, "different"),
        ("max_body_prefix_bytes", 2, "max_body_prefix_bytes"),
    ],
)
def test_config_rejects_unsafe_or_conflicting_values(
    tmp_path: Path,
    field: str,
    value: object,
    match: str,
) -> None:
    kwargs: dict[str, object] = {field: value}
    if field == "max_body_prefix_bytes":
        kwargs["max_body_bytes"] = 1
    if field == "app_port":
        kwargs["proxy_port"] = 8080
    kwargs.update(
        app_executable=tmp_path / "python",
        mitmdump_executable=tmp_path / "mitmdump",
        addon_path=tmp_path / "addon.py",
    )
    with pytest.raises(RuntimeConfigError, match=match):
        RuntimeConfig(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("field", [
    "readiness_timeout_seconds",
    "graceful_shutdown_seconds",
    "kill_wait_seconds",
    "poll_interval_seconds",
])
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, 0.0, -1.0])
def test_config_rejects_nonfinite_or_nonpositive_timeouts(
    tmp_path: Path,
    field: str,
    value: float,
) -> None:
    with pytest.raises(RuntimeConfigError, match="finite positive"):
        runtime_config(tmp_path, **{field: value})


def test_uppercase_scheme_is_normalized_and_matches_pinned_mitmproxy_parser() -> None:
    from mitmproxy.proxy.mode_specs import ProxyMode

    for target in (
        "HTTPS://api.anthropic.com",
        "http://Example.com:8080",
        "https://127.0.0.1",
        "https://[::1]:8443",
    ):
        config = RuntimeConfig(reverse_upstream=target)
        assert config.reverse_upstream.startswith(("http://", "https://"))
        parsed = ProxyMode.parse(f"reverse:{config.reverse_upstream}")
        assert parsed.data == config.reverse_upstream


@pytest.mark.parametrize(
    "target",
    [
        "https://user@example.com",
        "https://example.com/path",
        "https://example.com?query",
        "https://example.com:",
    ],
)
def test_rejected_targets_are_also_rejected_by_pinned_mitmproxy_parser(target: str) -> None:
    from mitmproxy.proxy.mode_specs import ProxyMode

    with pytest.raises(RuntimeConfigError):
        RuntimeConfig(reverse_upstream=target)
    with pytest.raises(ValueError):
        ProxyMode.parse(f"reverse:{target}")


def test_process_specs_share_exact_ipc_environment_and_packaged_addon(tmp_path: Path) -> None:
    config = runtime_config(
        tmp_path,
        capture_ipc=CaptureIPCConfig(
            socket_path=Path("/tmp/mitm-inspector-test.sock"),
            source_id="synthetic-source",
            max_body_prefix_bytes=1024,
            max_in_memory_bytes=4096,
        ),
        max_body_prefix_bytes=1024,
        max_body_bytes=4096,
    )
    proxy = build_proxy_spec(config)
    app = build_app_spec(config)

    assert proxy.env == app.env == {
        CAPTURE_SOCKET_ENV: "/tmp/mitm-inspector-test.sock",
        CAPTURE_SOURCE_ID_ENV: "synthetic-source",
        CAPTURE_MAX_BODY_PREFIX_ENV: "1024",
        CAPTURE_MAX_MEMORY_ENV: "4096",
    }
    assert proxy.argv[-2:] == ("-s", str(config.addon_path))
    assert Path(proxy.argv[-1]).is_absolute()
    assert str(config.capture.socket_path) in app.argv
    assert "--capture-source-id" in app.argv
    with pytest.raises(TypeError):
        proxy.env[CAPTURE_SOURCE_ID_ENV] = "changed"  # type: ignore[index]


def test_argv_builders_are_direct_vectors(tmp_path: Path) -> None:
    config = runtime_config(tmp_path)
    assert build_proxy_argv(config) == config_spec(config, "proxy").argv
    assert build_app_argv(config) == config_spec(config, "app").argv
    assert all(isinstance(argument, str) for argument in build_proxy_argv(config))


def config_spec(config: RuntimeConfig, name: str) -> ProcessSpec:
    return build_proxy_spec(config) if name == "proxy" else build_app_spec(config)


def test_supervisor_starts_app_then_fully_stops_proxy_before_app(tmp_path: Path) -> None:
    events: list[str] = []

    def children(pid: int, seen: list[str]) -> FakeChild:
        return FakeChild(pid, seen, terminate_exits=pid == 1)

    factory = FakeFactory(events, children)
    supervisor = fake_supervisor(tmp_path, factory)

    supervisor.start()
    supervisor.stop()
    supervisor.stop()

    assert supervisor.state is RuntimeState.STOPPED
    assert events == [
        "spawn:1",
        "spawn:2",
        "terminate:2",
        "kill:2",
        "terminate:1",
    ]


def test_proxy_leader_exit_does_not_skip_descendant_process_group_kill(tmp_path: Path) -> None:
    events: list[str] = []

    def children(pid: int, seen: list[str]) -> FakeChild:
        return FakeChild(
            pid,
            seen,
            terminate_exits=pid == 1,
            leader_exits_on_terminate=pid == 2,
        )

    factory = FakeFactory(events, children)
    supervisor = fake_supervisor(tmp_path, factory)
    supervisor.start()
    supervisor.stop()

    assert events == ["spawn:1", "spawn:2", "terminate:2", "kill:2", "terminate:1"]
    assert factory.children[1].process_group_id == 1002


def test_cleanup_exception_still_cleans_sibling_and_aggregates(tmp_path: Path) -> None:
    events: list[str] = []

    def children(pid: int, seen: list[str]) -> FakeChild:
        child = FakeChild(pid, seen, terminate_exits=True)
        if pid == 2:
            child.terminate_raises = RuntimeError("terminate boom")
        return child

    factory = FakeFactory(events, children)
    supervisor = fake_supervisor(tmp_path, factory)
    supervisor.start()

    with pytest.raises(CleanupError) as raised:
        supervisor.stop()
    assert raised.value.failures[0].component == "proxy"
    assert "terminate boom" in str(raised.value)
    assert "terminate:1" in events
    assert supervisor.state is RuntimeState.FAILED


def test_surviving_group_is_failure_and_second_stop_retries_without_false_success(
    tmp_path: Path,
) -> None:
    events: list[str] = []

    def children(pid: int, seen: list[str]) -> FakeChild:
        return FakeChild(pid, seen, terminate_exits=pid == 1, kill_exits=False)

    factory = FakeFactory(events, children)
    supervisor = fake_supervisor(tmp_path, factory)
    supervisor.start()

    with pytest.raises(CleanupError, match="survived kill deadline"):
        supervisor.stop()
    assert supervisor.state is RuntimeState.FAILED
    assert supervisor.cleanup_failures[0].group_survived
    first_count = len(events)

    with pytest.raises(CleanupError):
        supervisor.stop()
    assert len(events) > first_count
    assert supervisor.state is RuntimeState.FAILED


def test_unexpected_child_exit_is_propagated_after_sibling_cleanup(tmp_path: Path) -> None:
    events: list[str] = []

    def children(pid: int, seen: list[str]) -> FakeChild:
        child = FakeChild(pid, seen, terminate_exits=True)
        if pid == 2:
            child.returncode = 23
            child._group_alive = False
        return child

    factory = FakeFactory(events, children)
    supervisor = fake_supervisor(tmp_path, factory)

    with pytest.raises(ChildExitedError, match="proxy exited with status 23"):
        supervisor.run()
    assert "terminate:1" in events
    assert supervisor.state is RuntimeState.STOPPED


def test_signal_handlers_cover_startup_and_map_sigint_to_130(tmp_path: Path) -> None:
    events: list[str] = []
    readiness = RecordingReadiness()
    supervisor: RuntimeSupervisor

    def signal_during_app(component: str) -> None:
        if component == "app":
            supervisor.handle_signal(signal.SIGINT)

    readiness.on_ready = signal_during_app
    factory = FakeFactory(events)
    supervisor = fake_supervisor(tmp_path, factory, readiness=readiness)

    assert supervisor.run() == 130
    assert [spec.name for spec in factory.specs] == ["app"]
    assert supervisor.state is RuntimeState.STOPPED


def test_signal_during_steady_state_maps_sigterm_to_143(tmp_path: Path) -> None:
    events: list[str] = []
    factory = FakeFactory(events)
    clock = FakeClock()
    supervisor = fake_supervisor(tmp_path, factory, clock=clock)

    def stop_on_first_sleep(_seconds: float) -> None:
        supervisor.handle_signal(signal.SIGTERM)
        clock.sleep(_seconds)

    supervisor._sleep = stop_on_first_sleep  # type: ignore[attr-defined]
    assert supervisor.run() == 143
    assert supervisor.state is RuntimeState.STOPPED


def test_signal_handler_remains_installed_through_shutdown_and_is_restored(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    before = signal.getsignal(signal.SIGTERM)
    captured: list[object] = []

    def children(pid: int, seen: list[str]) -> FakeChild:
        child = FakeChild(pid, seen, terminate_exits=True)
        handler = signal.getsignal(signal.SIGTERM)
        captured.append(handler)
        if callable(handler):
            child.on_terminate = lambda: handler(signal.SIGTERM, None)
        return child

    factory = FakeFactory(events, children)
    clock = FakeClock()
    supervisor = fake_supervisor(tmp_path, factory, clock=clock)

    def request_shutdown(_seconds: float) -> None:
        supervisor.request_stop("test", 0)
        clock.sleep(_seconds)

    supervisor._sleep = request_shutdown  # type: ignore[attr-defined]
    assert supervisor.run() == 143
    assert captured and captured[0] is not before
    assert signal.getsignal(signal.SIGTERM) is before


def test_signal_policy_fails_clearly_off_main_thread(tmp_path: Path) -> None:
    errors: list[BaseException] = []
    factory = FakeFactory([])
    supervisor = fake_supervisor(tmp_path, factory)

    def run() -> None:
        try:
            supervisor.run()
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    thread.join()
    assert isinstance(errors[0], SignalHandlingError)
    assert factory.specs == []
    assert supervisor.state is RuntimeState.FAILED


def test_non_main_thread_requires_explicit_disabled_policy(tmp_path: Path) -> None:
    events: list[str] = []
    factory = FakeFactory(events)
    supervisor = fake_supervisor(
        tmp_path,
        factory,
        signal_policy=SignalPolicy.DISABLED_FOR_TEST,
    )
    supervisor.request_stop("test", 0)
    assert supervisor.run() == 0
    assert supervisor.state is RuntimeState.STOPPED


@pytest.mark.parametrize("opener", [browser_returns_false, browser_raises])
def test_browser_failure_is_nonfatal_but_warning_is_observable(
    tmp_path: Path,
    opener: BrowserOpener,
) -> None:
    events: list[str] = []
    warnings_seen: list[str] = []
    factory = FakeFactory(events)
    config = runtime_config(tmp_path, open_browser=True)
    supervisor = fake_supervisor(
        tmp_path,
        factory,
        config=config,
        browser_opener=opener,
        warning_sink=warnings_seen.append,
    )

    supervisor.start()
    assert supervisor.state is RuntimeState.RUNNING
    assert warnings_seen
    supervisor.stop()


def test_browser_is_not_called_when_disabled(tmp_path: Path) -> None:
    called: list[str] = []
    factory = FakeFactory([])
    supervisor = fake_supervisor(
        tmp_path,
        factory,
        browser_opener=lambda url: called.append(url) is None,
    )

    supervisor.start()
    assert called == []
    supervisor.stop()


def test_preflight_reports_and_rejects_missing_or_nonexecutable_paths(tmp_path: Path) -> None:
    config = runtime_config(tmp_path)
    assert preflight_issues(config)
    with pytest.raises(RuntimePreflightError):
        from mitm_inspector.runtime.config import validate_preflight

        validate_preflight(config)

    for path in (config.app_executable, config.mitmdump_executable, config.addon_path):
        path.write_text("fixture")
    os.chmod(config.app_executable, 0o755)
    os.chmod(config.mitmdump_executable, 0o755)
    assert preflight_issues(config) == ()


def test_live_start_runs_preflight_before_spawning(tmp_path: Path) -> None:
    factory = FakeFactory([])
    supervisor = RuntimeSupervisor(
        runtime_config(tmp_path),
        process_factory=factory,
        readiness_probe=RecordingReadiness(),
    )

    with pytest.raises(RuntimePreflightError):
        supervisor.start()
    assert factory.specs == []
    assert supervisor.state is RuntimeState.FAILED


def test_cli_plan_reports_preflight_and_never_launches(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["plan", "--json"]) == 0
    output = capsys.readouterr().out
    assert '"reverse:https://api.anthropic.com"' in output
    assert '"preflight"' in output
    assert '"8000"' in output

    assert main(["run", "--dry-run"]) == 0
    assert "start order: app -> proxy" in capsys.readouterr().out

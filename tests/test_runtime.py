from __future__ import annotations

import math
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from mitm_inspector.api.limits import MAX_INGEST_BODY_PREFIX_BYTES
from mitm_inspector.runtime.cli import default_supervisor_factory, main
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
    CAPTURE_MAX_PENDING_ENV,
    CAPTURE_SOCKET_ENV,
    CAPTURE_SOURCE_ID_ENV,
    DEFAULT_ADDON_PATH,
    DEFAULT_APP_EXECUTABLE,
    DEFAULT_MITMDUMP_EXECUTABLE,
    MAX_UINT64,
    CaptureIPCConfig,
    RuntimeConfig,
    RuntimeConfigError,
    RuntimePreflightError,
    cleanup_private_runtime_dir,
    preflight_issues,
    validate_private_runtime_dir,
)
from mitm_inspector.runtime.readiness import (
    HttpHealthReadinessProbe,
    ReadinessTimeoutError,
)
from mitm_inspector.runtime.supervisor import (
    BrowserOpener,
    ChildExitedError,
    ChildProcess,
    CleanupError,
    ImmediateReadinessProbe,
    PopenChild,
    ProcessGroupProbeError,
    RuntimeState,
    RuntimeSupervisor,
    RuntimeSupervisorError,
    SignalHandlingError,
    SignalPolicy,
    SubprocessFactory,
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


def test_oversized_timeout_is_a_runtime_config_error(tmp_path: Path) -> None:
    with pytest.raises(RuntimeConfigError, match="finite positive"):
        runtime_config(tmp_path, readiness_timeout_seconds=10**1000)


@pytest.mark.parametrize("field", [
    "retention_max_flows",
    "retention_max_age_seconds",
    "max_body_bytes",
    "max_body_prefix_bytes",
    "max_pending_messages",
])
def test_integer_caps_are_uint64_bounded(tmp_path: Path, field: str) -> None:
    accepted_value = MAX_UINT64
    accepted: dict[str, object] = {}
    if field == "max_body_prefix_bytes":
        # The body prefix is additionally bounded by the ingest line capacity.
        accepted_value = MAX_INGEST_BODY_PREFIX_BYTES
        accepted["max_body_bytes"] = MAX_UINT64
    accepted[field] = accepted_value
    config = runtime_config(tmp_path, **accepted)
    assert getattr(config, field) == accepted_value
    rejected = dict(accepted)
    rejected[field] = MAX_UINT64 + 1
    with pytest.raises(RuntimeConfigError, match="unsigned 64-bit"):
        runtime_config(tmp_path, **rejected)


def test_zero_body_prefix_is_supported_by_shared_contract(tmp_path: Path) -> None:
    config = runtime_config(tmp_path, max_body_prefix_bytes=0, max_body_bytes=1)
    assert config.capture.max_body_prefix_bytes == 0
    assert config.capture.environment()[CAPTURE_MAX_BODY_PREFIX_ENV] == "0"
    for spec in (build_proxy_spec(config), build_app_spec(config)):
        assert _parse_b2_capture_environment(dict(spec.env))[2:] == (0, 1, 4096)


@pytest.mark.parametrize(
    "field",
    [
        "retention_max_flows",
        "retention_max_age_seconds",
        "max_body_bytes",
        "max_pending_messages",
    ],
)
def test_b2_nonzero_uint_ranges_reject_zero(tmp_path: Path, field: str) -> None:
    with pytest.raises(RuntimeConfigError, match="unsigned 64-bit"):
        runtime_config(tmp_path, **{field: 0})


@pytest.mark.parametrize("field", ["max_in_memory_bytes", "max_pending_messages"])
def test_b2_capture_environment_nonzero_ranges_reject_zero(field: str) -> None:
    with pytest.raises(RuntimeConfigError, match="unsigned 64-bit"):
        CaptureIPCConfig(**{field: 0})


def _parse_b2_capture_environment(env: dict[str, str]) -> tuple[str, str, int, int, int]:
    """Test-only copy of B2 CaptureConfig.from_environment's five-field seam."""

    socket_path = env[CAPTURE_SOCKET_ENV]
    source_id = env[CAPTURE_SOURCE_ID_ENV]
    assert socket_path.startswith("/") and "\x00" not in socket_path
    assert source_id and "\x00" not in source_id
    parsed: list[int] = []
    for name, allow_zero in (
        (CAPTURE_MAX_BODY_PREFIX_ENV, True),
        (CAPTURE_MAX_MEMORY_ENV, False),
        (CAPTURE_MAX_PENDING_ENV, False),
    ):
        value = env[name]
        assert re.fullmatch(r"(?:0|[1-9][0-9]*)", value)
        number = int(value)
        assert number <= MAX_UINT64
        assert allow_zero or number >= 1
        parsed.append(number)
    assert parsed[0] <= parsed[1]
    return socket_path, source_id, *parsed


def test_lowercase_authority_matches_pinned_mitmproxy_parser() -> None:
    from mitmproxy.proxy.mode_specs import ProxyMode

    for target in (
        "http://Example.com:8080",
        "https://127.0.0.1",
        "https://[::1]:8443",
    ):
        config = RuntimeConfig(reverse_upstream=target)
        assert config.reverse_upstream == target
        parsed = ProxyMode.parse(f"reverse:{config.reverse_upstream}")
        assert parsed.data == config.reverse_upstream


def test_uppercase_scheme_is_rejected_without_repair() -> None:
    from mitmproxy.proxy.mode_specs import ProxyMode

    target = "HTTPS://api.anthropic.com"
    with pytest.raises(RuntimeConfigError, match="lowercase"):
        RuntimeConfig(reverse_upstream=target)
    with pytest.raises(ValueError):
        ProxyMode.parse(f"reverse:{target}")


@pytest.mark.parametrize(
    "target",
    [
        "https://user@example.com",
        "https://example.com/path",
        "https://example.com?query",
        "https://example.com?",
        "https://example.com#",
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
        CAPTURE_MAX_PENDING_ENV: "4096",
    }
    assert _parse_b2_capture_environment(dict(proxy.env)) == (
        "/tmp/mitm-inspector-test.sock",
        "synthetic-source",
        1024,
        4096,
        4096,
    )
    assert proxy.argv[-2:] == ("-s", str(config.addon_path))
    assert Path(proxy.argv[-1]).is_absolute()
    assert str(config.capture.socket_path) in app.argv
    assert "--capture-source-id" in app.argv
    assert "--capture-max-pending-messages" in app.argv
    with pytest.raises(TypeError):
        proxy.env[CAPTURE_SOURCE_ID_ENV] = "changed"  # type: ignore[index]


def test_argv_builders_are_direct_vectors(tmp_path: Path) -> None:
    config = runtime_config(tmp_path)
    assert build_proxy_argv(config) == config_spec(config, "proxy").argv
    assert build_app_argv(config) == config_spec(config, "app").argv
    assert all(isinstance(argument, str) for argument in build_proxy_argv(config))


class ProbeProcess:
    pid = 12345

    def poll(self) -> int | None:
        return 17

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        return 17

    def terminate(self) -> None:
        raise AssertionError("fallback terminate must not be called")

    def kill(self) -> None:
        raise AssertionError("fallback kill must not be called")


def test_reaped_group_eperm_reprobe_esrch_is_gone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mitm_inspector.runtime.supervisor.os.getpgid", lambda _pid: 54321)
    probes: list[int] = []

    def killpg(_pgid: int, signum: int) -> None:
        probes.append(signum)
        if len(probes) == 1:
            raise PermissionError(1, "operation not permitted")
        raise ProcessLookupError

    monkeypatch.setattr("mitm_inspector.runtime.supervisor.os.killpg", killpg)
    child = PopenChild(ProbeProcess())  # type: ignore[arg-type]

    assert child.group_alive() is False
    assert probes == [0, 0]


def test_reaped_group_persistent_eperm_is_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mitm_inspector.runtime.supervisor.os.getpgid", lambda _pid: 54321)
    probes: list[int] = []

    def killpg(_pgid: int, signum: int) -> None:
        probes.append(signum)
        raise PermissionError(1, "operation not permitted")

    monkeypatch.setattr("mitm_inspector.runtime.supervisor.os.killpg", killpg)
    child = PopenChild(ProbeProcess())  # type: ignore[arg-type]

    with pytest.raises(ProcessGroupProbeError, match="could not determine"):
        child.group_alive()
    assert probes == [0] * PopenChild._GROUP_PROBE_ATTEMPTS


def test_signal_eperm_then_group_esrch_is_not_a_cleanup_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("mitm_inspector.runtime.supervisor.os.getpgid", lambda _pid: 54321)
    probes: list[int] = []

    def killpg(_pgid: int, signum: int) -> None:
        probes.append(signum)
        if signum == signal.SIGTERM:
            raise PermissionError(1, "operation not permitted")
        raise ProcessLookupError

    monkeypatch.setattr("mitm_inspector.runtime.supervisor.os.killpg", killpg)
    child = PopenChild(ProbeProcess())  # type: ignore[arg-type]

    child.terminate()
    assert probes == [signal.SIGTERM, 0]


def test_capture_allocations_are_unique_private_and_cleanable() -> None:
    capture = CaptureIPCConfig()
    first = capture.allocate()
    second = capture.allocate()
    try:
        assert first.runtime_dir is not None
        assert second.runtime_dir is not None
        assert first.runtime_dir != second.runtime_dir
        assert first.socket_path.parent == first.runtime_dir
        assert second.socket_path.parent == second.runtime_dir
        assert first.runtime_dir.stat().st_mode & 0o777 == 0o700
        assert second.runtime_dir.stat().st_mode & 0o777 == 0o700
        validate_private_runtime_dir(first.runtime_dir)
        validate_private_runtime_dir(second.runtime_dir)
    finally:
        assert cleanup_private_runtime_dir(first) == ()
        assert cleanup_private_runtime_dir(second) == ()
    assert not first.runtime_dir.exists()
    assert not second.runtime_dir.exists()


@pytest.mark.skipif(os.name != "posix", reason="private Unix socket adversary test")
def test_capture_cleanup_rejects_symlink_and_wrong_mode(tmp_path: Path) -> None:
    # Keep the Unix socket path below macOS's sockaddr_un limit; the adversary
    # files themselves can still live under pytest's longer temporary path.
    capture = CaptureIPCConfig(runtime_base_dir=Path(tempfile.gettempdir()).resolve()).allocate()
    assert capture.runtime_dir is not None
    endpoint = capture.socket_path
    outside = tmp_path / "outside"
    outside.write_text("must survive")
    try:
        endpoint.symlink_to(outside)
        issues = cleanup_private_runtime_dir(capture)
        assert issues and "symlink" in issues[0]
        assert outside.exists()
        endpoint.unlink()
        endpoint.write_text("not a socket")
        issues = cleanup_private_runtime_dir(capture)
        assert issues and "non-socket" in issues[0]
        endpoint.unlink()
        os.chmod(capture.runtime_dir, 0o755)
        issues = cleanup_private_runtime_dir(capture)
        assert issues and "mode 0700" in issues[0]
        os.chmod(capture.runtime_dir, 0o700)
    finally:
        if endpoint.is_symlink() or endpoint.exists():
            endpoint.unlink()
        assert cleanup_private_runtime_dir(capture) == ()


def test_capture_cleanup_removes_a_real_socket_and_refuses_non_socket(tmp_path: Path) -> None:
    capture = CaptureIPCConfig(runtime_base_dir=Path(tempfile.gettempdir()).resolve()).allocate()
    assert capture.runtime_dir is not None
    endpoint = capture.socket_path
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        listener.bind(str(endpoint))
        listener.close()
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.close()
        assert cleanup_private_runtime_dir(capture) == ()
    finally:
        listener.close()
        if capture.runtime_dir.exists():
            endpoint.unlink(missing_ok=True)
            capture.runtime_dir.rmdir()


def test_supervisor_uses_one_ephemeral_ipc_endpoint_and_cleans_it(tmp_path: Path) -> None:
    events: list[str] = []
    factory = FakeFactory(events, lambda pid, seen: FakeChild(pid, seen, terminate_exits=True))
    supervisor = fake_supervisor(tmp_path, factory)

    supervisor.start()
    assert len(factory.specs) == 2
    proxy_socket = factory.specs[1].env[CAPTURE_SOCKET_ENV]
    app_socket = factory.specs[0].env[CAPTURE_SOCKET_ENV]
    runtime_dir = Path(proxy_socket).parent
    assert proxy_socket == app_socket
    assert runtime_dir.stat().st_mode & 0o777 == 0o700
    supervisor.stop()
    assert not runtime_dir.exists()


def test_supervisor_cleans_ephemeral_ipc_after_start_failure(tmp_path: Path) -> None:
    events: list[str] = []

    class FailsOnProxy(FakeFactory):
        def spawn(self, spec: ProcessSpec) -> ChildProcess:
            if len(self.specs) == 1:
                self.specs.append(spec)
                raise OSError("synthetic proxy spawn failure")
            return super().spawn(spec)

    factory = FailsOnProxy(
        events,
        lambda pid, seen: FakeChild(pid, seen, terminate_exits=True),
    )
    supervisor = fake_supervisor(tmp_path, factory)
    with pytest.raises(OSError, match="synthetic proxy spawn failure"):
        supervisor.start()
    app_socket = factory.specs[0].env[CAPTURE_SOCKET_ENV]
    assert not Path(app_socket).parent.exists()


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


def test_persistent_group_probe_eperm_escalates_and_never_stops(tmp_path: Path) -> None:
    events: list[str] = []

    class UnknownChild(FakeChild):
        def group_alive(self) -> bool:
            raise ProcessGroupProbeError(
                self.process_group_id,
                PermissionError(1, "operation not permitted"),
            )

    factory = FakeFactory(events, lambda pid, seen: UnknownChild(pid, seen))
    supervisor = fake_supervisor(tmp_path, factory)

    supervisor.start()
    with pytest.raises(CleanupError, match="could not determine"):
        supervisor.stop()

    assert supervisor.state is RuntimeState.FAILED
    assert all(failure.group_survived for failure in supervisor.cleanup_failures)
    assert "terminate:2" in events and "kill:2" in events
    assert "terminate:1" in events and "kill:1" in events


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


def test_disabled_signal_policy_is_a_true_noop_on_main_thread(tmp_path: Path) -> None:
    before = (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM))
    factory = FakeFactory([])
    supervisor = fake_supervisor(
        tmp_path,
        factory,
        signal_policy=SignalPolicy.DISABLED_FOR_TEST,
    )
    supervisor.request_stop("test", 0)
    assert supervisor.run() == 0
    assert (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)) == before


def test_spawn_oserror_is_not_translated_by_signal_context(tmp_path: Path) -> None:
    class FailsToSpawn:
        def spawn(self, spec: ProcessSpec) -> ChildProcess:
            raise OSError(f"cannot spawn {spec.name}")

    supervisor = RuntimeSupervisor(
        runtime_config(tmp_path),
        process_factory=FailsToSpawn(),
        preflight_checker=lambda _config: None,
    )
    with pytest.raises(OSError, match="cannot spawn app"):
        supervisor.run()
    assert supervisor.state is RuntimeState.FAILED


def test_keyboard_interrupt_during_direct_start_cleans_runtime_lease(tmp_path: Path) -> None:
    class InterruptFactory:
        spec: ProcessSpec | None = None

        def spawn(self, spec: ProcessSpec) -> ChildProcess:
            self.spec = spec
            raise KeyboardInterrupt

    factory = InterruptFactory()
    supervisor = RuntimeSupervisor(
        runtime_config(tmp_path),
        process_factory=factory,
        preflight_checker=lambda _config: None,
        signal_policy=SignalPolicy.DISABLED_FOR_TEST,
    )
    with pytest.raises(KeyboardInterrupt):
        supervisor.start()
    assert factory.spec is not None
    assert not Path(factory.spec.env[CAPTURE_SOCKET_ENV]).parent.exists()


def test_body_exception_is_not_translated_by_signal_context(tmp_path: Path) -> None:
    supervisor = fake_supervisor(tmp_path, FakeFactory([]))
    with pytest.raises(RuntimeError, match="body failure"):
        with supervisor._signal_handlers():  # type: ignore[attr-defined]
            raise RuntimeError("body failure")


def test_signal_handlers_are_restored_after_shutdown_exception(tmp_path: Path) -> None:
    before = (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM))
    supervisor = fake_supervisor(tmp_path, FakeFactory([]))
    with pytest.raises(RuntimeError, match="shutdown body failure"):
        with supervisor._signal_handlers():  # type: ignore[attr-defined]
            raise RuntimeError("shutdown body failure")
    assert (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)) == before


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


@pytest.mark.skipif(os.name != "posix", reason="process-group semantics are POSIX-specific")
def test_real_process_shutdown_reaps_leaders_without_false_kill() -> None:
    events: list[str] = []
    specs: list[ProcessSpec] = []

    class RecordingChild:
        def __init__(self, name: str, child: PopenChild) -> None:
            self.name = name
            self.child = child

        @property
        def pid(self) -> int:
            return self.child.pid

        @property
        def process_group_id(self) -> int | None:
            return self.child.process_group_id

        def poll(self) -> int | None:
            return self.child.poll()

        def wait(self, timeout: float | None = None) -> int:
            return self.child.wait(timeout)

        def group_alive(self) -> bool:
            return self.child.group_alive()

        def terminate(self) -> None:
            events.append(f"terminate:{self.name}")
            self.child.terminate()

        def kill(self) -> None:
            events.append(f"kill:{self.name}")
            self.child.kill()

    class RealFactory:
        def spawn(self, spec: ProcessSpec) -> ChildProcess:
            specs.append(spec)
            code = (
                "import signal, time\n"
                "signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit(0)))\n"
                "while True: time.sleep(1)\n"
            )
            process_spec = ProcessSpec(spec.name, (sys.executable, "-c", code), spec.env)
            child = SubprocessFactory().spawn(process_spec)
            assert isinstance(child, PopenChild)
            return RecordingChild(spec.name, child)

    config = RuntimeConfig(
        readiness_timeout_seconds=1.0,
        graceful_shutdown_seconds=0.1,
        kill_wait_seconds=0.1,
        poll_interval_seconds=0.01,
    )
    supervisor = RuntimeSupervisor(
        config,
        process_factory=RealFactory(),
        readiness_probe=ImmediateReadinessProbe(),
        preflight_checker=lambda _config: None,
        signal_policy=SignalPolicy.DISABLED_FOR_TEST,
    )
    started = time.monotonic()
    supervisor.start()
    supervisor.stop()
    elapsed = time.monotonic() - started

    assert elapsed < 2.0
    assert events == ["terminate:proxy", "terminate:app"]
    assert supervisor.state is RuntimeState.STOPPED
    assert specs[0].env[CAPTURE_SOCKET_ENV] == specs[1].env[CAPTURE_SOCKET_ENV]


@pytest.mark.skipif(os.name != "posix", reason="process-group semantics are POSIX-specific")
def test_real_process_exit_between_group_check_and_signal_is_benign() -> None:
    class RaceChild:
        def __init__(self, child: PopenChild) -> None:
            self.child = child
            self.raced = False

        @property
        def pid(self) -> int:
            return self.child.pid

        @property
        def process_group_id(self) -> int | None:
            return self.child.process_group_id

        def poll(self) -> int | None:
            return self.child.poll()

        def wait(self, timeout: float | None = None) -> int:
            return self.child.wait(timeout)

        def group_alive(self) -> bool:
            alive = self.child.group_alive()
            if alive and not self.raced:
                self.raced = True
                os.kill(self.child.pid, signal.SIGTERM)
                self.child.wait(timeout=1.0)
                return True
            return alive

        def terminate(self) -> None:
            self.child.terminate()

        def kill(self) -> None:
            self.child.kill()

    class RaceFactory:
        def spawn(self, spec: ProcessSpec) -> ChildProcess:
            code = "import time; time.sleep(60)"
            child = SubprocessFactory().spawn(
                ProcessSpec(spec.name, (sys.executable, "-c", code), spec.env)
            )
            assert isinstance(child, PopenChild)
            return RaceChild(child)

    supervisor = RuntimeSupervisor(
        RuntimeConfig(
            readiness_timeout_seconds=1.0,
            graceful_shutdown_seconds=0.1,
            kill_wait_seconds=0.1,
            poll_interval_seconds=0.01,
        ),
        process_factory=RaceFactory(),
        readiness_probe=ImmediateReadinessProbe(),
        preflight_checker=lambda _config: None,
        signal_policy=SignalPolicy.DISABLED_FOR_TEST,
    )
    try:
        supervisor.start()
        supervisor.stop()
    finally:
        if supervisor.state is not RuntimeState.STOPPED:
            try:
                supervisor.stop()
            except CleanupError:
                pass

    assert supervisor.state is RuntimeState.STOPPED
    assert supervisor.cleanup_failures == ()


# -- B3 readiness probing and live CLI wiring -------------------------------


def _free_port() -> int:
    with socket.socket() as probe_socket:
        probe_socket.bind(("127.0.0.1", 0))
        return int(probe_socket.getsockname()[1])


def _health_server() -> tuple[ThreadingHTTPServer, int]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - http.server contract
            if self.path == "/api/v1/health":
                body = b'{"status":"ok"}'
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *_args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, int(server.server_address[1])


def _probe_config(app_port: int, proxy_port: int) -> RuntimeConfig:
    return RuntimeConfig(
        app_port=app_port,
        proxy_port=proxy_port,
        readiness_timeout_seconds=5.0,
        poll_interval_seconds=0.01,
    )


def test_http_readiness_probe_accepts_healthy_app_and_listening_proxy() -> None:
    server, app_port = _health_server()
    proxy_listener = socket.socket()
    proxy_listener.bind(("127.0.0.1", 0))
    proxy_listener.listen(1)
    proxy_port = int(proxy_listener.getsockname()[1])
    try:
        probe = HttpHealthReadinessProbe(_probe_config(app_port, proxy_port))
        probe.wait_until_ready("app", FakeChild(1, []), 5.0)
        probe.wait_until_ready("proxy", FakeChild(2, []), 5.0)
    finally:
        server.shutdown()
        server.server_close()
        proxy_listener.close()


def test_http_readiness_probe_requires_a_health_success_not_just_a_listener() -> None:
    class Refuser(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - http.server contract
            self.send_response(503)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *_args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Refuser)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    app_port = int(server.server_address[1])
    fake_clock = FakeClock()
    try:
        probe = HttpHealthReadinessProbe(
            _probe_config(app_port, _free_port()),
            clock=fake_clock.clock,
            sleeper=fake_clock.sleep,
        )
        with pytest.raises(ReadinessTimeoutError):
            probe.wait_until_ready("app", FakeChild(1, []), 0.05)
    finally:
        server.shutdown()
        server.server_close()
    assert fake_clock.sleeps


def test_http_readiness_probe_times_out_when_nothing_listens() -> None:
    fake_clock = FakeClock()
    probe = HttpHealthReadinessProbe(
        _probe_config(_free_port(), _free_port()),
        clock=fake_clock.clock,
        sleeper=fake_clock.sleep,
    )
    with pytest.raises(ReadinessTimeoutError) as excinfo:
        probe.wait_until_ready("app", FakeChild(1, []), 0.05)
    assert excinfo.value.component == "app"
    with pytest.raises(ReadinessTimeoutError):
        probe.wait_until_ready("proxy", FakeChild(2, []), 0.05)


def test_http_readiness_probe_reports_an_exited_child_immediately() -> None:
    probe = HttpHealthReadinessProbe(_probe_config(_free_port(), _free_port()))
    child = FakeChild(1, [])
    child.returncode = 3
    with pytest.raises(ChildExitedError) as excinfo:
        probe.wait_until_ready("app", child, 5.0)
    assert excinfo.value.returncode == 3


def test_http_readiness_probe_rejects_unknown_components() -> None:
    probe = HttpHealthReadinessProbe(_probe_config(_free_port(), _free_port()))
    with pytest.raises(RuntimeSupervisorError):
        probe.wait_until_ready("browser", FakeChild(1, []), 0.1)


def test_cli_live_run_uses_the_injected_supervisor(
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen: list[RuntimeConfig] = []

    class FakeSupervisor:
        def __init__(self, config: RuntimeConfig) -> None:
            seen.append(config)

        def run(self) -> int:
            return 7

    assert main(["run"], supervisor_factory=FakeSupervisor) == 7
    assert len(seen) == 1
    assert seen[0].app_port == 8000
    assert "serving: http://127.0.0.1:8000/" in capsys.readouterr().out


def test_cli_live_run_reports_supervisor_failures(
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FailingSupervisor:
        def __init__(self, config: RuntimeConfig) -> None:
            self.config = config

        def run(self) -> int:
            raise RuntimeSupervisorError("proxy was not ready")

    assert main(["run"], supervisor_factory=FailingSupervisor) == 1
    assert "proxy was not ready" in capsys.readouterr().err


def test_default_supervisor_factory_wires_the_health_probe() -> None:
    supervisor = default_supervisor_factory(RuntimeConfig())
    assert isinstance(supervisor, RuntimeSupervisor)
    assert isinstance(supervisor._readiness_probe, HttpHealthReadinessProbe)


def test_app_server_module_serves_health_and_stops_on_sigterm() -> None:
    app_port = _free_port()
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mitm_inspector.api.server",
            "--host",
            "127.0.0.1",
            "--port",
            str(app_port),
        ],
        stdin=subprocess.DEVNULL,
    )
    try:
        probe = HttpHealthReadinessProbe(_probe_config(app_port, _free_port()))
        probe.wait_until_ready("app", PopenChild(process), 10.0)
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=10.0) == 143
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5.0)


def test_capture_ipc_bounds_prefix_to_the_ingest_line_capacity() -> None:
    CaptureIPCConfig(
        max_body_prefix_bytes=MAX_INGEST_BODY_PREFIX_BYTES,
        max_in_memory_bytes=MAX_INGEST_BODY_PREFIX_BYTES * 4,
    )
    with pytest.raises(RuntimeConfigError):
        CaptureIPCConfig(
            max_body_prefix_bytes=MAX_INGEST_BODY_PREFIX_BYTES + 1,
            max_in_memory_bytes=MAX_INGEST_BODY_PREFIX_BYTES * 4,
        )
    with pytest.raises(RuntimeConfigError):
        RuntimeConfig(
            max_body_prefix_bytes=MAX_INGEST_BODY_PREFIX_BYTES + 1,
            max_body_bytes=MAX_INGEST_BODY_PREFIX_BYTES * 4,
        )

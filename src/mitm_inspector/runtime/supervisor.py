"""Deterministic, injectable supervision of the app and proxy children."""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
import webbrowser
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path
from types import FrameType
from typing import Any, Protocol

from mitm_inspector.runtime.commands import build_app_argv, build_proxy_argv
from mitm_inspector.runtime.config import RuntimeConfig


class RuntimeState(StrEnum):
    NEW = "new"
    STARTING_APP = "starting_app"
    APP_READY = "app_ready"
    STARTING_PROXY = "starting_proxy"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


class RuntimeSupervisorError(RuntimeError):
    """Base error for runtime lifecycle failures."""


class ChildExitedError(RuntimeSupervisorError):
    """A runtime child exited before an explicit shutdown was requested."""

    def __init__(self, component: str, returncode: int) -> None:
        self.component = component
        self.returncode = returncode
        super().__init__(f"{component} exited with status {returncode}")


class ChildProcess(Protocol):
    @property
    def pid(self) -> int: ...

    def poll(self) -> int | None: ...

    def wait(self, timeout: float | None = None) -> int: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...


class ProcessFactory(Protocol):
    def spawn(self, argv: Sequence[str]) -> ChildProcess: ...


class ReadinessProbe(Protocol):
    def wait_until_ready(
        self,
        component: str,
        child: ChildProcess,
        timeout_seconds: float,
    ) -> None: ...


class BrowserOpener(Protocol):
    def __call__(self, url: str) -> bool: ...


class PopenChild:
    """Child process wrapper that owns a separate process group where supported."""

    def __init__(self, process: subprocess.Popen[bytes]) -> None:
        self._process = process

    @property
    def pid(self) -> int:
        return self._process.pid

    def poll(self) -> int | None:
        return self._process.poll()

    def wait(self, timeout: float | None = None) -> int:
        return self._process.wait(timeout=timeout)

    def _signal_group(self, signum: int, fallback: Callable[[], None]) -> None:
        if self.poll() is not None:
            return
        if os.name == "posix":
            try:
                os.killpg(os.getpgid(self.pid), signum)
                return
            except ProcessLookupError:
                return
        fallback()

    def terminate(self) -> None:
        self._signal_group(signal.SIGTERM, self._process.terminate)

    def kill(self) -> None:
        self._signal_group(signal.SIGKILL, self._process.kill)


class SubprocessFactory:
    """Production process factory; every command is passed as an argv sequence."""

    def spawn(self, argv: Sequence[str]) -> ChildProcess:
        arguments = list(argv)
        # CREATE_NEW_PROCESS_GROUP is 0x00000200 on Windows.  Keeping the
        # numeric constant avoids importing a platform-only subprocess symbol
        # on POSIX while retaining group-aware signal handling on Windows.
        windows_process_group = 0x00000200 if os.name == "nt" else 0
        process = subprocess.Popen(
            arguments,
            shell=False,
            stdin=subprocess.DEVNULL,
            stdout=None,
            stderr=None,
            close_fds=os.name != "nt",
            start_new_session=os.name == "posix",
            creationflags=windows_process_group,
        )
        return PopenChild(process)


class ImmediateReadinessProbe:
    """Default seam for B3; it verifies spawn success without opening a socket."""

    def wait_until_ready(
        self,
        component: str,
        child: ChildProcess,
        timeout_seconds: float,
    ) -> None:
        returncode = child.poll()
        if returncode is not None:
            raise ChildExitedError(component, returncode)


class RuntimeSupervisor:
    """Start app first, then proxy; stop proxy first, then app.

    The readiness probe is deliberately injected.  The default probe is inert
    until B3 owns a real health endpoint, while tests can model readiness and
    failure without starting listeners or child processes.
    """

    def __init__(
        self,
        config: RuntimeConfig,
        *,
        addon_path: Path | str = Path("src/mitm_inspector/capture/addon.py"),
        process_factory: ProcessFactory | None = None,
        readiness_probe: ReadinessProbe | None = None,
        browser_opener: BrowserOpener | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.addon_path = addon_path
        self._process_factory = process_factory or SubprocessFactory()
        self._readiness_probe = readiness_probe or ImmediateReadinessProbe()
        self._browser_opener = browser_opener or webbrowser.open
        self._clock = clock
        self._sleep = sleeper
        self._children: dict[str, ChildProcess] = {}
        self._stop_requested = False
        self._stop_reason: str | None = None
        self._last_error: BaseException | None = None
        self._state = RuntimeState.NEW
        self._state_history: list[RuntimeState] = [self._state]
        self._lock = threading.RLock()

    @property
    def state(self) -> RuntimeState:
        return self._state

    @property
    def state_history(self) -> tuple[RuntimeState, ...]:
        return tuple(self._state_history)

    @property
    def last_error(self) -> BaseException | None:
        return self._last_error

    @property
    def children(self) -> tuple[str, ...]:
        return tuple(self._children)

    @property
    def browser_url(self) -> str:
        host = f"[{self.config.app_host}]" if ":" in self.config.app_host else self.config.app_host
        return f"http://{host}:{self.config.app_port}/"

    def request_stop(self, reason: str = "requested") -> None:
        with self._lock:
            self._stop_requested = True
            self._stop_reason = reason

    def start(self) -> None:
        with self._lock:
            if self._state is not RuntimeState.NEW:
                raise RuntimeSupervisorError(f"cannot start from state {self._state.value}")
            try:
                self._set_state(RuntimeState.STARTING_APP)
                self._children["app"] = self._process_factory.spawn(build_app_argv(self.config))
                self._readiness_probe.wait_until_ready(
                    "app", self._children["app"], self.config.readiness_timeout_seconds
                )
                self._set_state(RuntimeState.APP_READY)
                self._set_state(RuntimeState.STARTING_PROXY)
                self._children["proxy"] = self._process_factory.spawn(
                    build_proxy_argv(self.config, self.addon_path)
                )
                self._readiness_probe.wait_until_ready(
                    "proxy", self._children["proxy"], self.config.readiness_timeout_seconds
                )
                self._set_state(RuntimeState.RUNNING)
                if self.config.open_browser:
                    self._browser_opener(self.browser_url)
            except BaseException as exc:
                self._last_error = exc
                self._set_state(RuntimeState.FAILED)
                self._shutdown_children()
                raise

    def run(self) -> int:
        """Run until cancellation; unexpected child exits are raised."""

        try:
            self.start()
            with self._signal_handlers():
                while self.state is RuntimeState.RUNNING and not self._stop_requested:
                    for component in ("proxy", "app"):
                        child = self._children.get(component)
                        if child is None:
                            continue
                        returncode = child.poll()
                        if returncode is not None:
                            raise ChildExitedError(component, returncode)
                    self._sleep(self.config.poll_interval_seconds)
            return 0
        except KeyboardInterrupt:
            self.request_stop("keyboard interrupt")
            return 130
        except ChildExitedError as exc:
            self._last_error = exc
            self._set_state(RuntimeState.FAILED)
            raise
        finally:
            self.stop()

    def stop(self) -> None:
        """Idempotently perform bounded graceful shutdown followed by kill."""

        with self._lock:
            if self._state is RuntimeState.STOPPED:
                return
            self._set_state(RuntimeState.STOPPING)
            self._shutdown_children()
            if self._state is RuntimeState.STOPPING:
                self._set_state(RuntimeState.STOPPED)

    close = stop

    def _set_state(self, state: RuntimeState) -> None:
        self._state = state
        if not self._state_history or self._state_history[-1] is not state:
            self._state_history.append(state)

    def _shutdown_children(self) -> None:
        live = [
            (component, child)
            for component, child in (
                ("proxy", self._children.get("proxy")),
                ("app", self._children.get("app")),
            )
            if child is not None and child.poll() is None
        ]
        for _, child in live:
            child.terminate()
        self._wait_for_children(live, self.config.graceful_shutdown_seconds)
        remaining = [(component, child) for component, child in live if child.poll() is None]
        for _, child in remaining:
            child.kill()
        self._wait_for_children(remaining, self.config.kill_wait_seconds, reap=True)

    def _wait_for_children(
        self,
        children: Sequence[tuple[str, ChildProcess]],
        timeout_seconds: float,
        *,
        reap: bool = False,
    ) -> None:
        if not children:
            return
        deadline = self._clock() + timeout_seconds
        while self._clock() < deadline:
            if all(child.poll() is not None for _, child in children):
                return
            try:
                remaining = max(0.0, deadline - self._clock())
                self._sleep(min(self.config.poll_interval_seconds, remaining))
            except KeyboardInterrupt:
                break
        if reap:
            for _, child in children:
                if child.poll() is None:
                    try:
                        child.wait(timeout=0)
                    except (subprocess.TimeoutExpired, TimeoutError):
                        pass

    @contextmanager
    def _signal_handlers(self) -> Iterator[None]:
        if threading.current_thread() is not threading.main_thread():
            yield
            return
        previous: dict[signal.Signals, Any] = {}

        def handle(signum: int, _frame: FrameType | None) -> None:
            self.request_stop(signal.Signals(signum).name)

        try:
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous[signum] = signal.getsignal(signum)
                signal.signal(signum, handle)
            yield
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)

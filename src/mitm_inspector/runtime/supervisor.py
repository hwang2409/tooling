"""Deterministic, injectable supervision of the app and proxy children."""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
import warnings
import webbrowser
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from enum import StrEnum
from types import FrameType
from typing import Any, Protocol

from mitm_inspector.runtime.commands import ProcessSpec, build_app_spec, build_proxy_spec
from mitm_inspector.runtime.config import (
    RuntimeConfig,
    cleanup_private_runtime_dir,
    validate_preflight,
)


class RuntimeState(StrEnum):
    NEW = "new"
    STARTING_APP = "starting_app"
    APP_READY = "app_ready"
    STARTING_PROXY = "starting_proxy"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


class SignalPolicy(StrEnum):
    REQUIRE_MAIN_THREAD = "require_main_thread"
    DISABLED_FOR_TEST = "disabled_for_test"


class RuntimeSupervisorError(RuntimeError):
    """Base error for runtime lifecycle failures."""


class SignalHandlingError(RuntimeSupervisorError):
    """Raised when signals cannot be installed under the selected policy."""


class ChildExitedError(RuntimeSupervisorError):
    """A runtime child exited before an explicit shutdown was requested."""

    def __init__(self, component: str, returncode: int) -> None:
        self.component = component
        self.returncode = returncode
        super().__init__(f"{component} exited with status {returncode}")


class ChildCleanupFailure(RuntimeSupervisorError):
    def __init__(self, component: str, errors: tuple[str, ...], group_survived: bool) -> None:
        self.component = component
        self.errors = errors
        self.group_survived = group_survived
        super().__init__(f"{component} cleanup failed: {', '.join(errors)}")


class CleanupError(RuntimeSupervisorError):
    """One or more children could not be fully stopped and reaped."""

    def __init__(self, failures: tuple[ChildCleanupFailure, ...]) -> None:
        self.failures = failures
        details = "; ".join(
            f"{failure.component}: {', '.join(failure.errors)}"
            for failure in failures
        )
        super().__init__(f"runtime cleanup failed: {details}")


class ChildProcess(Protocol):
    @property
    def pid(self) -> int: ...

    @property
    def process_group_id(self) -> int | None: ...

    def poll(self) -> int | None: ...

    def wait(self, timeout: float | None = None) -> int: ...

    def group_alive(self) -> bool: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...


class ProcessFactory(Protocol):
    def spawn(self, spec: ProcessSpec) -> ChildProcess: ...


class ReadinessProbe(Protocol):
    def wait_until_ready(
        self,
        component: str,
        child: ChildProcess,
        timeout_seconds: float,
    ) -> None: ...


class WarningSink(Protocol):
    def __call__(self, message: str) -> None: ...


class BrowserOpener(Protocol):
    def __call__(self, url: str) -> bool: ...


class PopenChild:
    """Child wrapper retaining process-group identity after leader exit."""

    def __init__(self, process: subprocess.Popen[bytes]) -> None:
        self._process = process
        if os.name == "posix":
            try:
                self._process_group_id: int | None = os.getpgid(process.pid)
            except ProcessLookupError:
                self._process_group_id = process.pid
        else:
            self._process_group_id = None

    @property
    def pid(self) -> int:
        return self._process.pid

    @property
    def process_group_id(self) -> int | None:
        return self._process_group_id

    def poll(self) -> int | None:
        return self._process.poll()

    def wait(self, timeout: float | None = None) -> int:
        return self._process.wait(timeout=timeout)

    def group_alive(self) -> bool:
        if os.name != "posix" or self._process_group_id is None:
            return self.poll() is None
        # Popen.poll() also reaps the group leader.  On macOS, probing a
        # cached group after that can report EPERM for a stale leader.  Only a
        # successful probe is evidence of a live descendant in this case.
        if self.poll() is not None:
            try:
                os.killpg(self._process_group_id, 0)
            except (ProcessLookupError, PermissionError):
                return False
            return True
        try:
            os.killpg(self._process_group_id, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return False
        return True

    def _signal_group(self, signum: int, fallback: Callable[[], None]) -> None:
        if os.name == "posix" and self._process_group_id is not None:
            try:
                os.killpg(self._process_group_id, signum)
                return
            except ProcessLookupError:
                if self.poll() is not None:
                    return
        fallback()

    def terminate(self) -> None:
        self._signal_group(signal.SIGTERM, self._process.terminate)

    def kill(self) -> None:
        self._signal_group(signal.SIGKILL, self._process.kill)


class SubprocessFactory:
    """Production process factory; every command is passed as a ProcessSpec."""

    def spawn(self, spec: ProcessSpec) -> ChildProcess:
        environment = os.environ.copy()
        environment.update(spec.env)
        windows_process_group = 0x00000200 if os.name == "nt" else 0
        process = subprocess.Popen(
            list(spec.argv),
            shell=False,
            stdin=subprocess.DEVNULL,
            stdout=None,
            stderr=None,
            env=environment,
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
        del timeout_seconds
        returncode = child.poll()
        if returncode is not None:
            raise ChildExitedError(component, returncode)


class RuntimeSupervisor:
    """Start app first; stop proxy fully before stopping app.

    The readiness probe is injected.  The default probe is inert until B3 owns
    a real health endpoint, while tests can model readiness and failure without
    starting listeners or child processes.
    """

    def __init__(
        self,
        config: RuntimeConfig,
        *,
        process_factory: ProcessFactory | None = None,
        readiness_probe: ReadinessProbe | None = None,
        preflight_checker: Callable[[RuntimeConfig], None] = validate_preflight,
        browser_opener: BrowserOpener | None = None,
        warning_sink: WarningSink | None = None,
        signal_policy: SignalPolicy = SignalPolicy.REQUIRE_MAIN_THREAD,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self._process_factory = process_factory or SubprocessFactory()
        self._readiness_probe = readiness_probe or ImmediateReadinessProbe()
        self._preflight_checker = preflight_checker
        self._browser_opener = browser_opener or webbrowser.open
        self._warning_sink = warning_sink or self._default_warning_sink
        self._signal_policy = signal_policy
        self._clock = clock
        self._sleep = sleeper
        self._children: dict[str, ChildProcess] = {}
        self._stop_requested = False
        self._stop_reason: str | None = None
        self._stop_status_code = 0
        self._last_error: BaseException | None = None
        self._cleanup_failures: tuple[ChildCleanupFailure, ...] = ()
        self._active_config: RuntimeConfig | None = None
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
    def cleanup_failures(self) -> tuple[ChildCleanupFailure, ...]:
        return self._cleanup_failures

    @property
    def children(self) -> tuple[str, ...]:
        return tuple(self._children)

    @property
    def browser_url(self) -> str:
        host = f"[{self.config.app_host}]" if ":" in self.config.app_host else self.config.app_host
        return f"http://{host}:{self.config.app_port}/"

    def request_stop(self, reason: str = "requested", status_code: int = 0) -> None:
        with self._lock:
            self._stop_requested = True
            self._stop_reason = reason
            self._stop_status_code = max(self._stop_status_code, status_code)

    def handle_signal(self, signum: int) -> None:
        if signum == signal.SIGINT:
            self.request_stop("SIGINT", 130)
        elif signum == signal.SIGTERM:
            self.request_stop("SIGTERM", 143)
        else:
            raise ValueError(f"unsupported runtime signal: {signum}")

    def start(self) -> None:
        with self._lock:
            if self._state is not RuntimeState.NEW:
                raise RuntimeSupervisorError(f"cannot start from state {self._state.value}")
            try:
                self._preflight_checker(self.config)
                if self._stop_requested:
                    return
                allocated_capture = self.config.capture.allocate()
                try:
                    self._active_config = replace(
                        self.config,
                        capture_ipc=allocated_capture,
                    )
                except BaseException:
                    cleanup_private_runtime_dir(allocated_capture)
                    raise
                active_config = self._active_config
                if active_config is None:  # pragma: no cover - defensive invariant.
                    raise RuntimeSupervisorError("runtime configuration was not allocated")
                self._set_state(RuntimeState.STARTING_APP)
                self._children["app"] = self._process_factory.spawn(build_app_spec(active_config))
                self._readiness_probe.wait_until_ready(
                    "app", self._children["app"], active_config.readiness_timeout_seconds
                )
                if self._stop_requested:
                    return
                self._set_state(RuntimeState.APP_READY)
                self._set_state(RuntimeState.STARTING_PROXY)
                self._children["proxy"] = self._process_factory.spawn(
                    build_proxy_spec(active_config)
                )
                self._readiness_probe.wait_until_ready(
                    "proxy", self._children["proxy"], active_config.readiness_timeout_seconds
                )
                if self._stop_requested:
                    return
                self._set_state(RuntimeState.RUNNING)
                self._open_browser_if_requested()
            except KeyboardInterrupt as exc:
                self.request_stop("keyboard interrupt", 130)
                failures = self._cleanup_runtime()
                if failures:
                    self._cleanup_failures = failures
                    raise CleanupError(failures) from exc
                raise
            except Exception as exc:
                self._last_error = exc
                self._set_state(RuntimeState.FAILED)
                failures = self._cleanup_runtime()
                if failures:
                    self._cleanup_failures = failures
                    raise CleanupError(failures) from exc
                raise

    def run(self) -> int:
        """Run until cancellation; child and cleanup failures are raised."""

        try:
            with self._signal_handlers():
                try:
                    self.start()
                    while self.state is RuntimeState.RUNNING and not self._stop_requested:
                        for component in ("proxy", "app"):
                            child = self._children.get(component)
                            if child is None:
                                continue
                            returncode = child.poll()
                            if returncode is not None:
                                raise ChildExitedError(component, returncode)
                        self._sleep(self.config.poll_interval_seconds)
                except KeyboardInterrupt:
                    self.request_stop("keyboard interrupt", 130)
                finally:
                    self.stop()
            return self._stop_status_code
        except RuntimeSupervisorError as exc:
            self._last_error = exc
            if self._state is RuntimeState.NEW:
                self._set_state(RuntimeState.FAILED)
            raise

    def stop(self) -> None:
        """Idempotently stop proxy fully, then app, aggregating all failures."""

        with self._lock:
            if self._state is RuntimeState.STOPPED:
                return
            if self._state is RuntimeState.NEW:
                if self._stop_requested:
                    self._set_state(RuntimeState.STOPPING)
                    self._set_state(RuntimeState.STOPPED)
                return
            if self._state is not RuntimeState.FAILED:
                self._set_state(RuntimeState.STOPPING)
            failures = self._cleanup_runtime()
            self._cleanup_failures = failures
            if failures:
                error = CleanupError(failures)
                self._last_error = error
                self._set_state(RuntimeState.FAILED)
                raise error
            if self._state is RuntimeState.STOPPING:
                self._set_state(RuntimeState.STOPPED)

    close = stop

    def _set_state(self, state: RuntimeState) -> None:
        self._state = state
        if not self._state_history or self._state_history[-1] is not state:
            self._state_history.append(state)

    def _open_browser_if_requested(self) -> None:
        if not self.config.open_browser:
            return
        try:
            opened = self._browser_opener(self.browser_url)
        except Exception as exc:
            self._emit_warning(f"browser open failed: {exc}")
        else:
            if not opened:
                self._emit_warning("browser open was not accepted")

    def _emit_warning(self, message: str) -> None:
        try:
            self._warning_sink(message)
        except Exception as exc:
            warnings.warn(f"runtime warning sink failed: {exc}", RuntimeWarning, stacklevel=3)

    @staticmethod
    def _default_warning_sink(message: str) -> None:
        warnings.warn(message, RuntimeWarning, stacklevel=3)

    def _cleanup_runtime(self) -> tuple[ChildCleanupFailure, ...]:
        failures = list(self._shutdown_children())
        active_config = self._active_config
        if active_config is not None and active_config.capture.runtime_dir is not None:
            try:
                resource_errors = cleanup_private_runtime_dir(active_config.capture)
            except BaseException as exc:
                resource_errors = (f"runtime directory cleanup raised: {exc}",)
            if resource_errors:
                failures.append(ChildCleanupFailure("ipc", resource_errors, True))
            else:
                self._active_config = None
        else:
            self._active_config = None
        return tuple(failures)

    def _shutdown_children(self) -> tuple[ChildCleanupFailure, ...]:
        failures: list[ChildCleanupFailure] = []
        for component in ("proxy", "app"):
            child = self._children.get(component)
            if child is None:
                continue
            try:
                self._shutdown_child(component, child)
            except ChildCleanupFailure as exc:
                failures.append(exc)
            except BaseException as exc:
                failures.append(
                    ChildCleanupFailure(component, (f"unexpected cleanup exception: {exc}",), True)
                )
        return tuple(failures)

    def _shutdown_child(self, component: str, child: ChildProcess) -> None:
        errors: list[str] = []
        group_alive = self._safe_group_alive(child, errors)
        if group_alive:
            try:
                child.terminate()
            except BaseException as exc:
                errors.append(f"terminate failed: {exc}")
            if not self._wait_for_group_exit(child, self.config.graceful_shutdown_seconds):
                try:
                    child.kill()
                except BaseException as exc:
                    errors.append(f"kill failed: {exc}")
                if not self._wait_for_group_exit(child, self.config.kill_wait_seconds):
                    errors.append("process group survived kill deadline")
        try:
            child.wait(timeout=self.config.kill_wait_seconds)
        except (subprocess.TimeoutExpired, TimeoutError):
            if child.poll() is None:
                errors.append("child leader was not reaped")
        except BaseException as exc:
            errors.append(f"wait failed: {exc}")
        survived = self._safe_group_alive(child, errors)
        if survived:
            errors.append("process group is still alive")
        if errors:
            raise ChildCleanupFailure(component, tuple(dict.fromkeys(errors)), survived)

    def _safe_group_alive(self, child: ChildProcess, errors: list[str]) -> bool:
        try:
            return child.group_alive()
        except BaseException as exc:
            errors.append(f"group status failed: {exc}")
            return child.poll() is None

    def _wait_for_group_exit(self, child: ChildProcess, timeout_seconds: float) -> bool:
        deadline = self._clock() + timeout_seconds
        while self._clock() < deadline:
            try:
                if not child.group_alive():
                    return True
            except BaseException:
                return False
            try:
                remaining = max(0.0, deadline - self._clock())
                self._sleep(min(self.config.poll_interval_seconds, remaining))
            except KeyboardInterrupt:
                return False
        try:
            return not child.group_alive()
        except BaseException:
            return False

    @contextmanager
    def _signal_handlers(self) -> Iterator[None]:
        if self._signal_policy is SignalPolicy.DISABLED_FOR_TEST:
            # A true no-op, including on the main thread.  This policy is
            # intentionally explicit so tests never perturb process-global
            # handlers while exercising startup and cleanup.
            yield
            return
        if threading.current_thread() is not threading.main_thread():
            raise SignalHandlingError(
                "runtime signals require the main thread; use DISABLED_FOR_TEST only in tests"
            )
        previous: dict[signal.Signals, Any] = {}

        def handle(signum: int, _frame: FrameType | None) -> None:
            self.handle_signal(signum)

        try:
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous[signum] = signal.getsignal(signum)
                signal.signal(signum, handle)
        except (OSError, ValueError) as exc:
            install_restore_errors: list[str] = []
            for signum, handler in previous.items():
                try:
                    signal.signal(signum, handler)
                except (OSError, ValueError) as restore_exc:
                    install_restore_errors.append(str(restore_exc))
            detail = str(exc)
            if install_restore_errors:
                detail += "; partial restore failed: " + "; ".join(install_restore_errors)
            raise SignalHandlingError(
                f"could not install runtime signal handlers: {detail}"
            ) from exc

        body_error: BaseException | None = None
        try:
            yield
        except BaseException as exc:
            body_error = exc
            raise
        finally:
            restore_errors: list[str] = []
            for signum, handler in previous.items():
                try:
                    signal.signal(signum, handler)
                except (OSError, ValueError) as exc:
                    restore_errors.append(str(exc))
            if restore_errors and body_error is None:
                raise SignalHandlingError(
                    "could not restore runtime signal handlers: " + "; ".join(restore_errors)
                )

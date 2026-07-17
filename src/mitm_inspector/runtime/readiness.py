"""Component-aware readiness probing for the supervised runtime children.

The app child is ready when its versioned health endpoint answers 200; the
proxy child is ready when its loopback listener accepts a TCP connection.
Both checks are bounded, poll the child for early exit, and never shell out.
"""

from __future__ import annotations

import http.client
import socket
import time
from collections.abc import Callable

from mitm_inspector.runtime.config import RuntimeConfig
from mitm_inspector.runtime.supervisor import (
    ChildExitedError,
    ChildProcess,
    RuntimeSupervisorError,
)

HEALTH_PATH = "/api/v1/health"
_PROBE_ATTEMPT_TIMEOUT_SECONDS = 1.0


class ReadinessTimeoutError(RuntimeSupervisorError):
    """A child kept running but never became ready within the deadline."""

    def __init__(self, component: str, timeout_seconds: float) -> None:
        self.component = component
        self.timeout_seconds = timeout_seconds
        super().__init__(
            f"{component} was not ready within {timeout_seconds:.1f} seconds"
        )


def _check_app_health(host: str, port: int) -> bool:
    connection = http.client.HTTPConnection(
        host, port, timeout=_PROBE_ATTEMPT_TIMEOUT_SECONDS
    )
    try:
        connection.request("GET", HEALTH_PATH)
        response = connection.getresponse()
        response.read()
        return response.status == 200
    except (OSError, http.client.HTTPException):
        return False
    finally:
        connection.close()


def _check_tcp_listener(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=_PROBE_ATTEMPT_TIMEOUT_SECONDS):
            return True
    except OSError:
        return False


class HttpHealthReadinessProbe:
    """Readiness seam implementation used by the live CLI run."""

    def __init__(
        self,
        config: RuntimeConfig,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self._config = config
        self._clock = clock
        self._sleep = sleeper

    def _check(self, component: str) -> bool:
        if component == "app":
            return _check_app_health(self._config.app_host, self._config.app_port)
        if component == "proxy":
            return _check_tcp_listener(self._config.proxy_host, self._config.proxy_port)
        raise RuntimeSupervisorError(f"unknown readiness component: {component}")

    def wait_until_ready(
        self,
        component: str,
        child: ChildProcess,
        timeout_seconds: float,
    ) -> None:
        deadline = self._clock() + timeout_seconds
        while True:
            returncode = child.poll()
            if returncode is not None:
                raise ChildExitedError(component, returncode)
            if self._check(component):
                return
            if self._clock() >= deadline:
                raise ReadinessTimeoutError(component, timeout_seconds)
            self._sleep(self._config.poll_interval_seconds)

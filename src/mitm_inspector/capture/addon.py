"""Small addon seam for documented mitmproxy lifecycle hooks.

The addon is intentionally inert in S0. Later capture work can attach bounded
emitters without making the web app depend on mitmproxy internals.
"""

from __future__ import annotations

from collections.abc import Callable

from mitmproxy import http

from mitm_inspector.protocol import ParsedMessage

MessageEmitter = Callable[[ParsedMessage], None]


class CaptureAddon:
    """Documented-hook adapter with no forwarding or persistence behavior yet."""

    def __init__(self, emit: MessageEmitter | None = None) -> None:
        self.emit = emit

    def request(self, flow: http.HTTPFlow) -> None:
        """Receive a request lifecycle event through mitmproxy's public hook."""

    def response(self, flow: http.HTTPFlow) -> None:
        """Receive a response lifecycle event through mitmproxy's public hook."""

    def error(self, flow: http.HTTPFlow) -> None:
        """Receive an error lifecycle event through mitmproxy's public hook."""

"""Transport seam; sockets and endpoints are intentionally out of S0."""

from __future__ import annotations

from typing import cast

from mitm_inspector.protocol import ProtocolMessage


def encode_for_browser(message: ProtocolMessage) -> ProtocolMessage:
    """Return a protocol-owned message for a future API/WebSocket adapter."""

    return cast(ProtocolMessage, dict(message))

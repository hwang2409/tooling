"""Transport seam; sockets and endpoints are intentionally out of S0."""

from __future__ import annotations

from mitm_inspector.protocol import ParsedMessage, ParsedMessageResult, require_parsed_message


def encode_for_browser(message: ParsedMessage) -> ParsedMessageResult:
    """Return a protocol-owned message for a future API/WebSocket adapter."""

    return require_parsed_message(message)

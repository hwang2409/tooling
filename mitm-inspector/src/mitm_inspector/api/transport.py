"""Transport seam; sockets and endpoints are intentionally out of S0."""

from __future__ import annotations

from mitm_inspector.json_boundary import PlainJsonObject
from mitm_inspector.protocol import ParsedMessage, parsed_message_to_plain_json


def encode_for_browser(message: ParsedMessage) -> PlainJsonObject:
    """Return an independent plain-JSON protocol message for browser transport."""

    return parsed_message_to_plain_json(message)

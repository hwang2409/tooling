"""Shared wire limits for the capture ingest transport.

A ``flow.metadata`` line can carry two base64 body prefixes plus the
sanitized identity and header envelope, so the largest configurable body
prefix must keep ``2 * ceil(prefix / 3) * 4`` plus the envelope allowance
inside one bounded ingest line.  Runtime and API configuration both validate
against this bound so a legal capture configuration can never produce lines
the ingest listener would drop.
"""

MAX_INGEST_LINE_BYTES = 8 * 1024 * 1024
INGEST_ENVELOPE_ALLOWANCE_BYTES = 64 * 1024
MAX_INGEST_BODY_PREFIX_BYTES = (
    (MAX_INGEST_LINE_BYTES - INGEST_ENVELOPE_ALLOWANCE_BYTES) * 3
) // 8

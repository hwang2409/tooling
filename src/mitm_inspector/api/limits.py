"""Shared wire limits for the capture ingest transport.

A ``flow.metadata`` line can carry two base64 body prefixes plus the
sanitized identity and header envelope, so the largest configurable body
prefix must keep ``2 * ceil(prefix / 3) * 4`` plus the complete envelope
allowance inside one bounded ingest line. Runtime and API configuration both
validate against this bound so a legal capture configuration can never
produce lines the ingest listener would drop.
"""

from mitm_inspector.protocol import MAX_INGEST_LINE_BYTES, MAX_METADATA_HEADER_BYTES

__all__ = [
    "INGEST_ENVELOPE_ALLOWANCE_BYTES",
    "INGEST_FIXED_ENVELOPE_BYTES",
    "MAX_INGEST_BODY_PREFIX_BYTES",
    "MAX_INGEST_LINE_BYTES",
]

INGEST_FIXED_ENVELOPE_BYTES = 1024
# Request and response headers each have this bound; reserve both sections in
# the line budget even when a given flow only carries one of them.
INGEST_ENVELOPE_ALLOWANCE_BYTES = (2 * MAX_METADATA_HEADER_BYTES) + INGEST_FIXED_ENVELOPE_BYTES
MAX_INGEST_BODY_PREFIX_BYTES = (
    (MAX_INGEST_LINE_BYTES - INGEST_ENVELOPE_ALLOWANCE_BYTES) * 3
) // 8

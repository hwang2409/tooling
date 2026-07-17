"""Pure helpers for the irreversible pre-transport redaction boundary."""

from __future__ import annotations

REDACTED = "[REDACTED]"

# S0 deliberately exposes values only for a small, reviewed set of metadata
# headers. Unknown, custom, malformed, and otherwise unlisted names fail closed.
SAFE_HEADER_VALUE_NAMES = frozenset(
    {
        "accept",
        "accept-charset",
        "accept-encoding",
        "accept-language",
        "accept-ranges",
        "age",
        "cache-control",
        "connection",
        "content-encoding",
        "content-length",
        "content-range",
        "content-type",
        "date",
        "etag",
        "expires",
        "host",
        "last-modified",
        "range",
        "server",
        "traceparent",
        "transfer-encoding",
        "user-agent",
        "x-b3-spanid",
        "x-b3-traceid",
        "x-correlation-id",
        "x-request-id",
        "x-trace-id",
    }
)


def sanitize_header(name: str, value: str) -> tuple[str, str]:
    """Preserve the name and expose the value only for an explicitly safe header."""

    exposed_value = value if name.lower() in SAFE_HEADER_VALUE_NAMES else REDACTED
    return name, exposed_value


def sanitize_path(path: str) -> str:
    """Drop query material until a query-aware allow-list exists."""

    return path.split("?", 1)[0]

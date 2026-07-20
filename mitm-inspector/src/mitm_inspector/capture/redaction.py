"""Pure helpers for the irreversible pre-transport redaction boundary."""

from __future__ import annotations

from mitm_inspector.json_boundary import JsonBoundaryError, canonicalize_json

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

    canonical_name = _canonical_text(name, label="header name")
    canonical_value = _canonical_text(value, label="header value")
    exposed_value = (
        canonical_value if canonical_name.lower() in SAFE_HEADER_VALUE_NAMES else REDACTED
    )
    return canonical_name, exposed_value


def sanitize_path(path: str) -> str:
    """Drop query material until a query-aware allow-list exists."""

    return _canonical_text(path, label="path").split("?", 1)[0]


def _canonical_text(value: object, *, label: str) -> str:
    try:
        canonical = canonicalize_json(value, label=label)
    except JsonBoundaryError as error:
        raise ValueError(str(error)) from error
    if type(canonical) is not str:
        raise ValueError(f"{label} must be a string")
    return canonical

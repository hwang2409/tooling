"""Pure helpers for the irreversible pre-transport redaction boundary."""

from __future__ import annotations

SENSITIVE_HEADER_NAMES = frozenset(
    {"authorization", "proxy-authorization", "cookie", "set-cookie", "x-api-key"}
)
REDACTED = "[REDACTED]"


def sanitize_header_name(name: str) -> str:
    return name.lower()


def sanitize_header(name: str, value: str) -> tuple[str, str]:
    normalized_name = sanitize_header_name(name)
    return normalized_name, REDACTED if normalized_name in SENSITIVE_HEADER_NAMES else value


def sanitize_path(path: str) -> str:
    """Drop query material until a query-aware allow-list exists."""

    return path.split("?", 1)[0]

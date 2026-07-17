"""Pure helpers for the irreversible pre-transport redaction boundary."""

from __future__ import annotations

import re

SENSITIVE_HEADER_NAMES = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "cookie",
        "set-cookie",
        "api-key",
        "x-api-key",
        "x-auth-token",
        "x-amz-security-token",
        "access-token",
        "refresh-token",
        "id-token",
        "client-secret",
    }
)
_SEPARATOR_RUN = re.compile(r"[-_]+")
REDACTED = "[REDACTED]"


def sanitize_header_name(name: str) -> str:
    return _SEPARATOR_RUN.sub("-", name.strip().lower())


def is_sensitive_header_name(name: str) -> bool:
    normalized_name = sanitize_header_name(name)
    return (
        normalized_name in SENSITIVE_HEADER_NAMES
        or normalized_name.endswith("-api-key")
        or normalized_name.endswith("-auth-token")
        or normalized_name.endswith("-security-token")
        or normalized_name.endswith("-secret")
    )


def sanitize_header(name: str, value: str) -> tuple[str, str]:
    normalized_name = sanitize_header_name(name)
    return normalized_name, REDACTED if is_sensitive_header_name(normalized_name) else value


def sanitize_path(path: str) -> str:
    """Drop query material until a query-aware allow-list exists."""

    return path.split("?", 1)[0]

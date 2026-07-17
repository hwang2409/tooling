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
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
REDACTED = "[REDACTED]"
_EXPLICIT_SENSITIVE_NAMES = frozenset(
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
        "x-access-token",
        "refresh-token",
        "x-refresh-token",
        "id-token",
        "x-id-token",
        "bearer-token",
        "x-bearer-token",
        "token",
        "x-token",
        "signature",
        "x-signature",
        "credential",
        "credentials",
    }
)
_HARMLESS_SHAPED_NAMES = frozenset(
    {
        "content-key",
        "x-auth-mode",
        "x-cookie-state",
        "x-credential-type",
        "x-key-id",
        "x-secret-version",
        "x-signature-version",
        "x-token-count",
    }
)


def sanitize_header_name(name: str) -> str:
    camel_separated = _CAMEL_BOUNDARY.sub("-", name.strip())
    return _SEPARATOR_RUN.sub("-", camel_separated.lower())


def is_sensitive_header_name(name: str) -> bool:
    normalized_name = sanitize_header_name(name)
    if normalized_name in _HARMLESS_SHAPED_NAMES:
        return False
    compact_name = normalized_name.replace("-", "")
    if normalized_name in SENSITIVE_HEADER_NAMES or normalized_name in _EXPLICIT_SENSITIVE_NAMES:
        return True
    if compact_name in {name.replace("-", "") for name in _EXPLICIT_SENSITIVE_NAMES}:
        return True
    parts = normalized_name.split("-")
    credential_terms = {
        "auth",
        "authorization",
        "bearer",
        "cookie",
        "credential",
        "credentials",
        "secret",
        "signature",
    }
    if any(part in credential_terms for part in parts):
        return True
    return parts[-1] in {"token", "key"}


def sanitize_header(name: str, value: str) -> tuple[str, str]:
    normalized_name = sanitize_header_name(name)
    return normalized_name, REDACTED if is_sensitive_header_name(normalized_name) else value


def sanitize_path(path: str) -> str:
    """Drop query material until a query-aware allow-list exists."""

    return path.split("?", 1)[0]

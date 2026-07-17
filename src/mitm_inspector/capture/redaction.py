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
_TCHAR_PUNCTUATION = "!#$%&'*+-.^_`|~"
_SEPARATOR_RUN = re.compile(f"[{re.escape(_TCHAR_PUNCTUATION)}]+")
_LOWER_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_ACRONYM_CAMEL_BOUNDARY = re.compile(r"(?<=[A-Z])(?=[A-Z][a-z])")
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
_EXPLICIT_SENSITIVE_COMPACT_NAMES = frozenset(
    name.replace("-", "") for name in _EXPLICIT_SENSITIVE_NAMES
)
_COMPACT_SENSITIVE_SUFFIXES = (
    "authorization",
    "authtoken",
    "bearertoken",
    "accesstoken",
    "refreshtoken",
    "idtoken",
    "securitytoken",
    "apikey",
    "clientsecret",
    "credentials",
    "credential",
    "signature",
    "cookie",
    "secret",
    "token",
)
_CREDENTIAL_SEGMENTS = frozenset(
    {
        "auth",
        "authorization",
        "bearer",
        "cookie",
        "credential",
        "credentials",
        "secret",
        "signature",
    }
)


def sanitize_header_name(name: str) -> str:
    """Return a comparison-only token form without changing the emitted name."""

    camel_separated = _ACRONYM_CAMEL_BOUNDARY.sub("-", name.strip())
    camel_separated = _LOWER_CAMEL_BOUNDARY.sub("-", camel_separated)
    return _SEPARATOR_RUN.sub("-", camel_separated.lower())


def is_sensitive_header_name(name: str) -> bool:
    normalized_name = sanitize_header_name(name)
    if normalized_name in _HARMLESS_SHAPED_NAMES:
        return False
    compact_name = normalized_name.replace("-", "")
    if normalized_name in SENSITIVE_HEADER_NAMES or normalized_name in _EXPLICIT_SENSITIVE_NAMES:
        return True
    if compact_name in _EXPLICIT_SENSITIVE_COMPACT_NAMES:
        return True
    if any(compact_name.endswith(suffix) for suffix in _COMPACT_SENSITIVE_SUFFIXES):
        return True
    parts = normalized_name.split("-")
    if any(part in _CREDENTIAL_SEGMENTS for part in parts):
        return True
    return parts[-1] in {"token", "key"}


def sanitize_header(name: str, value: str) -> tuple[str, str]:
    return name, REDACTED if is_sensitive_header_name(name) else value


def sanitize_path(path: str) -> str:
    """Drop query material until a query-aware allow-list exists."""

    return path.split("?", 1)[0]

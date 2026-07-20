"""Exact built-in JSON canonicalization for untrusted boundaries."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import cast

type PlainJsonValue = (
    None | bool | int | float | str | list[PlainJsonValue] | dict[str, PlainJsonValue]
)
type PlainJsonObject = dict[str, PlainJsonValue]


class JsonBoundaryError(ValueError):
    """Raised when a value cannot be represented as canonical plain JSON."""


def canonicalize_json(
    value: object,
    *,
    label: str = "value",
    _ancestors: set[int] | None = None,
) -> PlainJsonValue:
    """Copy JSON-shaped input to exact built-in scalars, dicts, and lists."""

    value_type = type(value)
    if value is None:
        return None
    if value_type is bool:
        return cast(bool, value)
    if value_type is int:
        return cast(int, value)
    if isinstance(value, str):
        text = str.__str__(value)
        if type(text) is not str:
            raise AssertionError("built-in string normalization returned a subclass")
        return text
    if value_type is float:
        number = cast(float, value)
        if not math.isfinite(number):
            raise JsonBoundaryError(f"{label} must contain finite JSON numbers")
        return number
    if isinstance(value, bool | int | float):
        raise JsonBoundaryError(f"{label} must not contain JSON scalar subclasses")

    ancestors = _ancestors if _ancestors is not None else set()
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in ancestors:
            raise JsonBoundaryError(f"{label} must not contain cycles")
        ancestors.add(identity)
        try:
            copied: dict[str, PlainJsonValue] = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    raise JsonBoundaryError(f"{label} object keys must be strings")
                canonical_key = str.__str__(key)
                if type(canonical_key) is not str:
                    raise AssertionError("built-in key normalization returned a subclass")
                if canonical_key in copied:
                    raise JsonBoundaryError(
                        f"{label} contains duplicate keys after string normalization"
                    )
                copied[canonical_key] = canonicalize_json(
                    item,
                    label=f"{label}.{canonical_key}",
                    _ancestors=ancestors,
                )
            return copied
        finally:
            ancestors.remove(identity)

    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        identity = id(value)
        if identity in ancestors:
            raise JsonBoundaryError(f"{label} must not contain cycles")
        ancestors.add(identity)
        try:
            return [
                canonicalize_json(
                    item,
                    label=f"{label}[{index}]",
                    _ancestors=ancestors,
                )
                for index, item in enumerate(value)
            ]
        finally:
            ancestors.remove(identity)

    raise JsonBoundaryError(f"{label} must contain only JSON values")

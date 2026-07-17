"""Thin stock-mitmdump entrypoint for the capture addon."""

from mitm_inspector.capture.adapter import (
    CaptureAddon,
    addons,
    make_addon_from_environment,
)

__all__ = ["CaptureAddon", "addons", "make_addon_from_environment"]

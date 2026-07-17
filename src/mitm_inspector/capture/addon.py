"""Thin stock-mitmdump entrypoint composing capture with the socket pump."""

from mitm_inspector.capture.adapter import CaptureAddon, make_addon_from_environment
from mitm_inspector.capture.pump import CaptureSocketPump

_capture_addon = CaptureAddon()

# mitmdump -s imports this module and discovers the documented addon list.
# The pump shares the capture addon's bounded sink and starts only when the
# runtime provided a capture socket in the environment.
addons = [_capture_addon, CaptureSocketPump(_capture_addon)]

__all__ = ["CaptureAddon", "CaptureSocketPump", "addons", "make_addon_from_environment"]

"""Public-hook capture and pre-transport redaction boundary."""

from mitm_inspector.capture.addon import CaptureAddon
from mitm_inspector.capture.sink import BoundedMessageSink

__all__ = ["BoundedMessageSink", "CaptureAddon"]

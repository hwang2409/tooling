"""Versioned loopback HTTP/WebSocket API boundary for protocol-v1 messages.

The server entry point stays in :mod:`mitm_inspector.api.server` and is not
re-exported here, so ``python -m mitm_inspector.api.server`` never imports
the module twice.
"""

from mitm_inspector.api.app import ApiApplication

__all__ = ["ApiApplication"]

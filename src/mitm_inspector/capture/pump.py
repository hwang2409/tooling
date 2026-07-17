"""Proxy-side pump draining the capture sink into the runtime's ingest socket.

The pump is a second mitmproxy addon that shares one :class:`CaptureAddon`.
It runs entirely on mitmproxy's event loop: ``running`` starts one background
task, ``done`` requests a bounded final flush.  Messages are serialized as
strict JSONL and written to the private Unix socket owned by the API server.

Delivery is at-most-once by design: while the socket is unavailable the
bounded sink keeps absorbing hooks and records drops, and a batch that fails
mid-write is recorded as lost so the sequencer emits ``stream.gap`` on the
next successful drain instead of silently hiding the hole.
"""

from __future__ import annotations

import asyncio
import json

from mitm_inspector.api.limits import MAX_INGEST_LINE_BYTES
from mitm_inspector.capture.adapter import CaptureAddon
from mitm_inspector.protocol import ParsedMessageResult, parsed_message_to_plain_json

DEFAULT_DRAIN_LIMIT = 256
DEFAULT_POLL_INTERVAL_SECONDS = 0.05
DEFAULT_RECONNECT_MIN_SECONDS = 0.1
DEFAULT_RECONNECT_MAX_SECONDS = 2.0
SHUTDOWN_FLUSH_TIMEOUT_SECONDS = 2.0
WRITER_CLOSE_TIMEOUT_SECONDS = 0.25


def serialize_message(message: ParsedMessageResult) -> bytes:
    """Serialize one drained message as a strict single-line JSONL record."""

    text = json.dumps(parsed_message_to_plain_json(message), separators=(",", ":"))
    return text.encode("utf-8") + b"\n"


class CaptureSocketPump:
    """Ship drained capture messages to ``MITM_INSPECTOR_CAPTURE_SOCKET``."""

    def __init__(
        self,
        addon: CaptureAddon,
        *,
        drain_limit: int = DEFAULT_DRAIN_LIMIT,
        poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
        reconnect_min_seconds: float = DEFAULT_RECONNECT_MIN_SECONDS,
        reconnect_max_seconds: float = DEFAULT_RECONNECT_MAX_SECONDS,
        shutdown_flush_timeout_seconds: float = SHUTDOWN_FLUSH_TIMEOUT_SECONDS,
    ) -> None:
        if type(drain_limit) is not int or drain_limit < 1:
            raise ValueError("drain_limit must be a positive integer")
        for name, value in (
            ("poll_interval_seconds", poll_interval_seconds),
            ("reconnect_min_seconds", reconnect_min_seconds),
            ("reconnect_max_seconds", reconnect_max_seconds),
            ("shutdown_flush_timeout_seconds", shutdown_flush_timeout_seconds),
        ):
            if type(value) not in {int, float} or not value > 0:
                raise ValueError(f"{name} must be a positive number")
        self._addon = addon
        self._drain_limit = drain_limit
        self._poll_interval = float(poll_interval_seconds)
        self._reconnect_min = float(reconnect_min_seconds)
        self._reconnect_max = float(reconnect_max_seconds)
        self._shutdown_flush_timeout = float(shutdown_flush_timeout_seconds)
        self._task: asyncio.Task[None] | None = None
        self._stopping = False

    @property
    def active(self) -> bool:
        return self._task is not None and not self._task.done()

    def running(self) -> None:
        """Start the pump once mitmproxy's event loop is live."""

        if self._task is not None and not self._task.done():
            return
        socket_path = self._addon.capture_socket
        if socket_path is None:
            return
        self._stopping = False
        self._task = asyncio.get_running_loop().create_task(self._pump(socket_path))

    async def done(self) -> None:
        """Request a bounded final flush, then cancel whatever remains."""

        task = self._task
        if task is None:
            return
        self._stopping = True
        try:
            await asyncio.wait_for(asyncio.shield(task), self._shutdown_flush_timeout)
        except (TimeoutError, asyncio.CancelledError):
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        finally:
            self._task = None

    async def _pump(self, socket_path: str) -> None:
        backoff = self._reconnect_min
        while True:
            try:
                _reader, writer = await asyncio.open_unix_connection(socket_path)
            except OSError:
                if self._stopping:
                    return
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self._reconnect_max)
                continue
            backoff = self._reconnect_min
            clean_stop = False
            try:
                clean_stop = await self._run_connected(writer)
            except OSError:
                pass
            finally:
                await _close_writer(writer)
            if clean_stop or self._stopping:
                return

    async def _run_connected(self, writer: asyncio.StreamWriter) -> bool:
        """Drain and write until the transport fails or a stop drains dry."""

        while True:
            messages = self._addon.drain(self._drain_limit)
            if not messages:
                if self._stopping:
                    return True
                await asyncio.sleep(self._poll_interval)
                continue
            try:
                for message in messages:
                    line = serialize_message(message)
                    if len(line) > MAX_INGEST_LINE_BYTES:
                        # A line the ingest listener would reject must never
                        # reach the wire: it would desync and drop the whole
                        # producer connection.  Configured body prefixes are
                        # provably bounded, so only pathological header/URL
                        # envelopes can get here; count the message as lost.
                        self._addon.sink.record_loss()
                        continue
                    writer.write(line)
                await writer.drain()
            except (OSError, RuntimeError) as error:
                # The batch was already acknowledged by drain(); whatever the
                # reader did not receive is a real loss, so record it and let
                # the sequencer surface a stream.gap after reconnect.
                for _ in messages:
                    self._addon.sink.record_loss()
                if isinstance(error, RuntimeError):
                    raise ConnectionResetError("ingest writer is closed") from error
                raise


async def _close_writer(writer: asyncio.StreamWriter) -> None:
    try:
        writer.close()
        await asyncio.wait_for(writer.wait_closed(), WRITER_CLOSE_TIMEOUT_SECONDS)
    except (OSError, RuntimeError, TimeoutError):
        writer.transport.abort()
    except asyncio.CancelledError:
        writer.transport.abort()
        raise


__all__ = ["CaptureSocketPump", "serialize_message"]

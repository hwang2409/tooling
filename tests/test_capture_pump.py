"""Unit tests for the proxy-side capture socket pump."""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from mitm_inspector.capture.adapter import CaptureAddon
from mitm_inspector.capture.addon import addons
from mitm_inspector.capture.pump import CaptureSocketPump, serialize_message


@pytest.fixture
def socket_dir() -> Iterator[Path]:
    # macOS bounds AF_UNIX paths at 104 bytes; pytest tmp_path is too deep.
    path = Path(tempfile.mkdtemp(prefix="mi-pump-", dir="/tmp"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _hello(source_id: str = "pump-test") -> dict[str, object]:
    return {
        "protocol_version": "1",
        "type": "source.hello",
        "source_id": source_id,
        "occurred_at": "2026-01-01T00:00:00Z",
        "capabilities": {"body_chunks": True, "redaction": "headers-and-query"},
        "limits": {"max_body_prefix_bytes": "64", "max_in_memory_bytes": "1024"},
    }


def _lifecycle(sequence: int, flow_id: str = "flow-1") -> dict[str, object]:
    return {
        "protocol_version": "1",
        "type": "flow.lifecycle",
        "source_id": "pump-test",
        "flow_id": flow_id,
        "event_id": f"pump-test:{sequence}",
        "occurred_at": "2026-01-01T00:00:00Z",
        "sequence": str(sequence),
        "state": "request_started",
    }


class _LineCollector:
    def __init__(self) -> None:
        self.lines: list[dict[str, object]] = []
        self.connections = 0
        self.writers: list[asyncio.StreamWriter] = []

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.connections += 1
        self.writers.append(writer)
        try:
            while True:
                line = await reader.readline()
                if not line:
                    return
                self.lines.append(json.loads(line))
        finally:
            writer.close()

    async def close_connections(self) -> None:
        writers = tuple(self.writers)
        for writer in writers:
            writer.close()
        await asyncio.gather(*(writer.wait_closed() for writer in writers), return_exceptions=True)


class _FailFirstWriteCollector(_LineCollector):
    """Abort the first live transport after its hello has been received."""

    def __init__(self) -> None:
        super().__init__()
        self.hello_received = asyncio.Event()
        self.transport_aborted = asyncio.Event()

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.connections += 1
        self.writers.append(writer)
        try:
            line = await reader.readline()
            if not line:
                return
            self.lines.append(json.loads(line))
            if self.connections == 1:
                self.hello_received.set()
                writer.transport.abort()
                self.transport_aborted.set()
                return
            while True:
                line = await reader.readline()
                if not line:
                    return
                self.lines.append(json.loads(line))
        finally:
            writer.close()


class _BackpressureWriter:
    def __init__(self) -> None:
        self.pending_bytes = 0
        self.max_pending_bytes = 0
        self.drain_calls = 0

    def write(self, data: bytes) -> None:
        self.pending_bytes += len(data)
        self.max_pending_bytes = max(self.max_pending_bytes, self.pending_bytes)

    async def drain(self) -> None:
        self.drain_calls += 1
        self.pending_bytes = 0


async def _wait_for(predicate, timeout: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("condition was not reached in time")
        await asyncio.sleep(0.01)


def test_addon_module_composes_capture_and_pump() -> None:
    assert isinstance(addons[0], CaptureAddon)
    assert isinstance(addons[1], CaptureSocketPump)
    assert addons[1]._addon is addons[0]


def test_pump_rejects_invalid_settings() -> None:
    addon = CaptureAddon()
    with pytest.raises(ValueError):
        CaptureSocketPump(addon, drain_limit=0)
    with pytest.raises(ValueError):
        CaptureSocketPump(addon, poll_interval_seconds=0)
    with pytest.raises(ValueError):
        CaptureSocketPump(addon, reconnect_max_seconds=-1)


def test_serialize_message_is_single_line_utf8() -> None:
    addon = CaptureAddon()
    addon.sink.offer(_hello())
    (message,) = addon.drain()
    line = serialize_message(message)
    assert line.endswith(b"\n")
    assert line.count(b"\n") == 1
    decoded = json.loads(line)
    assert decoded["type"] == "source.hello"
    assert decoded["delivery_position"] == "1"


def test_running_without_socket_is_inert() -> None:
    async def scenario() -> None:
        addon = CaptureAddon()
        pump = CaptureSocketPump(addon)
        pump.running()
        assert not pump.active
        await pump.done()

    asyncio.run(scenario())


def test_pump_delivers_messages_and_flushes_on_done(socket_dir: Path) -> None:
    async def scenario() -> None:
        socket_path = str(socket_dir / "capture.sock")
        collector = _LineCollector()
        server = await asyncio.start_unix_server(collector.handle, path=socket_path)
        addon = CaptureAddon()
        addon.capture_socket = socket_path
        pump = CaptureSocketPump(addon, poll_interval_seconds=0.01)
        assert addon.sink.offer(_hello())
        pump.running()
        assert pump.active
        await _wait_for(lambda: len(collector.lines) >= 1)
        assert addon.sink.offer(_lifecycle(0))
        assert addon.sink.offer(_lifecycle(1, flow_id="flow-2"))
        await pump.done()
        assert not pump.active
        server.close()
        await server.wait_closed()
        types = [line["type"] for line in collector.lines]
        assert types == ["source.hello", "flow.lifecycle", "flow.lifecycle"]
        positions = [line["delivery_position"] for line in collector.lines]
        assert positions == ["1", "2", "3"]

    asyncio.run(scenario())


def test_pump_reconnects_and_reports_gap_after_listener_restart(socket_dir: Path) -> None:
    async def scenario() -> None:
        socket_path = str(socket_dir / "capture.sock")
        collector = _FailFirstWriteCollector()
        server = await asyncio.start_unix_server(collector.handle, path=socket_path)
        addon = CaptureAddon()
        addon.capture_socket = socket_path
        pump = CaptureSocketPump(
            addon,
            poll_interval_seconds=0.01,
            reconnect_min_seconds=0.01,
            reconnect_max_seconds=0.05,
        )
        assert addon.sink.offer(_hello())
        pump.running()
        await asyncio.wait_for(collector.hello_received.wait(), timeout=5.0)
        await asyncio.wait_for(collector.transport_aborted.wait(), timeout=5.0)

        # The peer reset is a real write failure, not a mocked drain error.
        # The queued lifecycle must be recorded as lost before reconnecting.
        assert addon.sink.offer(_lifecycle(0))
        await _wait_for(lambda: addon.sink.dropped_count >= 1)
        assert addon.sink.dropped_count >= 1
        assert addon.sink.offer(_lifecycle(1, flow_id="flow-2"))
        await _wait_for(
            lambda: "stream.gap" in [line["type"] for line in collector.lines]
            and any(line.get("flow_id") == "flow-2" for line in collector.lines)
        )
        await pump.done()
        server.close()
        await server.wait_closed()

        types = [line["type"] for line in collector.lines]
        assert types[0] == "source.hello"
        assert collector.connections >= 2
        assert "stream.gap" in types
        assert any(line.get("flow_id") == "flow-2" for line in collector.lines)

    asyncio.run(scenario())


def test_pump_drops_oversized_lines_instead_of_desyncing(
    socket_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A line the ingest listener would reject is counted as a loss, not sent."""

    import mitm_inspector.capture.pump as pump_module

    monkeypatch.setattr(pump_module, "MAX_INGEST_LINE_BYTES", 256)

    async def scenario() -> None:
        socket_path = str(socket_dir / "capture.sock")
        collector = _LineCollector()
        server = await asyncio.start_unix_server(collector.handle, path=socket_path)
        addon = CaptureAddon()
        addon.capture_socket = socket_path
        pump = CaptureSocketPump(addon, poll_interval_seconds=0.01)
        oversized = _lifecycle(0)
        oversized["flow_id"] = "f" * 400
        oversized["event_id"] = f"{oversized['flow_id']}:0"
        assert addon.sink.offer(oversized)
        assert addon.sink.offer(_lifecycle(1, flow_id="small"))
        pump.running()
        await _wait_for(
            lambda: any(line["type"] == "flow.lifecycle" for line in collector.lines)
        )
        await pump.done()
        server.close()
        await server.wait_closed()

        lifecycles = [line for line in collector.lines if line["type"] == "flow.lifecycle"]
        assert [line["flow_id"] for line in lifecycles] == ["small"]
        assert all(len(json.dumps(line)) <= 512 for line in collector.lines)
        assert addon.sink.dropped_count >= 1

    asyncio.run(scenario())


def test_pump_applies_backpressure_per_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """A full drain batch never accumulates more than one serialized message."""

    import mitm_inspector.capture.pump as pump_module

    message_size = 32 * 1024
    monkeypatch.setattr(pump_module, "serialize_message", lambda _message: b"x" * message_size)

    async def scenario() -> None:
        addon = CaptureAddon()
        for sequence in range(256):
            assert addon.sink.offer(_lifecycle(sequence, flow_id=f"flow-{sequence}"))
        pump = CaptureSocketPump(addon, drain_limit=256)
        pump._stopping = True
        writer = _BackpressureWriter()
        assert await pump._run_connected(writer) is True
        assert writer.drain_calls == 256
        assert writer.max_pending_bytes <= message_size

    asyncio.run(scenario())

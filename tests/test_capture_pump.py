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

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.connections += 1
        try:
            while True:
                line = await reader.readline()
                if not line:
                    return
                self.lines.append(json.loads(line))
        finally:
            writer.close()


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
        collector = _LineCollector()
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
        await _wait_for(lambda: len(collector.lines) >= 1)

        # Drop the listener entirely, lose one batch, then restart it.
        server.close()
        await server.wait_closed()
        Path(socket_path).unlink()
        assert addon.sink.offer(_lifecycle(0))
        await _wait_for(lambda: addon.sink.dropped_count >= 1 or pump.active)
        # Force the pending message out against the dead endpoint.
        await asyncio.sleep(0.1)

        server = await asyncio.start_unix_server(collector.handle, path=socket_path)
        assert addon.sink.offer(_lifecycle(1))
        await _wait_for(lambda: any(line["type"] == "flow.lifecycle" for line in collector.lines))
        await pump.done()
        server.close()
        await server.wait_closed()

        types = [line["type"] for line in collector.lines]
        assert types[0] == "source.hello"
        assert "flow.lifecycle" in types
        # The lost write surfaced as a stream.gap before the next delivery.
        if addon.sink.dropped_count:
            assert "stream.gap" in types

    asyncio.run(scenario())

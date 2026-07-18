from __future__ import annotations

import base64
import json
import os
import stat
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from mitm_inspector.api.app import ApiApplication
from mitm_inspector.api.projection import collect_grid_flows
from mitm_inspector.api.server import ApiServer, ApiServerConfig
from mitm_inspector.capture.addon import CaptureAddon
from mitm_inspector.protocol import KnownParsedMessage, parsed_message_to_plain_json
from mitm_inspector.store.memory import MemoryStore
from mitm_inspector.store.sqlite import SQLiteFlowStorage


class FakeHeaders(dict[str, str]):
    def items(self, multi: bool = False):
        return iter(super().items())


def fake_flow(flow_id: str = "adapter-flow") -> SimpleNamespace:
    return SimpleNamespace(
        id=flow_id,
        request=SimpleNamespace(
            method="POST",
            scheme="https",
            host="example.test",
            port=443,
            path="/messages",
            headers=FakeHeaders({"Host": "example.test"}),
            raw_content=b"request-body",
            stream=False,
        ),
        response=SimpleNamespace(
            status_code=200,
            headers=FakeHeaders({"Content-Type": "text/plain"}),
            raw_content=b"response-body",
            stream=False,
        ),
        error=None,
    )


def body(data: bytes, *, state: str = "captured") -> dict[str, object]:
    return {
        "state": state,
        "size_bytes": str(len(data)),
        "encoding": "base64",
        "data": base64.b64encode(data).decode("ascii"),
    }


def metadata(flow_id: str, request: bytes, response: bytes) -> dict[str, object]:
    return {
        "protocol_version": "1",
        "type": "flow.metadata",
        "metadata": {
            "flow_id": flow_id,
            "method": "POST",
            "scheme": "https",
            "host": "example.test",
            "port": "443",
            "path": "/messages",
            "request_headers": [{"name": "content-type", "value": "application/json"}],
            "response_headers": [{"name": "content-type", "value": "text/plain"}],
            "response_status": "200",
            "request_body": body(request),
            "response_body": body(response),
        },
    }


def lifecycle(flow_id: str, sequence: int, occurred_at: str, state: str) -> dict[str, object]:
    return {
        "protocol_version": "1",
        "type": "flow.lifecycle",
        "source_id": "test-source",
        "flow_id": flow_id,
        "event_id": f"event-{flow_id}-{sequence}",
        "occurred_at": occurred_at,
        "sequence": str(sequence),
        "state": state,
    }


def test_sqlite_round_trip_preserves_metadata_lifecycle_and_bodies(tmp_path: Path) -> None:
    storage = SQLiteFlowStorage(tmp_path / "flows.sqlite")
    try:
        storage.offer(metadata("flow-1", b"request\x00", b"response\xff"))
        storage.offer(lifecycle("flow-1", 1, "2026-01-01T00:00:00Z", "request_started"))
        storage.offer(lifecycle("flow-1", 2, "2026-01-01T00:00:01Z", "flow_completed"))
        storage.flush()

        messages = [parsed_message_to_plain_json(message) for message in storage.replay()]
        replayed_metadata = next(
            message for message in messages if message["type"] == "flow.metadata"
        )
        metadata_value = replayed_metadata["metadata"]
        assert isinstance(metadata_value, dict)
        assert base64.b64decode(metadata_value["request_body"]["data"]) == b"request\x00"
        assert base64.b64decode(metadata_value["response_body"]["data"]) == b"response\xff"
        assert [message["type"] for message in messages] == [
            "flow.metadata",
            "body.end",
            "body.end",
            "flow.lifecycle",
            "flow.lifecycle",
        ]
    finally:
        storage.close()


def test_sqlite_retention_evicts_oldest_flows(tmp_path: Path) -> None:
    storage = SQLiteFlowStorage(tmp_path / "flows.sqlite", max_flows=2)
    try:
        for index in range(3):
            flow_id = f"flow-{index}"
            storage.offer(metadata(flow_id, b"r", b"s"))
            storage.offer(
                lifecycle(
                    flow_id,
                    1,
                    f"2026-01-01T00:00:0{index}Z",
                    "request_started",
                )
            )
        storage.flush()
        replayed = storage.replay()
        flow_ids = [
            parsed_message_to_plain_json(message)["metadata"]["flow_id"]
            for message in replayed
            if isinstance(message, KnownParsedMessage)
            and message.message.get("type") == "flow.metadata"
        ]
        assert flow_ids == ["flow-2", "flow-1"]
    finally:
        storage.close()


def test_sqlite_replay_materializes_newest_flows_oldest_first_in_browser_store(
    tmp_path: Path,
) -> None:
    storage = SQLiteFlowStorage(tmp_path / "flows.sqlite")
    try:
        for index in range(3):
            flow_id = f"flow-{index}"
            storage.offer(metadata(flow_id, b"r", b"s"))
            storage.offer(
                lifecycle(flow_id, 1, f"2026-01-01T00:00:0{index}Z", "request_started")
            )
        memory = MemoryStore(10)
        storage.replay_into(memory, 2)
        assert list(collect_grid_flows(memory)) == ["flow-1", "flow-2"]
    finally:
        storage.close()


def test_sqlite_backpressure_drops_oldest_pending_message(tmp_path: Path) -> None:
    storage = SQLiteFlowStorage(tmp_path / "flows.sqlite", queue_size=1)
    started = threading.Event()
    release = threading.Event()
    original_write = storage._write

    def blocked_write(connection: object, item: object) -> None:
        started.set()
        release.wait(5)
        original_write(connection, item)  # type: ignore[arg-type]

    storage._write = blocked_write  # type: ignore[method-assign]
    try:
        storage.offer(metadata("flow-1", b"r", b"s"))
        assert started.wait(5)
        storage.offer(metadata("flow-2", b"r", b"s"))
        storage.offer(metadata("flow-3", b"r", b"s"))
        assert storage.counters["dropped_messages"] == 1
        release.set()
        storage.flush()
    finally:
        release.set()
        storage.close()


def test_partial_body_chunks_survive_a_crash_mid_capture(tmp_path: Path) -> None:
    path = tmp_path / "flows.sqlite"
    storage = SQLiteFlowStorage(path)
    storage.offer(metadata("partial", b"", b""))
    storage.offer(
        {
            "protocol_version": "1",
            "type": "body.chunk",
            "flow_id": "partial",
            "body_side": "response",
            "chunk_index": "0",
            "offset_bytes": "0",
            "data_base64": base64.b64encode(b"partial-response").decode("ascii"),
        }
    )
    storage.close()

    restarted = SQLiteFlowStorage(path)
    try:
        messages = [parsed_message_to_plain_json(message) for message in restarted.replay()]
        chunks = [message for message in messages if message["type"] == "body.chunk"]
        assert len(chunks) == 1
        assert base64.b64decode(chunks[0]["data_base64"]) == b"partial-response"
    finally:
        restarted.close()


def test_adapter_sink_writer_fresh_backend_replays_browser_state(tmp_path: Path) -> None:
    path = tmp_path / "flows.sqlite"
    storage = SQLiteFlowStorage(path)
    addon = CaptureAddon(emit=storage.offer, max_body_prefix_bytes=1024)
    flow = fake_flow()
    addon.requestheaders(flow)
    addon.responseheaders(flow)
    addon.response(flow)
    addon.request(flow)
    addon.drain()
    storage.close()

    restarted_storage = SQLiteFlowStorage(path)
    try:
        memory = MemoryStore(20)
        restarted_storage.replay_into(memory, 20)
        application = ApiApplication(memory, source_id="mitm-inspector", storage=restarted_storage)
        frames: list[dict[str, object]] = []
        application.subscribe(
            lambda frame: frames.append(json.loads(frame)) or True,
        )
        snapshot = next(frame for frame in frames if frame["type"] == "browser.snapshot")
        assert [flow["flow_id"] for flow in snapshot["flows"]] == ["adapter-flow"]
        assert any(
            frame["type"] == "flow.lifecycle" and frame["flow_id"] == "adapter-flow"
            for frame in frames
        )
    finally:
        restarted_storage.close()


def test_replay_uses_reverse_chronological_flow_order(tmp_path: Path) -> None:
    storage = SQLiteFlowStorage(tmp_path / "flows.sqlite")
    try:
        for index in range(3):
            flow_id = f"ordered-{index}"
            storage.offer(metadata(flow_id, b"r", b"s"))
            storage.offer(
                lifecycle(flow_id, 1, f"2026-02-01T00:00:0{index}Z", "request_started")
            )
        replayed = storage.replay(2)
        metadata_ids = [
            parsed_message_to_plain_json(message)["metadata"]["flow_id"]
            for message in replayed
            if isinstance(message, KnownParsedMessage)
            and message.message.get("type") == "flow.metadata"
        ]
        assert metadata_ids == ["ordered-2", "ordered-1"]
    finally:
        storage.close()


def test_replay_after_fresh_backend_instance_delivers_lifecycle_history(tmp_path: Path) -> None:
    path = tmp_path / "flows.sqlite"
    first = SQLiteFlowStorage(path)
    first.offer(metadata("restart-flow", b"r", b"s"))
    first.offer(lifecycle("restart-flow", 1, "2026-02-01T00:00:00Z", "request_started"))
    first.offer(lifecycle("restart-flow", 2, "2026-02-01T00:00:01Z", "flow_completed"))
    first.close()

    second = SQLiteFlowStorage(path)
    try:
        memory = MemoryStore(20)
        second.replay_into(memory, 20)
        application = ApiApplication(memory, source_id="mitm-inspector", storage=second)
        frames: list[dict[str, object]] = []
        application.subscribe(lambda frame: frames.append(json.loads(frame)) or True)
        lifecycle_frames = [frame for frame in frames if frame["type"] == "flow.lifecycle"]
        assert [frame["state"] for frame in lifecycle_frames] == [
            "request_started",
            "flow_completed",
        ]
    finally:
        second.close()


def test_no_storage_and_memory_storage_paths_disable_persistence(tmp_path: Path) -> None:
    memory_server = ApiServer(ApiServerConfig(storage_path=":memory:"))
    disabled_server = ApiServer(
        ApiServerConfig(storage_path=tmp_path / "disabled.sqlite", no_storage=True)
    )
    assert memory_server.application.storage is None
    assert disabled_server.application.storage is None
    assert not (tmp_path / "disabled.sqlite").exists()


def test_storage_byte_retention_bounds_the_database_file_set(tmp_path: Path) -> None:
    path = tmp_path / "flows.sqlite"
    max_bytes = 1_000_000
    storage = SQLiteFlowStorage(path, max_flows=10_000, max_bytes=max_bytes)
    try:
        for index in range(100):
            storage.offer(metadata(f"bytes-{index}", b"x" * 1_000, b"y" * 1_000))
        storage.flush()
        files = [path, Path(f"{path}-wal"), Path(f"{path}-shm")]
        assert sum(file.stat().st_size for file in files if file.exists()) <= max_bytes
        assert sum(
            1
            for message in storage.replay()
            if isinstance(message, KnownParsedMessage)
            and message.message.get("type") == "flow.metadata"
        ) < 100
    finally:
        storage.close()


def test_storage_rejects_loose_permissions_and_creates_private_files(tmp_path: Path) -> None:
    unsafe = tmp_path / "unsafe"
    unsafe.mkdir(mode=0o755)
    os.chmod(unsafe, 0o755)
    with pytest.raises(PermissionError):
        SQLiteFlowStorage(unsafe / "flows.sqlite")

    secure = tmp_path / "secure"
    storage = SQLiteFlowStorage(secure / "flows.sqlite")
    try:
        assert stat.S_IMODE(secure.stat().st_mode) == 0o700
        assert stat.S_IMODE((secure / "flows.sqlite").stat().st_mode) == 0o600
    finally:
        storage.close()


def test_representative_load_does_not_drop_messages(tmp_path: Path) -> None:
    storage = SQLiteFlowStorage(
        tmp_path / "flows.sqlite",
        max_flows=10_000,
        max_bytes=32 * 1024 * 1024,
        queue_size=4_096,
    )
    try:
        for index in range(1_000):
            assert storage.offer(metadata(f"load-{index}", b"r", b"s"))
        storage.flush()
        assert storage.counters["dropped_messages"] == 0
        assert storage.counters["write_errors"] == 0
    finally:
        storage.close()

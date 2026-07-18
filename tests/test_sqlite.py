from __future__ import annotations

import asyncio
import base64
import json
import os
import stat
import subprocess
import sys
import threading
import time
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


def test_replay_preserves_global_interleaved_lifecycle_sequence(tmp_path: Path) -> None:
    storage = SQLiteFlowStorage(tmp_path / "flows.sqlite")
    try:
        for flow_id in ("flow-a", "flow-b"):
            storage.offer(metadata(flow_id, b"r", b"s"))
        for flow_id, sequence in (
            ("flow-a", 1),
            ("flow-b", 2),
            ("flow-a", 3),
            ("flow-b", 4),
        ):
            storage.offer(
                lifecycle(
                    flow_id,
                    sequence,
                    f"2026-01-01T00:00:0{sequence}Z",
                    "request_started",
                )
            )
        storage.flush()
        replayed = [
            parsed_message_to_plain_json(message)
            for message in storage.replay()
            if isinstance(message, KnownParsedMessage)
            and message.message.get("type") == "flow.lifecycle"
        ]
        assert [(message["flow_id"], message["sequence"]) for message in replayed] == [
            ("flow-a", "1"),
            ("flow-b", "2"),
            ("flow-a", "3"),
            ("flow-b", "4"),
        ]
    finally:
        storage.close()


def test_sqlite_retention_evicts_oldest_flows_with_max_plus_ten_writes(tmp_path: Path) -> None:
    storage = SQLiteFlowStorage(tmp_path / "flows.sqlite", max_flows=2)
    try:
        for index in range(12):
            flow_id = f"flow-{index}"
            storage.offer(metadata(flow_id, b"r", b"s"))
            storage.offer(
                lifecycle(
                    flow_id,
                    1,
                    f"2026-01-01T00:00:{index:02d}Z",
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
        assert flow_ids == ["flow-11", "flow-10"]
    finally:
        storage.close()


def test_sqlite_replay_materializes_newest_flows_newest_first_in_browser_store(
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
        assert list(collect_grid_flows(memory)) == ["flow-2", "flow-1"]
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
    application = ApiApplication(MemoryStore(20), storage=storage)
    addon = CaptureAddon(emit=application.ingest, max_body_prefix_bytes=1024)
    flow = fake_flow()
    addon.requestheaders(flow)
    addon.responseheaders(flow)
    addon.response(flow)
    addon.request(flow)
    addon.drain()
    storage.close()

    restarted = ApiServer(
        ApiServerConfig(storage_path=path, storage_replay=20, max_retained_flows=20)
    )
    try:
        frames: list[dict[str, object]] = []
        restarted.application.subscribe(
            lambda frame: frames.append(json.loads(frame)) or True,
        )
        snapshot = next(frame for frame in frames if frame["type"] == "browser.snapshot")
        assert [flow["flow_id"] for flow in snapshot["flows"]] == ["adapter-flow"]
        assert any(
            frame["type"] == "flow.lifecycle" and frame["flow_id"] == "adapter-flow"
            for frame in frames
        )
        detail = json.loads(restarted.application.flow_detail_text("adapter-flow") or "null")
        assert detail["flow_id"] == "adapter-flow"
        assert any(message["type"] == "body.chunk" for message in detail["messages"])
    finally:
        asyncio.run(restarted.close())


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

    second = ApiServer(ApiServerConfig(storage_path=path, storage_replay=20))
    try:
        frames: list[dict[str, object]] = []
        second.application.subscribe(lambda frame: frames.append(json.loads(frame)) or True)
        lifecycle_frames = [frame for frame in frames if frame["type"] == "flow.lifecycle"]
        assert [frame["state"] for frame in lifecycle_frames] == [
            "request_started",
            "flow_completed",
        ]
    finally:
        asyncio.run(second.close())


def test_fresh_backend_browser_surface_is_reverse_chronological(tmp_path: Path) -> None:
    path = tmp_path / "ordered.sqlite"
    first = SQLiteFlowStorage(path)
    for index in range(2):
        first.offer(metadata(f"surface-{index}", b"r", b"s"))
        first.offer(
            lifecycle(f"surface-{index}", 1, f"2026-02-01T00:00:0{index}Z", "request_started")
        )
    first.close()

    second = ApiServer(ApiServerConfig(storage_path=path, storage_replay=20))
    try:
        frames: list[dict[str, object]] = []
        second.application.subscribe(lambda frame: frames.append(json.loads(frame)) or True)
        snapshot = next(frame for frame in frames if frame["type"] == "browser.snapshot")
        assert [flow["flow_id"] for flow in snapshot["flows"]] == ["surface-1", "surface-0"]
    finally:
        asyncio.run(second.close())


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


def test_storage_drops_single_oversized_protected_flow_to_honor_byte_cap(tmp_path: Path) -> None:
    path = tmp_path / "flows.sqlite"
    max_bytes = 100_000
    storage = SQLiteFlowStorage(path, max_flows=10_000, max_bytes=max_bytes)
    try:
        storage.offer(metadata("oversized", b"x" * 300_000, b"y" * 300_000))
        storage.flush()
        files = [path, Path(f"{path}-wal"), Path(f"{path}-shm")]
        assert sum(file.stat().st_size for file in files if file.exists()) <= max_bytes
        assert storage.replay() == []
    finally:
        storage.close()


def test_storage_rejects_caps_below_sqlite_overhead(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="SQLite overhead"):
        SQLiteFlowStorage(tmp_path / "flows.sqlite", max_bytes=1_000)


def test_storage_cap_uses_live_file_set_boundary(tmp_path: Path) -> None:
    probe = SQLiteFlowStorage(tmp_path / "probe.sqlite", max_bytes=10_000_000)
    baseline = 0
    for _ in range(100):
        files = [probe.path, Path(f"{probe.path}-wal"), Path(f"{probe.path}-shm")]
        if all(file.exists() for file in files):
            baseline = sum(file.stat().st_size for file in files)
            break
        time.sleep(0.01)
    assert baseline > 0
    probe.close()

    with pytest.raises(ValueError, match="SQLite overhead"):
        SQLiteFlowStorage(tmp_path / "below.sqlite", max_bytes=baseline - 1)
    accepted = SQLiteFlowStorage(tmp_path / "above.sqlite", max_bytes=baseline + 1)
    try:
        files = [accepted.path, Path(f"{accepted.path}-wal"), Path(f"{accepted.path}-shm")]
        assert sum(file.stat().st_size for file in files if file.exists()) <= baseline + 1
    finally:
        accepted.close()


def test_invalid_cap_reopen_preserves_retained_history(tmp_path: Path) -> None:
    path = tmp_path / "preserve.sqlite"
    seeded = SQLiteFlowStorage(path, max_bytes=10_000_000)
    for index in range(3):
        seeded.offer(metadata(f"preserved-{index}", bytes([index]), bytes([index + 3])))
    seeded.flush()
    before = [parsed_message_to_plain_json(message) for message in seeded.replay()]
    seeded.close()

    with pytest.raises(ValueError, match="SQLite overhead"):
        SQLiteFlowStorage(path, max_bytes=1_000)

    reopened = SQLiteFlowStorage(path, max_bytes=10_000_000)
    try:
        after = [parsed_message_to_plain_json(message) for message in reopened.replay()]
        assert after == before
        assert sum(
            1
            for message in after
            if message["type"] == "flow.metadata"
        ) == 3
    finally:
        reopened.close()


def test_storage_reopens_after_crash_during_retention_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "crash.sqlite"
    script = """
import base64
import os
import sys
from pathlib import Path
import mitm_inspector.store.sqlite as sqlite

path = Path(sys.argv[1])
storage = sqlite.SQLiteFlowStorage(path, max_bytes=100_000)
original_checkpoint = sqlite._checkpoint
calls = 0

def crash_checkpoint(connection):
    global calls
    calls += 1
    if calls == 1:
        os._exit(73)
    original_checkpoint(connection)

sqlite._checkpoint = crash_checkpoint
body = {
    "state": "captured",
    "size_bytes": "300000",
    "encoding": "base64",
    "data": base64.b64encode(b"x" * 300000).decode("ascii"),
}
storage.offer({
    "protocol_version": "1",
    "type": "flow.metadata",
    "metadata": {
        "flow_id": "crash-flow",
        "method": "POST",
        "scheme": "https",
        "host": "example.test",
        "port": "443",
        "path": "/",
        "request_headers": [],
        "request_body": body,
        "response_body": body,
    },
})
storage.flush()
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 73, completed.stderr
    reopened = SQLiteFlowStorage(path, max_bytes=100_000)
    try:
        assert reopened.replay() == []
        files = [path, Path(f"{path}-wal"), Path(f"{path}-shm")]
        assert sum(file.stat().st_size for file in files if file.exists()) <= 100_000
    finally:
        reopened.close()


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


@pytest.mark.parametrize("sidecar", ["db", "wal", "shm"])
def test_storage_rejects_preexisting_loose_database_files(tmp_path: Path, sidecar: str) -> None:
    path = tmp_path / f"{sidecar}.sqlite"
    storage = SQLiteFlowStorage(path)
    storage.close()
    candidate = path if sidecar == "db" else Path(f"{path}-{sidecar}")
    if not candidate.exists():
        candidate.write_bytes(b"sidecar")
    os.chmod(candidate, 0o644)
    with pytest.raises(PermissionError):
        SQLiteFlowStorage(path)


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


def test_retention_does_not_run_payload_aggregate_scans_per_message(tmp_path: Path) -> None:
    traces: list[str] = []
    storage = SQLiteFlowStorage(
        tmp_path / "flows.sqlite",
        max_flows=10_000,
        max_bytes=32 * 1024 * 1024,
        trace_sql=traces.append,
    )
    try:
        for index in range(100):
            assert storage.offer(metadata(f"trace-{index}", b"r", b"s"))
        storage.flush()
        assert not any("SUM(" in sql.upper() or "COUNT(" in sql.upper() for sql in traces)
    finally:
        storage.close()


def test_storage_normalizes_dotdot_and_rejects_ancestor_symlinks(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must not contain '..'"):
        SQLiteFlowStorage(tmp_path / "nested" / ".." / "nested" / "flows.sqlite")

    real_parent = tmp_path / "real"
    real_parent.mkdir()
    symlink_parent = tmp_path / "linked"
    symlink_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(PermissionError, match="symlinked ancestor"):
        SQLiteFlowStorage(symlink_parent / "flows.sqlite")

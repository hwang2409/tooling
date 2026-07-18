from __future__ import annotations

import base64
import threading
from pathlib import Path

from mitm_inspector.api.projection import collect_grid_flows
from mitm_inspector.protocol import KnownParsedMessage, parsed_message_to_plain_json
from mitm_inspector.store.memory import MemoryStore
from mitm_inspector.store.sqlite import SQLiteFlowStorage


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

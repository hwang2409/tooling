import asyncio
import base64
import gzip
import json
from pathlib import Path

from mitm_inspector.api.server import ApiServer, ApiServerConfig
from mitm_inspector.store.sqlite import SQLiteFlowStorage


def metadata(flow_id: str, session_id: str | None, prompt: str) -> dict[str, object]:
    body = json.dumps(
        {"model": "claude-test", "messages": [{"role": "user", "content": prompt}]}
    ).encode()
    descriptor = {
        "state": "captured",
        "size_bytes": str(len(body)),
        "encoding": "base64",
        "data": base64.b64encode(body).decode(),
        "content_type": "application/json",
    }
    return {
        "protocol_version": "1",
        "type": "flow.metadata",
        "metadata": {
            "flow_id": flow_id,
            "session_id": session_id,
            "method": "POST",
            "scheme": "https",
            "host": "api.anthropic.com",
            "port": "443",
            "path": "/v1/messages",
            "request_headers": [],
            "request_body": descriptor,
        },
    }


def lifecycle(flow_id: str, sequence: str, state: str, occurred_at: str) -> dict[str, object]:
    return {
        "protocol_version": "1",
        "type": "flow.lifecycle",
        "source_id": "test",
        "flow_id": flow_id,
        "event_id": f"{flow_id}-{sequence}",
        "occurred_at": occurred_at,
        "sequence": sequence,
        "state": state,
    }


async def request(port: int, path: str) -> tuple[bytes, bytes]:
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(
        f"GET {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n".encode()
    )
    await writer.drain()
    raw = await reader.read()
    writer.close()
    await writer.wait_closed()
    head, _, body = raw.partition(b"\r\n\r\n")
    return head, body


def test_http_session_list_and_detail_are_sqlite_backed(tmp_path: Path) -> None:
    async def scenario() -> None:
        server = ApiServer(ApiServerConfig(port=0, storage_path=tmp_path / "flows.sqlite"))
        server.application.ingest(metadata("flow-old", "session-a", "first prompt"))
        server.application.ingest(
            lifecycle("flow-old", "1", "request_started", "2026-01-01T00:00:00Z")
        )
        server.application.ingest(
            lifecycle("flow-old", "2", "flow_completed", "2026-01-01T00:00:01Z")
        )
        server.application.ingest(metadata("flow-new", "session-a", "second prompt"))
        server.application.ingest(
            lifecycle("flow-new", "1", "request_started", "2026-01-01T00:00:02Z")
        )
        server.application.ingest(
            lifecycle("flow-new", "2", "flow_completed", "2026-01-01T00:00:03Z")
        )
        assert server.application.storage is not None
        server.application.storage.flush()
        await server.start()
        try:
            head, body = await request(server.bound_port, "/api/v1/sessions")
            assert head.startswith(b"HTTP/1.1 200 ")
            sessions = json.loads(body)["sessions"]
            assert sessions[0]["session_id"] == "session-a"
            assert sessions[0]["flow_count"] == 2
            # Session browsing is metadata-only. Bodies are fetched only after
            # selecting an individual flow, so summaries cannot inspect body
            # content for a query preview or model.
            assert sessions[0]["first_query"] is None
            assert sessions[0]["models"] == []

            head, body = await request(server.bound_port, "/api/v1/sessions/session-a")
            assert head.startswith(b"HTTP/1.1 200 ")
            detail = json.loads(body)
            assert [flow["flow_id"] for flow in detail["flows"]] == ["flow-old", "flow-new"]
            assert detail["flows"][0]["request_body"]["data"] == ""

            head, _ = await request(server.bound_port, "/api/v1/stream")
            assert head.startswith(b"HTTP/1.1 404 ")
        finally:
            await server.close()

    asyncio.run(scenario())


def test_session_list_cap_is_by_session_count(tmp_path: Path) -> None:
    storage = SQLiteFlowStorage(tmp_path / "flows.sqlite")
    try:
        for index in range(200):
            flow_id = f"dominant-{index}"
            storage.offer(metadata(flow_id, "session-dominant", f"prompt-{index}"))
            minute, second = divmod(index, 60)
            storage.offer(
                lifecycle(
                    flow_id,
                    "1",
                    "request_started",
                    f"2026-01-01T00:{minute:02d}:{second:02d}Z",
                )
            )
        storage.offer(metadata("singleton-old", "session-old", "old"))
        storage.offer(
            lifecycle("singleton-old", "1", "request_started", "2026-01-02T00:00:00Z")
        )
        storage.offer(metadata("singleton-new", "session-new", "new"))
        storage.offer(
            lifecycle("singleton-new", "1", "request_started", "2026-01-03T00:00:00Z")
        )
        storage.flush()

        summaries = storage.session_summaries(limit=3)
        assert [summary["session_id"] for summary in summaries] == [
            "session-new",
            "session-old",
            "session-dominant",
        ]
    finally:
        storage.close()


def test_flow_detail_decodes_gzip_response_body(tmp_path: Path) -> None:
    compressed = gzip.compress(b'{"ok":true}')
    descriptor = {
        "state": "captured",
        "size_bytes": str(len(compressed)),
        "encoding": "base64",
        "data": base64.b64encode(compressed).decode("ascii"),
        "content_type": "application/json",
    }
    async def scenario() -> None:
        server = ApiServer(ApiServerConfig(port=0, storage_path=tmp_path / "flows.sqlite"))
        server.application.ingest(
            {
                "protocol_version": "1",
                "type": "flow.metadata",
                "metadata": {
                    "flow_id": "flow-gzip",
                    "method": "GET",
                    "scheme": "https",
                    "host": "example.test",
                    "port": "443",
                    "path": "/gzip",
                    "request_headers": [],
                    "response_headers": [{"name": "content-encoding", "value": "gzip"}],
                    "response_status": "200",
                    "request_body": {"state": "empty", "size_bytes": "0"},
                    "response_body": descriptor,
                },
            }
        )
        server.application.ingest(
            {
                "protocol_version": "1",
                "type": "body.end",
                "flow_id": "flow-gzip",
                "body_side": "response",
                "total_bytes": str(len(compressed)),
                "body": descriptor,
            }
        )
        assert server.application.storage is not None
        server.application.storage.flush()
        await server.start()
        try:
            head, body = await request(server.bound_port, "/api/v1/flows/flow-gzip")
            assert head.startswith(b"HTTP/1.1 200 ")
            assert b"\x1f\x8b" not in body
            messages = json.loads(body)["messages"]
            body_end = next(message for message in messages if message["type"] == "body.end")
            assert base64.b64decode(body_end["body"]["data"]) == b'{"ok":true}'
            assert body_end["body"]["size_bytes"] == str(len(b'{"ok":true}'))
            assert body_end["total_bytes"] == str(len(b'{"ok":true}'))
        finally:
            await server.close()

    asyncio.run(scenario())

import asyncio
import base64
import json
from pathlib import Path

from mitm_inspector.api.server import ApiServer, ApiServerConfig


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
            assert sessions[0]["first_query"] == "first prompt"
            assert sessions[0]["models"] == ["claude-test"]

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

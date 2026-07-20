import asyncio
import json

import websockets

from mitm_inspector.api.app import ApiApplication
from mitm_inspector.api.server import ApiServer, ApiServerConfig
from mitm_inspector.store.memory import MemoryStore


def test_websocket_receives_coherent_frames_with_oversized_history() -> None:
    async def scenario() -> None:
        application = ApiApplication(
            MemoryStore(300),
            source_id="test-source",
            wall_clock=lambda: "2026-01-01T00:00:00Z",
        )
        for index in range(260):
            application.ingest(
                {
                    "protocol_version": "1",
                    "type": "flow.lifecycle",
                    "source_id": "test-source",
                    "flow_id": f"flow-{index:04}",
                    "event_id": f"event-{index:04}",
                    "occurred_at": "2026-01-01T00:00:00Z",
                    "sequence": str(index + 1),
                    "state": "request_started",
                }
            )
        server = ApiServer(
            ApiServerConfig(host="127.0.0.1", port=0, no_storage=True),
            application=application,
        )
        await server.start()
        try:
            async with websockets.connect(
                f"ws://127.0.0.1:{server.bound_port}/api/v1/stream"
            ) as websocket:
                messages = [
                    json.loads(await asyncio.wait_for(websocket.recv(), timeout=2))
                    for _ in range(3)
                ]
                assert [message["type"] for message in messages] == [
                    "source.hello",
                    "browser.resync",
                    "browser.snapshot",
                ]
                assert server.counters["subscribers"] == 1
        finally:
            await server.close()

    asyncio.run(scenario())

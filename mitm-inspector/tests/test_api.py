import asyncio
import base64
import gzip
import json
import os
import shutil
import stat as stat_module
import tempfile
import threading
from collections.abc import Callable, Coroutine, Iterator
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from mitm_inspector.api.app import ApiApplication
from mitm_inspector.api.httpwire import (
    Close,
    FrameDecoder,
    HttpWireError,
    Ping,
    TextMessage,
    WebSocketWireError,
    encode_close_frame,
    encode_frame,
    encode_pong_frame,
    encode_text_frame,
    is_loopback_host_header,
    is_loopback_origin,
    parse_request_head,
    websocket_accept_key,
)
from mitm_inspector.api.limits import (
    MAX_INGEST_BODY_PREFIX_BYTES,
    MAX_INGEST_LINE_BYTES,
    MAX_METADATA_HEADER_BYTES,
)
from mitm_inspector.api.projection import (
    collect_grid_flows,
    diff_grid_changes,
    grid_flow,
    redacted_body_descriptor,
)
from mitm_inspector.api.server import (
    SUBSCRIBER_QUEUE_FRAMES,
    ApiServer,
    ApiServerConfig,
    ApiServerError,
    config_from_argv,
    parse_ingest_line,
)
from mitm_inspector.api.server import (
    main as server_main,
)
from mitm_inspector.protocol import (
    MAX_U64,
    ProtocolError,
    parse_message,
)
from mitm_inspector.store.memory import MemoryStore
from mitm_inspector.store.sqlite import SearchCancelled

ROOT = Path(__file__).parents[1]
SCHEMA_PATH = ROOT / "contracts" / "protocol-v1.schema.json"
SECRET_BODY = b"request-secret-body-bytes"
SECRET_BODY_B64 = base64.b64encode(SECRET_BODY).decode("ascii")

_schema = json.loads(SCHEMA_PATH.read_text())
Draft202012Validator.check_schema(_schema)
SCHEMA_VALIDATOR = Draft202012Validator(_schema)


def assert_valid_wire_text(text: str) -> dict[str, object]:
    """Every emitted frame must be plain JSON, schema-valid, and parseable."""

    value = json.loads(text)
    assert isinstance(value, dict)
    SCHEMA_VALIDATOR.validate(value)
    parse_message(value)
    return value


def captured_body(data: bytes = SECRET_BODY) -> dict[str, object]:
    return {
        "state": "captured",
        "size_bytes": str(len(data)),
        "encoding": "base64",
        "data": base64.b64encode(data).decode("ascii"),
        "content_type": "application/json",
    }


def metadata_message(
    flow_id: str = "flow-1",
    *,
    request_body: dict[str, object] | None = None,
    path: str = "/v1/messages",
    session_id: str | None = None,
) -> dict[str, object]:
    return {
        "protocol_version": "1",
        "type": "flow.metadata",
        "metadata": {
            "flow_id": flow_id,
            "session_id": session_id,
            "method": "POST",
            "scheme": "https",
            "host": "api.example.test",
            "port": "443",
            "path": path,
            "request_headers": [{"name": "host", "value": "api.example.test"}],
            "request_body": request_body or {"state": "missing"},
        },
    }


def lifecycle_message(
    flow_id: str = "flow-1",
    *,
    state: str = "request_started",
    sequence: str = "1",
    source_id: str = "test-source",
) -> dict[str, object]:
    return {
        "protocol_version": "1",
        "type": "flow.lifecycle",
        "source_id": source_id,
        "flow_id": flow_id,
        "event_id": f"{flow_id}-{sequence}",
        "occurred_at": "2026-01-01T00:00:00Z",
        "sequence": sequence,
        "state": state,
    }


def body_end_message(flow_id: str = "flow-1") -> dict[str, object]:
    return {
        "protocol_version": "1",
        "type": "body.end",
        "flow_id": flow_id,
        "body_side": "request",
        "total_bytes": str(len(SECRET_BODY)),
        "body": captured_body(),
    }


def body_chunk_message(flow_id: str = "flow-1") -> dict[str, object]:
    return {
        "protocol_version": "1",
        "type": "body.chunk",
        "flow_id": flow_id,
        "body_side": "request",
        "chunk_index": "0",
        "offset_bytes": "0",
        "data_base64": SECRET_BODY_B64,
    }


def gap_message() -> dict[str, object]:
    return {
        "protocol_version": "1",
        "type": "stream.gap",
        "expected_sequence": "4",
        "actual_sequence": "9",
        "dropped_count": "4",
    }


def make_application(
    *,
    max_items: int = 16,
    max_age_seconds: float = 3600.0,
    clock: Callable[[], float] | None = None,
    cursor_start: int = 0,
) -> ApiApplication:
    store = MemoryStore(
        max_items,
        max_age_seconds=max_age_seconds,
        clock=clock or (lambda: 0.0),
    )
    return ApiApplication(
        store,
        source_id="test-source",
        max_body_prefix_bytes=1024,
        max_in_memory_bytes=1024 * 1024,
        wall_clock=lambda: "2026-01-01T00:00:00Z",
        cursor_start=cursor_start,
    )


class Collector:
    def __init__(self, *, accept: int | None = None) -> None:
        self.frames: list[str] = []
        self.accept = accept
        self.dropped = False

    def deliver(self, frame: str) -> bool:
        if self.accept is not None and len(self.frames) >= self.accept:
            return False
        self.frames.append(frame)
        return True

    def on_drop(self) -> None:
        self.dropped = True

    def messages(self) -> list[dict[str, object]]:
        return [assert_valid_wire_text(frame) for frame in self.frames]


# -- projection ------------------------------------------------------------


def test_redacted_descriptor_strips_captured_data_but_keeps_counts() -> None:
    redacted = redacted_body_descriptor(captured_body())
    assert redacted == {
        "state": "truncated",
        "size_bytes": str(len(SECRET_BODY)),
        "captured_bytes": "0",
        "encoding": "base64",
        "data": "",
        "content_type": "application/json",
    }


def test_redacted_descriptor_strips_truncated_data() -> None:
    truncated = {
        "state": "truncated",
        "size_bytes": "100",
        "captured_bytes": "10",
        "encoding": "base64",
        "data": base64.b64encode(b"0123456789").decode("ascii"),
    }
    redacted = redacted_body_descriptor(truncated)
    assert redacted == {
        "state": "truncated",
        "size_bytes": "100",
        "captured_bytes": "0",
        "encoding": "base64",
        "data": "",
    }


def test_redacted_descriptor_passes_missing_and_empty_through() -> None:
    assert redacted_body_descriptor({"state": "missing"}) == {"state": "missing"}
    empty = {"state": "empty", "size_bytes": "0", "content_type": "text/plain"}
    assert redacted_body_descriptor(empty) == empty
    assert redacted_body_descriptor("not-an-object") == "not-an-object"


def test_grid_flow_is_independent_schema_valid_and_body_free() -> None:
    message = metadata_message(request_body=captured_body())
    metadata = message["metadata"]
    assert isinstance(metadata, dict)
    flow = grid_flow(metadata)
    # The projected flow revalidates inside a snapshot envelope.
    assert_valid_wire_text(
        json.dumps(
            {
                "protocol_version": "1",
                "type": "browser.snapshot",
                "snapshot_id": "s-1",
                "cursor": "0",
                "flows": [flow],
            }
        )
    )
    assert SECRET_BODY_B64 not in json.dumps(flow)
    flow["method"] = "MUTATED"
    assert metadata["method"] == "POST"


def test_collect_grid_flows_uses_newest_metadata_per_flow_oldest_first() -> None:
    store = MemoryStore(8, clock=lambda: 0.0)
    store.append(parse_message(metadata_message("flow-a", path="/old")))
    store.append(parse_message(metadata_message("flow-b")))
    store.append(parse_message(metadata_message("flow-a", path="/new")))
    store.append(parse_message(lifecycle_message("flow-a")))
    flows = collect_grid_flows(store)
    assert list(flows) == ["flow-b", "flow-a"] or list(flows) == ["flow-a", "flow-b"]
    assert flows["flow-a"]["path"] == "/new"


def test_diff_grid_changes_orders_removes_then_upserts() -> None:
    published = {"a": {"flow_id": "a"}, "b": {"flow_id": "b"}}
    current = {"b": {"flow_id": "b", "path": "/x"}, "c": {"flow_id": "c"}}
    changes = diff_grid_changes(published, current)
    assert changes == [
        {"op": "remove", "flow_id": "a"},
        {"op": "upsert", "flow": {"flow_id": "b", "path": "/x"}},
        {"op": "upsert", "flow": {"flow_id": "c"}},
    ]
    assert diff_grid_changes(current, current) == []


# -- application: connect and stream ---------------------------------------


def test_subscribe_sends_hello_initial_resync_then_snapshot() -> None:
    application = make_application()
    collector = Collector()
    subscriber = application.subscribe(collector.deliver)
    assert not subscriber.closed
    messages = collector.messages()
    assert [message["type"] for message in messages] == [
        "source.hello",
        "browser.resync",
        "browser.snapshot",
    ]
    hello, resync, snapshot = messages
    assert hello["source_id"] == "test-source"
    assert hello["occurred_at"] == "2026-01-01T00:00:00Z"
    assert hello["limits"] == {
        "max_body_prefix_bytes": "1024",
        "max_in_memory_bytes": "1048576",
    }
    assert resync["reason"] == "initial_connect"
    assert resync["requested_cursor"] == "0"
    assert snapshot["cursor"] == "0"
    assert snapshot["flows"] == []


def test_subscribe_registers_after_initial_frames_before_history() -> None:
    application = make_application(max_items=8)
    application.ingest(lifecycle_message(flow_id="history-flow"))
    registration_counts: list[int] = []

    def deliver(frame: str) -> bool:
        del frame
        registration_counts.append(len(application._subscribers))
        return True

    subscriber = application.subscribe(deliver)

    assert registration_counts == [0, 0, 0, 1]
    assert application._subscribers == [subscriber]


def test_subscribe_replays_fitting_history_without_partial_counter() -> None:
    application = make_application(max_items=8)
    for index in range(2):
        application.ingest(
            lifecycle_message(flow_id=f"flow-{index}", sequence=str(index + 1))
        )
    collector = Collector()

    subscriber = application.subscribe(collector.deliver)

    assert not subscriber.closed
    assert application.subscriber_count == 1
    assert [message["type"] for message in collector.messages()] == [
        "source.hello",
        "browser.resync",
        "browser.snapshot",
        "flow.lifecycle",
        "flow.lifecycle",
    ]
    assert application.counters["subscribers_partial_history"] == 0


def test_subscribe_stops_on_historical_delivery_refusal_without_dropping_subscriber() -> None:
    history_count = 10
    application = make_application(max_items=history_count + 1)
    for index in range(history_count):
        application.ingest(
            lifecycle_message(flow_id=f"flow-{index:04}", sequence=str(index + 1))
        )
    accepted_historical = 7
    collector = Collector(accept=3 + accepted_historical)

    subscriber = application.subscribe(collector.deliver)

    assert not subscriber.closed
    assert subscriber in application._subscribers
    assert len(collector.frames) == 3 + accepted_historical
    assert application.counters["subscribers_partial_history"] == 1
    assert application.counters["dropped_subscribers"] == 0


def test_subscribe_retains_newest_history_when_replay_is_truncated() -> None:
    history_count = SUBSCRIBER_QUEUE_FRAMES - 2
    application = make_application(max_items=history_count + 1)
    for index in range(history_count):
        application.ingest(
            lifecycle_message(flow_id=f"flow-{index:04}", sequence=str(index + 1))
        )
    collector = Collector(accept=SUBSCRIBER_QUEUE_FRAMES)

    application.subscribe(collector.deliver)

    lifecycle_frames = [
        message for message in collector.messages() if message["type"] == "flow.lifecycle"
    ]
    assert len(lifecycle_frames) == SUBSCRIBER_QUEUE_FRAMES - 3
    assert lifecycle_frames[0]["flow_id"] == "flow-0001"
    assert lifecycle_frames[-1]["flow_id"] == f"flow-{history_count - 1:04}"


@pytest.mark.parametrize("failure_index", [0, 1, 2])
def test_subscribe_initial_frame_failure_drops_without_partial_counter(
    failure_index: int,
) -> None:
    application = make_application()
    refused = Collector(accept=failure_index)

    subscriber = application.subscribe(refused.deliver, on_drop=refused.on_drop)

    assert subscriber.closed
    assert refused.dropped
    assert application.subscriber_count == 0
    assert subscriber not in application._subscribers
    assert application.counters["dropped_subscribers"] == 1
    assert application.counters["subscribers_partial_history"] == 0


def test_ingest_metadata_emits_redacted_delta_with_incremented_cursor() -> None:
    application = make_application()
    collector = Collector()
    application.subscribe(collector.deliver)
    application.ingest(metadata_message(request_body=captured_body(), session_id="session-1"))
    delta = collector.messages()[-1]
    assert delta["type"] == "browser.delta"
    assert delta["cursor"] == "1"
    changes = delta["changes"]
    assert isinstance(changes, list) and len(changes) == 1
    assert changes[0]["op"] == "upsert"
    assert changes[0]["flow"]["session_id"] == "session-1"
    body = changes[0]["flow"]["request_body"]
    assert body["state"] == "truncated"
    assert body["captured_bytes"] == "0"
    assert body["data"] == ""
    assert SECRET_BODY_B64 not in collector.frames[-1]


def test_snapshot_reflects_ingested_flows_for_late_subscribers() -> None:
    application = make_application()
    application.ingest(metadata_message("flow-1"))
    application.ingest(metadata_message("flow-2"))
    collector = Collector()
    application.subscribe(collector.deliver)
    snapshot = collector.messages()[-1]
    assert snapshot["type"] == "browser.snapshot"
    assert snapshot["cursor"] == application.cursor
    flow_ids = [flow["flow_id"] for flow in snapshot["flows"]]
    assert sorted(flow_ids) == ["flow-1", "flow-2"]


def test_lifecycle_gap_and_unknown_messages_are_relayed_verbatim() -> None:
    application = make_application()
    collector = Collector()
    application.subscribe(collector.deliver)
    unknown = {
        "protocol_version": "1",
        "type": "vendor.custom",
        "flow_id": "flow-1",
        "additive": {"kept": True},
    }
    application.ingest(lifecycle_message())
    application.ingest(gap_message())
    application.ingest(unknown)
    relayed = collector.messages()[3:]
    assert [message["type"] for message in relayed] == [
        "flow.lifecycle",
        "stream.gap",
        "vendor.custom",
    ]
    assert relayed[0]["sequence"] == "1"
    assert relayed[1]["dropped_count"] == "4"
    assert relayed[2]["additive"] == {"kept": True}


def test_body_messages_are_retained_but_never_relayed_to_the_stream() -> None:
    application = make_application()
    collector = Collector()
    application.subscribe(collector.deliver)
    before = len(collector.frames)
    application.ingest(body_chunk_message())
    application.ingest(body_end_message())
    assert len(collector.frames) == before
    assert SECRET_BODY_B64 not in "".join(collector.frames)
    counters = application.counters
    store_counters = counters["store"]
    assert isinstance(store_counters, dict)
    assert store_counters["retained_messages"] == 2


def test_identical_metadata_reingest_does_not_emit_a_delta() -> None:
    application = make_application()
    collector = Collector()
    application.subscribe(collector.deliver)
    application.ingest(metadata_message())
    frames = len(collector.frames)
    application.ingest(metadata_message())
    assert len(collector.frames) == frames
    assert application.cursor == "1"


def test_completed_flow_eviction_emits_remove_changes() -> None:
    application = make_application(max_items=1)
    collector = Collector()
    application.subscribe(collector.deliver)
    for flow_id in ("flow-a", "flow-b"):
        application.ingest(metadata_message(flow_id))
        application.ingest(
            lifecycle_message(flow_id, state="flow_completed", sequence="9")
        )
    deltas = [
        message
        for message in collector.messages()
        if message["type"] == "browser.delta"
    ]
    all_changes = [change for delta in deltas for change in delta["changes"]]
    assert {"op": "remove", "flow_id": "flow-a"} in all_changes
    cursors = [int(str(delta["cursor"])) for delta in deltas]
    assert cursors == sorted(cursors)
    assert cursors == list(range(1, len(cursors) + 1))


def test_sweep_publishes_age_expiry_removals_while_idle() -> None:
    now = {"value": 0.0}
    application = make_application(max_age_seconds=10.0, clock=lambda: now["value"])
    collector = Collector()
    application.subscribe(collector.deliver)
    application.ingest(metadata_message("flow-old"))
    now["value"] = 100.0
    assert application.sweep() is True
    delta = collector.messages()[-1]
    assert delta["type"] == "browser.delta"
    assert delta["changes"] == [{"op": "remove", "flow_id": "flow-old"}]
    assert application.sweep() is False


def test_slow_subscriber_is_dropped_without_affecting_others() -> None:
    application = make_application()
    slow = Collector(accept=3)
    healthy = Collector()
    application.subscribe(slow.deliver, on_drop=slow.on_drop)
    application.subscribe(healthy.deliver)
    application.ingest(metadata_message("flow-1"))
    assert slow.dropped is True
    assert application.subscriber_count == 1
    application.ingest(metadata_message("flow-2"))
    healthy_deltas = [
        message
        for message in healthy.messages()
        if message["type"] == "browser.delta"
    ]
    assert len(healthy_deltas) == 2
    counters = application.counters
    assert counters["dropped_subscribers"] == 1


def test_raising_subscriber_is_dropped_not_propagated() -> None:
    application = make_application()

    def deliver(_frame: str) -> bool:
        raise RuntimeError("subscriber exploded")

    subscriber = application.subscribe(deliver)
    assert subscriber.closed
    assert application.subscriber_count == 0


def test_failed_connect_sequence_never_registers_the_subscriber() -> None:
    application = make_application()
    refused = Collector(accept=0)
    subscriber = application.subscribe(refused.deliver, on_drop=refused.on_drop)
    assert subscriber.closed
    assert refused.dropped is True
    assert application.subscriber_count == 0
    application.ingest(metadata_message())
    assert refused.frames == []


def test_handle_client_resync_returns_echo_then_snapshot() -> None:
    application = make_application()
    application.ingest(metadata_message())
    request = {
        "protocol_version": "1",
        "type": "browser.resync",
        "reason": "cursor_gap",
        "requested_cursor": "0",
    }
    frames = application.handle_client_message(request)
    echo, snapshot = (assert_valid_wire_text(frame) for frame in frames)
    assert echo["type"] == "browser.resync"
    assert echo["reason"] == "cursor_gap"
    assert echo["requested_cursor"] == "0"
    assert snapshot["type"] == "browser.snapshot"
    assert snapshot["cursor"] == "1"
    assert [flow["flow_id"] for flow in snapshot["flows"]] == ["flow-1"]


@pytest.mark.parametrize(
    "value",
    [
        "not-json-object",
        {"protocol_version": "1", "type": "flow.lifecycle"},
        lifecycle_message(),
        {"protocol_version": "2", "type": "browser.resync"},
        {
            "protocol_version": "1",
            "type": "browser.resync",
            "reason": "cursor_gap",
            "requested_cursor": "not-a-number",
        },
    ],
)
def test_invalid_client_messages_are_rejected(value: object) -> None:
    application = make_application()
    with pytest.raises(ProtocolError):
        application.handle_client_message(value)


def test_rejected_ingest_leaves_store_and_stream_untouched() -> None:
    application = make_application()
    collector = Collector()
    application.subscribe(collector.deliver)
    frames = len(collector.frames)
    bad_metadata = metadata_message()
    metadata = bad_metadata["metadata"]
    assert isinstance(metadata, dict)
    metadata["port"] = "not-a-port"
    with pytest.raises(ProtocolError):
        application.ingest(bad_metadata)
    assert len(collector.frames) == frames
    counters = application.counters
    assert counters["ingested_messages"] == 0
    store_counters = counters["store"]
    assert isinstance(store_counters, dict)
    assert store_counters["retained_messages"] == 0


def test_cursor_exhaustion_is_terminal_and_drops_streams() -> None:
    application = make_application(cursor_start=MAX_U64)
    collector = Collector()
    application.subscribe(collector.deliver, on_drop=collector.on_drop)
    snapshot = collector.messages()[-1]
    assert snapshot["cursor"] == str(MAX_U64)
    application.ingest(metadata_message())
    assert collector.dropped is True
    assert application.subscriber_count == 0
    assert application.counters["cursor_exhausted"] is True
    # Reconnects still receive a coherent snapshot at the final cursor.
    late = Collector()
    application.subscribe(late.deliver)
    late_snapshot = late.messages()[-1]
    assert late_snapshot["cursor"] == str(MAX_U64)
    assert [flow["flow_id"] for flow in late_snapshot["flows"]] == ["flow-1"]


def test_ingest_input_aliases_cannot_mutate_retained_state() -> None:
    application = make_application()
    message = metadata_message()
    application.ingest(message)
    metadata = message["metadata"]
    assert isinstance(metadata, dict)
    metadata["method"] = "MUTATED"
    snapshot = assert_valid_wire_text(application.snapshot_text())
    assert snapshot["flows"][0]["method"] == "POST"


def test_flow_detail_returns_full_messages_oldest_first() -> None:
    application = make_application()
    application.ingest(metadata_message(request_body=captured_body()))
    application.ingest(body_chunk_message())
    application.ingest(lifecycle_message())
    application.ingest(body_end_message())
    application.ingest(metadata_message("other-flow"))
    detail_text = application.flow_detail_text("flow-1")
    assert detail_text is not None
    detail = json.loads(detail_text)
    assert detail["protocol_version"] == "1"
    assert detail["flow_id"] == "flow-1"
    types = [message["type"] for message in detail["messages"]]
    assert types == ["flow.metadata", "body.chunk", "flow.lifecycle", "body.end"]
    for message in detail["messages"]:
        SCHEMA_VALIDATOR.validate(message)
    # Selection is the only surface that carries body bytes.
    assert SECRET_BODY_B64 in detail_text
    assert application.flow_detail_text("missing-flow") is None
    assert application.flow_detail_text("") is None


# -- http wire -------------------------------------------------------------


def test_websocket_accept_key_matches_rfc_6455_vector() -> None:
    assert (
        websocket_accept_key("dGhlIHNhbXBsZSBub25jZQ==")
        == "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="
    )
    with pytest.raises(WebSocketWireError):
        websocket_accept_key("dG9vLXNob3J0")
    with pytest.raises(WebSocketWireError):
        websocket_accept_key("not base64!!")


def test_parse_request_head_accepts_a_normal_get() -> None:
    head = parse_request_head(
        b"GET /api/v1/health HTTP/1.1\r\nHost: 127.0.0.1:8000\r\n"
        b"Accept: application/json\r\n\r\n"
    )
    assert head.method == "GET"
    assert head.target == "/api/v1/health"
    assert head.single_header("host") == "127.0.0.1:8000"
    assert head.token_list("accept") == ("application/json",)


@pytest.mark.parametrize(
    "raw",
    [
        b"GET /\r\n\r\n",
        b"GET / HTTP/2\r\nHost: a\r\n\r\n",
        b"G@T / HTTP/1.1\r\nHost: a\r\n\r\n",
        b"GET /\x7f HTTP/1.1\r\nHost: a\r\n\r\n",
        b"GET / two words HTTP/1.1\r\nHost: a\r\n\r\n",
        b"GET / HTTP/1.1\r\nHost: a\r\n folded\r\n\r\n",
        b"GET / HTTP/1.1\r\nBad Header: a\r\n\r\n",
        b"GET / HTTP/1.1\r\nHost a\r\n\r\n",
        b"GET / HTTP/1.1\nHost: a\n\n",
        b"GET / HTTP/1.1\r\nHost: a\rb\r\n\r\n",
        "GET /é HTTP/1.1\r\nHost: a\r\n\r\n".encode(),
        b"GET / HTTP/1.1\r\nHost: a\r\n" + b"X-N: v\r\n" * 101 + b"\r\n",
    ],
)
def test_parse_request_head_rejects_malformed_input(raw: bytes) -> None:
    with pytest.raises(HttpWireError):
        parse_request_head(raw)


def test_duplicate_singleton_headers_are_rejected() -> None:
    head = parse_request_head(
        b"GET / HTTP/1.1\r\nHost: a\r\nOrigin: x\r\nOrigin: y\r\n\r\n"
    )
    with pytest.raises(HttpWireError):
        head.single_header("origin")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("127.0.0.1:8000", True),
        ("localhost", True),
        ("LOCALHOST:9", True),
        ("[::1]:9000", True),
        ("127.9.9.9", True),
        ("evil.example", False),
        ("127.0.0.1:0", False),
        ("127.0.0.1:", False),
        ("user@127.0.0.1", False),
        ("127.0.0.1/path", False),
        ("10.0.0.1:8000", False),
        ("", False),
    ],
)
def test_loopback_host_header_validation(value: str, expected: bool) -> None:
    assert is_loopback_host_header(value) is expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("http://127.0.0.1:5173", True),
        ("http://localhost:8000", True),
        ("https://[::1]", True),
        ("http://evil.example", False),
        ("null", False),
        ("file://localhost", False),
        ("http://127.0.0.1/path", False),
        ("http://user:pw@127.0.0.1", False),
        ("http://10.1.2.3:8000", False),
        ("", False),
    ],
)
def test_loopback_origin_validation(value: str, expected: bool) -> None:
    assert is_loopback_origin(value) is expected


def mask_frame(opcode: int, payload: bytes, *, fin: bool = True) -> bytes:
    mask = b"\x01\x02\x03\x04"
    header = bytearray([(0x80 if fin else 0x00) | opcode])
    length = len(payload)
    if length <= 125:
        header.append(0x80 | length)
    elif length <= 0xFFFF:
        header.append(0x80 | 126)
        header.extend(length.to_bytes(2, "big"))
    else:
        header.append(0x80 | 127)
        header.extend(length.to_bytes(8, "big"))
    masked = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
    return bytes(header) + mask + masked


def test_frame_decoder_handles_text_ping_pong_and_close() -> None:
    decoder = FrameDecoder()
    data = (
        mask_frame(0x1, b"hello")
        + mask_frame(0x9, b"ping-payload")
        + mask_frame(0xA, b"pong-payload")
        + mask_frame(0x8, (1000).to_bytes(2, "big") + b"bye")
    )
    events = decoder.feed(data)
    assert events[0] == TextMessage("hello")
    assert events[1] == Ping(b"ping-payload")
    assert events[3] == Close(1000)


def test_frame_decoder_reassembles_fragmented_text_across_feeds() -> None:
    decoder = FrameDecoder()
    first = mask_frame(0x1, b"hel", fin=False)
    middle = mask_frame(0x0, b"lo ", fin=False)
    last = mask_frame(0x0, b"world")
    assert decoder.feed(first) == []
    assert decoder.feed(middle[:2]) == []
    events = decoder.feed(middle[2:] + last)
    assert events == [TextMessage("hello world")]


@pytest.mark.parametrize(
    "data",
    [
        encode_frame(0x1, b"unmasked"),
        mask_frame(0x3, b"reserved-opcode"),
        mask_frame(0x2, b"binary"),
        mask_frame(0x9, b"x" * 126),
        mask_frame(0x9, b"ping", fin=False),
        mask_frame(0x8, b"\x03"),
        mask_frame(0x0, b"no-open-fragment"),
        mask_frame(0x1, b"\xff\xfe"),
        bytes([0xC1]) + mask_frame(0x1, b"rsv")[1:],
        mask_frame(0x1, b"x" * (64 * 1024 + 1)),
    ],
)
def test_frame_decoder_rejects_protocol_violations(data: bytes) -> None:
    decoder = FrameDecoder()
    with pytest.raises(WebSocketWireError):
        decoder.feed(data)


def test_frame_decoder_rejects_interleaved_data_frames() -> None:
    decoder = FrameDecoder()
    decoder.feed(mask_frame(0x1, b"open", fin=False))
    with pytest.raises(WebSocketWireError):
        decoder.feed(mask_frame(0x1, b"interleaved"))


def test_encode_frame_produces_extended_length_headers() -> None:
    medium = encode_frame(0x1, b"x" * 200)
    assert medium[1] == 126
    assert int.from_bytes(medium[2:4], "big") == 200
    large = encode_frame(0x1, b"x" * 70000)
    assert large[1] == 127
    assert int.from_bytes(large[2:10], "big") == 70000
    assert encode_text_frame("hi")[0] == 0x81
    assert encode_close_frame(1002)[0] == 0x88
    assert encode_pong_frame(b"p")[0] == 0x8A


def test_parse_ingest_line_is_strict_about_duplicates_and_constants() -> None:
    assert parse_ingest_line(b'{"a": 1}\n') == {"a": 1}
    with pytest.raises(ValueError):
        parse_ingest_line(b'{"a": 1, "a": 2}\n')
    with pytest.raises(ValueError):
        parse_ingest_line(b'{"a": NaN}\n')
    with pytest.raises(ValueError):
        parse_ingest_line(b"\xff\xfe\n")


# -- server configuration --------------------------------------------------


def test_config_from_argv_accepts_the_reserved_runtime_contract(
    tmp_path: Path,
) -> None:
    socket_path = tmp_path / "capture.sock"
    config = config_from_argv(
        [
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
            "--proxy-port",
            "8080",
            "--max-retained-flows",
            "2000",
            "--retention-seconds",
            "1800",
            "--max-body-bytes",
            "134217728",
            "--max-body-prefix-bytes",
            "1048576",
            "--capture-socket",
            str(socket_path),
            "--capture-source-id",
            "mitm-inspector",
            "--capture-max-body-prefix-bytes",
            "1048576",
            "--capture-max-in-memory-bytes",
            "134217728",
            "--capture-max-pending-messages",
            "4096",
        ]
    )
    assert config.host == "127.0.0.1"
    assert config.capture_socket == socket_path
    assert config.source_id == "mitm-inspector"
    assert config.max_body_bytes == 134217728
    assert config.max_body_prefix_bytes == 1048576


@pytest.mark.parametrize(
    "kwargs",
    [
        {"host": "0.0.0.0"},
        {"host": "example.com"},
        {"port": 70000},
        {"proxy_port": 0},
        {"source_id": ""},
        {"max_retained_flows": 0},
        {"retention_seconds": -1},
        {"max_body_prefix_bytes": 10, "max_body_bytes": 5},
        {"capture_socket": Path("relative/capture.sock")},
        {"sweep_interval_seconds": 0},
    ],
)
def test_server_config_rejects_unsafe_settings(kwargs: dict[str, object]) -> None:
    with pytest.raises(ApiServerError):
        ApiServerConfig(**kwargs)  # type: ignore[arg-type]


def test_server_main_reports_configuration_errors() -> None:
    assert server_main(["--host", "0.0.0.0"]) == 2


# -- server integration ----------------------------------------------------


def run_async(factory: Callable[[], Coroutine[Any, Any, None]]) -> None:
    async def bounded() -> None:
        await asyncio.wait_for(factory(), timeout=20)

    asyncio.run(bounded())


@pytest.fixture
def socket_dir() -> Iterator[Path]:
    # pytest's tmp_path exceeds the AF_UNIX path limit on macOS.
    path = Path(tempfile.mkdtemp(prefix="mitm-b3-", dir="/tmp"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def make_server(tmp_path: Path, *, with_ingest: bool = True) -> ApiServer:
    config = ApiServerConfig(
        host="127.0.0.1",
        port=0,
        source_id="test-source",
        capture_socket=(tmp_path / "capture.sock") if with_ingest else None,
        sweep_interval_seconds=0.05,
    )
    return ApiServer(config)


async def wait_until(condition: Callable[[], bool]) -> None:
    for _ in range(400):
        if condition():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition was not reached in time")


async def http_request(port: int, request: bytes) -> bytes:
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(request)
    await writer.drain()
    response = await reader.read()
    writer.close()
    await writer.wait_closed()
    return response


def get(path: str, port: int, *, host: str | None = None) -> bytes:
    authority = host if host is not None else f"127.0.0.1:{port}"
    return (
        f"GET {path} HTTP/1.1\r\nHost: {authority}\r\nConnection: close\r\n\r\n"
    ).encode()


def response_body(response: bytes) -> bytes:
    head, _, body = response.partition(b"\r\n\r\n")
    assert b"Content-Length" in head
    return body


async def send_ingest_lines(socket_path: Path, lines: list[bytes]) -> None:
    reader, writer = await asyncio.open_unix_connection(str(socket_path))
    for line in lines:
        writer.write(line)
    await writer.drain()
    writer.close()
    await writer.wait_closed()
    del reader


def test_http_health_root_snapshot_and_errors(tmp_path: Path) -> None:
    async def scenario() -> None:
        server = make_server(tmp_path, with_ingest=False)
        await server.start()
        try:
            port = server.bound_port
            health = await http_request(port, get("/api/v1/health", port))
            assert health.startswith(b"HTTP/1.1 200 ")
            payload = json.loads(response_body(health))
            assert payload["status"] == "ok"
            assert payload["counters"]["cursor"] == "0"

            root = await http_request(port, get("/", port))
            assert root.startswith(b"HTTP/1.1 200 ")
            assert b"nosniff" in root

            snapshot = await http_request(port, get("/api/v1/snapshot", port))
            message = assert_valid_wire_text(response_body(snapshot).decode())
            assert message["type"] == "browser.snapshot"

            counters = await http_request(port, get("/api/v1/counters", port))
            counters_head, _, _ = counters.partition(b"\r\n\r\n")
            assert counters.startswith(b"HTTP/1.1 200 ")
            assert b"Content-Type: application/json" in counters_head
            counters_payload = json.loads(response_body(counters))
            assert "subscribers_partial_history" in counters_payload
            assert "dropped_subscribers" in counters_payload

            missing = await http_request(port, get("/api/v1/nope", port))
            assert missing.startswith(b"HTTP/1.1 404 ")

            post = await http_request(
                port,
                b"POST /api/v1/health HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n",
            )
            assert post.startswith(b"HTTP/1.1 405 ")

            rebind = await http_request(
                port, get("/api/v1/health", port, host="evil.example")
            )
            assert rebind.startswith(b"HTTP/1.1 400 ")

            with_body = await http_request(
                port,
                b"GET / HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: 3\r\n\r\nabc",
            )
            assert with_body.startswith(b"HTTP/1.1 400 ")

            malformed = await http_request(port, b"NOT A REQUEST\r\n\r\n")
            assert malformed.startswith(b"HTTP/1.1 400 ")
        finally:
            await server.close()

    run_async(scenario)


def test_ingest_socket_feeds_the_store_and_rejects_bad_lines(
    socket_dir: Path,
) -> None:
    async def scenario() -> None:
        server = make_server(socket_dir)
        await server.start()
        socket_path = server.config.capture_socket
        assert socket_path is not None
        assert stat_is_socket(socket_path)
        try:
            good = json.dumps(metadata_message()).encode() + b"\n"
            lifecycle = json.dumps(lifecycle_message()).encode() + b"\n"
            await send_ingest_lines(socket_path, [good, lifecycle])
            await wait_until(
                lambda: server.application.counters["ingested_messages"] == 2
            )
            snapshot = assert_valid_wire_text(server.application.snapshot_text())
            assert [flow["flow_id"] for flow in snapshot["flows"]] == ["flow-1"]

            # A malformed line drops the producer connection and is counted.
            reader, writer = await asyncio.open_unix_connection(str(socket_path))
            writer.write(b"this is not json\n")
            await writer.drain()
            assert await reader.read() == b""
            writer.close()
            await writer.wait_closed()
            await wait_until(lambda: server.counters["rejected_ingest_lines"] == 1)

            # Invalid protocol messages also fail closed.
            bad_message = json.dumps({"protocol_version": "2", "type": "x"}).encode()
            reader, writer = await asyncio.open_unix_connection(str(socket_path))
            writer.write(bad_message + b"\n")
            await writer.drain()
            assert await reader.read() == b""
            writer.close()
            await writer.wait_closed()
            await wait_until(lambda: server.counters["rejected_ingest_lines"] == 2)

            # The listener keeps serving new producers after rejections.
            await send_ingest_lines(
                socket_path, [json.dumps(metadata_message("flow-2")).encode() + b"\n"]
            )
            await wait_until(
                lambda: server.application.counters["ingested_messages"] == 3
            )
        finally:
            await server.close()

    run_async(scenario)


def stat_is_socket(path: Path) -> bool:
    mode = os.lstat(path).st_mode
    return stat_module.S_ISSOCK(mode) and (mode & 0o777) == 0o600


def test_ingest_socket_refuses_non_socket_path(socket_dir: Path) -> None:
    async def scenario() -> None:
        socket_path = socket_dir / "capture.sock"
        socket_path.write_text("not a socket")
        server = ApiServer(
            ApiServerConfig(host="127.0.0.1", port=0, capture_socket=socket_path)
        )
        with pytest.raises(ApiServerError):
            await server.start()
        await server.close()

    run_async(scenario)


def test_ingest_socket_replaces_a_stale_socket_endpoint(socket_dir: Path) -> None:
    async def scenario() -> None:
        socket_path = socket_dir / "capture.sock"
        stale = await asyncio.start_unix_server(
            lambda reader, writer: None, path=str(socket_path)
        )
        stale.close()
        await stale.wait_closed()
        server = ApiServer(
            ApiServerConfig(host="127.0.0.1", port=0, capture_socket=socket_path)
        )
        await server.start()
        try:
            await send_ingest_lines(
                socket_path, [json.dumps(metadata_message()).encode() + b"\n"]
            )
            await wait_until(
                lambda: server.application.counters["ingested_messages"] == 1
            )
        finally:
            await server.close()

    run_async(scenario)


class WsClient:
    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        self.reader = reader
        self.writer = writer
        self._buffer = bytearray()

    @classmethod
    async def connect(
        cls,
        port: int,
        *,
        origin: str | None = "http://127.0.0.1:5173",
        host: str | None = None,
        extra: str = "",
    ) -> tuple["WsClient | None", bytes]:
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        authority = host if host is not None else f"127.0.0.1:{port}"
        origin_line = f"Origin: {origin}\r\n" if origin is not None else ""
        request = (
            f"GET /api/v1/stream HTTP/1.1\r\n"
            f"Host: {authority}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            f"{origin_line}{extra}\r\n"
        ).encode()
        writer.write(request)
        await writer.drain()
        head = await reader.readuntil(b"\r\n\r\n")
        if not head.startswith(b"HTTP/1.1 101 "):
            body = await reader.read()
            writer.close()
            await writer.wait_closed()
            return None, head + body
        assert b"Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=" in head
        return cls(reader, writer), head

    async def read_frame(self) -> tuple[int, bytes]:
        while True:
            frame = self._try_decode()
            if frame is not None:
                return frame
            data = await self.reader.read(4096)
            if not data:
                raise AssertionError("server closed the stream mid-frame")
            self._buffer.extend(data)

    def _try_decode(self) -> tuple[int, bytes] | None:
        buffer = self._buffer
        if len(buffer) < 2:
            return None
        opcode = buffer[0] & 0x0F
        assert not buffer[1] & 0x80, "server frames must be unmasked"
        length = buffer[1] & 0x7F
        offset = 2
        if length == 126:
            if len(buffer) < 4:
                return None
            length = int.from_bytes(buffer[2:4], "big")
            offset = 4
        elif length == 127:
            if len(buffer) < 10:
                return None
            length = int.from_bytes(buffer[2:10], "big")
            offset = 10
        if len(buffer) < offset + length:
            return None
        payload = bytes(buffer[offset : offset + length])
        del buffer[: offset + length]
        return opcode, payload

    async def read_message(self) -> dict[str, object]:
        opcode, payload = await self.read_frame()
        assert opcode == 0x1
        return assert_valid_wire_text(payload.decode())

    async def send_raw(self, data: bytes) -> None:
        self.writer.write(data)
        await self.writer.drain()

    async def close(self) -> None:
        self.writer.close()
        try:
            await self.writer.wait_closed()
        except ConnectionError:
            pass


def test_websocket_session_streams_snapshot_deltas_and_resync(
    socket_dir: Path,
) -> None:
    async def scenario() -> None:
        server = make_server(socket_dir)
        await server.start()
        socket_path = server.config.capture_socket
        assert socket_path is not None
        try:
            client, head = await WsClient.connect(server.bound_port)
            assert client is not None
            hello = await client.read_message()
            resync = await client.read_message()
            snapshot = await client.read_message()
            assert hello["type"] == "source.hello"
            assert resync["reason"] == "initial_connect"
            assert snapshot["type"] == "browser.snapshot"
            assert snapshot["cursor"] == "0"

            await send_ingest_lines(
                socket_path,
                [
                    json.dumps(
                        metadata_message(request_body=captured_body())
                    ).encode()
                    + b"\n",
                    json.dumps(lifecycle_message()).encode() + b"\n",
                    json.dumps(gap_message()).encode() + b"\n",
                ],
            )
            delta = await client.read_message()
            assert delta["type"] == "browser.delta"
            assert delta["cursor"] == "1"
            changes = delta["changes"]
            assert isinstance(changes, list)
            assert changes[0]["flow"]["request_body"]["data"] == ""
            lifecycle = await client.read_message()
            assert lifecycle["type"] == "flow.lifecycle"
            timing_delta = await client.read_message()
            assert timing_delta["type"] == "browser.delta"
            assert timing_delta["changes"][0]["flow"]["started_at"] == "2026-01-01T00:00:00Z"
            gap = await client.read_message()
            assert gap["type"] == "stream.gap"

            # Ping is answered with a matching pong.
            await client.send_raw(mask_frame(0x9, b"probe"))
            opcode, payload = await client.read_frame()
            assert (opcode, payload) == (0xA, b"probe")

            # A client resync request receives an echo and a fresh snapshot.
            request = {
                "protocol_version": "1",
                "type": "browser.resync",
                "reason": "cursor_gap",
                "requested_cursor": "1",
            }
            await client.send_raw(mask_frame(0x1, json.dumps(request).encode()))
            echo = await client.read_message()
            fresh = await client.read_message()
            assert echo["type"] == "browser.resync"
            assert echo["requested_cursor"] == "1"
            assert fresh["type"] == "browser.snapshot"
            assert fresh["cursor"] == "2"

            # A clean client close is answered with a close frame.
            await client.send_raw(mask_frame(0x8, (1000).to_bytes(2, "big")))
            opcode, _payload = await client.read_frame()
            assert opcode == 0x8
            await client.close()
            await wait_until(lambda: server.application.subscriber_count == 0)
        finally:
            await server.close()

    run_async(scenario)


def test_websocket_rejects_cross_site_and_malformed_handshakes(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        server = make_server(tmp_path, with_ingest=False)
        await server.start()
        try:
            port = server.bound_port
            client, response = await WsClient.connect(
                port, origin="http://evil.example"
            )
            assert client is None
            assert response.startswith(b"HTTP/1.1 403 ")

            client, response = await WsClient.connect(port, host="evil.example")
            assert client is None
            assert response.startswith(b"HTTP/1.1 400 ")

            missing_key = await http_request(
                port,
                (
                    f"GET /api/v1/stream HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
                    "Upgrade: websocket\r\nConnection: Upgrade\r\n"
                    "Sec-WebSocket-Version: 13\r\n\r\n"
                ).encode(),
            )
            assert missing_key.startswith(b"HTTP/1.1 400 ")

            # Absent Origin (non-browser client) is accepted.
            client, _head = await WsClient.connect(port, origin=None)
            assert client is not None
            hello = await client.read_message()
            assert hello["type"] == "source.hello"
            await client.close()
            assert server.application.subscriber_count <= 1
        finally:
            await server.close()

    run_async(scenario)


def test_websocket_closes_on_protocol_violations(tmp_path: Path) -> None:
    async def scenario() -> None:
        server = make_server(tmp_path, with_ingest=False)
        await server.start()
        try:
            port = server.bound_port
            client, _head = await WsClient.connect(port)
            assert client is not None
            for _ in range(3):
                await client.read_message()
            await client.send_raw(encode_frame(0x1, b"unmasked-client-frame"))
            opcode, payload = await drain_to_close(client)
            assert opcode == 0x8
            assert int.from_bytes(payload[:2], "big") == 1002
            await client.close()

            client, _head = await WsClient.connect(port)
            assert client is not None
            for _ in range(3):
                await client.read_message()
            await client.send_raw(mask_frame(0x1, b"this is not json"))
            opcode, payload = await drain_to_close(client)
            assert opcode == 0x8
            assert int.from_bytes(payload[:2], "big") == 1002
            await client.close()
            await wait_until(lambda: server.application.subscriber_count == 0)
        finally:
            await server.close()

    run_async(scenario)


async def drain_to_close(client: WsClient) -> tuple[int, bytes]:
    while True:
        opcode, payload = await client.read_frame()
        if opcode == 0x8:
            return opcode, payload


def test_flow_detail_endpoint_serves_bodies_only_on_selection(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        server = make_server(tmp_path, with_ingest=False)
        await server.start()
        try:
            port = server.bound_port
            server.application.ingest(metadata_message(request_body=captured_body()))
            server.application.ingest(body_end_message())

            snapshot = await http_request(port, get("/api/v1/snapshot", port))
            assert SECRET_BODY_B64.encode() not in snapshot

            detail = await http_request(port, get("/api/v1/flows/flow-1", port))
            assert detail.startswith(b"HTTP/1.1 200 ")
            assert SECRET_BODY_B64.encode() in detail
            payload = json.loads(response_body(detail))
            assert payload["flow_id"] == "flow-1"

            missing = await http_request(port, get("/api/v1/flows/absent", port))
            assert missing.startswith(b"HTTP/1.1 404 ")
            traversal = await http_request(
                port, get("/api/v1/flows/../../etc/passwd", port)
            )
            assert traversal.startswith(b"HTTP/1.1 404 ")
        finally:
            await server.close()

    run_async(scenario)


def test_flow_detail_decodes_gzip_and_surfaces_original_encoding(tmp_path: Path) -> None:
    async def scenario() -> None:
        server = make_server(tmp_path, with_ingest=False)
        await server.start()
        try:
            response_text = b'{"stop_reason":"end_turn","usage":{"output_tokens":5}}'
            compressed = gzip.compress(response_text)
            metadata = metadata_message()
            metadata_value = metadata["metadata"]
            assert isinstance(metadata_value, dict)
            metadata_value.update(
                {
                    "host": "api.anthropic.com",
                    "response_headers": [
                        {"name": "content-type", "value": "application/json"},
                        {"name": "content-encoding", "value": "gzip"},
                    ],
                    "response_status": "200",
                    "response_body": captured_body(compressed),
                }
            )
            server.application.ingest(metadata)
            server.application.ingest(
                {
                    "protocol_version": "1",
                    "type": "body.end",
                    "flow_id": "flow-1",
                    "body_side": "response",
                    "total_bytes": str(len(compressed)),
                    "body": captured_body(compressed),
                }
            )

            snapshot_response = await http_request(
                server.bound_port, get("/api/v1/snapshot", server.bound_port)
            )
            snapshot = json.loads(response_body(snapshot_response))
            flow = snapshot["flows"][0]
            assert flow["response_body_size"] == str(len(compressed))
            assert flow["content_encoding"] == {"response": "gzip"}

            detail_response = await http_request(
                server.bound_port, get("/api/v1/flows/flow-1", server.bound_port)
            )
            detail = json.loads(response_body(detail_response))
            decoded_bodies = [
                base64.b64decode(message["body"]["data"])
                for message in detail["messages"]
                if message["type"] == "body.end"
            ]
            assert decoded_bodies == [response_text]
            detail_metadata = next(
                message["metadata"]
                for message in detail["messages"]
                if message["type"] == "flow.metadata"
            )
            assert detail_metadata["content_encoding"] == {"response": "gzip"}
            assert base64.b64decode(detail_metadata["response_body"]["data"]) == response_text
        finally:
            await server.close()

    run_async(scenario)


def test_search_endpoint_queries_durable_decoded_bodies_and_validates_input(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        config = ApiServerConfig(
            host="127.0.0.1",
            port=0,
            max_retained_flows=1,
            storage_path=tmp_path / "flows.sqlite",
            sweep_interval_seconds=0.05,
        )
        server = ApiServer(config)
        await server.start()
        try:
            first = metadata_message("flow-a", request_body=captured_body(b'{"text":"Needle one"}'))
            first_value = first["metadata"]
            assert isinstance(first_value, dict)
            compressed = gzip.compress(b"response needle match")
            first_value.update(
                {
                    "response_headers": [{"name": "content-encoding", "value": "gzip"}],
                    "response_status": "200",
                    "response_body": captured_body(compressed),
                }
            )
            second = metadata_message(
                "flow-b", request_body=captured_body(b'{"text":"needle two"}')
            )
            server.application.ingest(first)
            server.application.ingest(
                lifecycle_message("flow-a", state="flow_completed", sequence="1")
            )
            server.application.ingest(second)
            server.application.ingest(
                lifecycle_message("flow-b", state="flow_completed", sequence="2")
            )

            limited_response = await http_request(
                server.bound_port, get("/api/v1/search?q=needle&limit=1", server.bound_port)
            )
            limited = json.loads(response_body(limited_response))
            assert limited["truncated"] is True
            assert len(limited["matches"]) == 1
            limited_match = limited["matches"][0]
            assert {
                key: limited_match[key] for key in ("flow_id", "field", "snippet")
            } == {
                "flow_id": "flow-b",
                "field": "request_body",
                "snippet": '{"text":"needle two"}',
            }
            assert limited_match["flow"]["flow_id"] == "flow-b"
            assert limited_match["flow"]["request_body"]["data"] == ""

            all_response = await http_request(
                server.bound_port, get("/api/v1/search?q=needle", server.bound_port)
            )
            all_matches = json.loads(response_body(all_response))["matches"]
            assert [(match["flow_id"], match["field"]) for match in all_matches] == [
                ("flow-b", "request_body"),
                ("flow-a", "request_body"),
                ("flow-a", "response_body"),
            ]
            snapshot_response = await http_request(
                server.bound_port, get("/api/v1/snapshot", server.bound_port)
            )
            snapshot_flow_ids = {
                flow["flow_id"]
                for flow in json.loads(response_body(snapshot_response))["flows"]
            }
            assert "flow-a" not in snapshot_flow_ids
            older_match = next(match for match in all_matches if match["flow_id"] == "flow-a")
            assert older_match["flow"]["flow_id"] == "flow-a"
            assert older_match["flow"]["request_body"]["data"] == ""

            no_match_response = await http_request(
                server.bound_port, get("/api/v1/search?q=absent", server.bound_port)
            )
            assert json.loads(response_body(no_match_response)) == {
                "matches": [],
                "truncated": False,
            }
            for path in ("/api/v1/search", "/api/v1/search?q="):
                invalid = await http_request(server.bound_port, get(path, server.bound_port))
                assert invalid.startswith(b"HTTP/1.1 400 ")
        finally:
            await server.close()

    run_async(scenario)


def test_detail_timing_uses_the_same_lifecycle_sequence_reducer_as_grid() -> None:
    application = make_application(max_items=32)
    application.ingest(metadata_message("flow-timing"))
    terminal_nine = lifecycle_message(
        "flow-timing", state="flow_completed", sequence="9"
    )
    terminal_nine["occurred_at"] = "2026-01-01T12:09:00Z"
    terminal_five = lifecycle_message(
        "flow-timing", state="flow_completed", sequence="5"
    )
    terminal_five["occurred_at"] = "2026-01-01T12:05:00Z"
    application.ingest(terminal_nine)
    application.ingest(terminal_five)

    snapshot = json.loads(application.snapshot_text())
    detail_text = application.flow_detail_text("flow-timing")
    assert detail_text is not None
    detail = json.loads(detail_text)
    detail_metadata = next(
        message["metadata"]
        for message in detail["messages"]
        if message["type"] == "flow.metadata"
    )
    assert snapshot["flows"][0]["ended_at"] == "2026-01-01T12:09:00Z"
    assert detail_metadata["ended_at"] == "2026-01-01T12:09:00Z"


def test_detail_preserves_truncated_compressed_body_total() -> None:
    application = make_application(max_items=32)
    compressed = gzip.compress(b"decoded" * 100_000)
    prefix = compressed[: len(compressed) // 2]
    truncated = {
        "state": "truncated",
        "size_bytes": str(len(compressed)),
        "captured_bytes": str(len(prefix)),
        "encoding": "base64",
        "data": base64.b64encode(prefix).decode("ascii"),
        "content_type": "application/json",
    }
    message = metadata_message("flow-truncated")
    metadata = message["metadata"]
    assert isinstance(metadata, dict)
    metadata.update(
        {
            "response_headers": [{"name": "content-encoding", "value": "gzip"}],
            "response_status": "200",
            "response_body": truncated,
        }
    )
    application.ingest(message)
    application.ingest(
        {
            "protocol_version": "1",
            "type": "body.end",
            "flow_id": "flow-truncated",
            "body_side": "response",
            "total_bytes": str(len(compressed)),
            "body": truncated,
        }
    )
    detail_text = application.flow_detail_text("flow-truncated")
    assert detail_text is not None
    detail = json.loads(detail_text)
    body_end = next(
        message for message in detail["messages"] if message["type"] == "body.end"
    )
    assert body_end["total_bytes"] == str(len(compressed))
    assert body_end["body"] == truncated


def test_new_search_cancels_and_serializes_an_older_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        server = ApiServer(
            ApiServerConfig(
                host="127.0.0.1",
                port=0,
                storage_path=tmp_path / "flows.sqlite",
            )
        )
        storage = server._storage
        assert storage is not None
        first_started = threading.Event()
        active = 0
        max_active = 0
        active_lock = threading.Lock()

        def fake_search(
            query: str, limit: int, cancel: threading.Event | None = None
        ) -> tuple[list[object], bool]:
            nonlocal active, max_active
            assert limit == 50
            assert cancel is not None
            with active_lock:
                active += 1
                max_active = max(max_active, active)
            try:
                if query == "first":
                    first_started.set()
                    cancel.wait(timeout=2)
                    raise SearchCancelled
                return [], False
            finally:
                with active_lock:
                    active -= 1

        monkeypatch.setattr(storage, "search", fake_search)

        class BufferWriter:
            def __init__(self) -> None:
                self.data = bytearray()

            def write(self, data: bytes) -> None:
                self.data.extend(data)

            async def drain(self) -> None:
                return

        first_writer = BufferWriter()
        second_writer = BufferWriter()
        first_task = asyncio.create_task(
            server._handle_plain_get("/api/v1/search?q=first", first_writer)  # type: ignore[arg-type]
        )
        assert await asyncio.to_thread(first_started.wait, 1)
        second_task = asyncio.create_task(
            server._handle_plain_get("/api/v1/search?q=second", second_writer)  # type: ignore[arg-type]
        )
        await asyncio.gather(first_task, second_task)
        assert first_writer.data == b""
        assert second_writer.data.startswith(b"HTTP/1.1 200 ")
        assert max_active == 1
        await server.close()

    run_async(scenario)


def test_sweep_task_publishes_expiry_without_traffic(tmp_path: Path) -> None:
    async def scenario() -> None:
        now = {"value": 0.0}
        store = MemoryStore(8, max_age_seconds=5.0, clock=lambda: now["value"])
        application = ApiApplication(store, source_id="test-source")
        config = ApiServerConfig(
            host="127.0.0.1", port=0, sweep_interval_seconds=0.02
        )
        server = ApiServer(config, application=application)
        await server.start()
        try:
            application.ingest(metadata_message())
            client, _head = await WsClient.connect(server.bound_port)
            assert client is not None
            for _ in range(3):
                await client.read_message()
            now["value"] = 60.0
            delta = await client.read_message()
            assert delta["type"] == "browser.delta"
            assert delta["changes"] == [{"op": "remove", "flow_id": "flow-1"}]
            await client.close()
        finally:
            await server.close()

    run_async(scenario)


def test_server_config_bounds_prefix_to_the_ingest_line_capacity() -> None:
    ApiServerConfig(max_body_prefix_bytes=MAX_INGEST_BODY_PREFIX_BYTES)
    with pytest.raises(ApiServerError):
        ApiServerConfig(max_body_prefix_bytes=MAX_INGEST_BODY_PREFIX_BYTES + 1)


def test_server_config_defaults_body_prefix_to_the_wire_ceiling() -> None:
    # F6: uncapped defaults let real API responses render whole in the UI. F5
    # shipped a 1 MiB default that quietly truncated typical Anthropic
    # responses. `--max-body-prefix-bytes` is still an explicit override.
    default = ApiServerConfig()
    assert default.max_body_prefix_bytes == MAX_INGEST_BODY_PREFIX_BYTES


def test_config_from_argv_defaults_body_prefix_to_the_wire_ceiling() -> None:
    config = config_from_argv([])
    assert config.max_body_prefix_bytes == MAX_INGEST_BODY_PREFIX_BYTES


def test_two_max_prefix_bodies_fit_one_bounded_ingest_line() -> None:
    prefix = b"x" * MAX_INGEST_BODY_PREFIX_BYTES
    descriptor = captured_body(prefix)
    message = metadata_message(request_body=descriptor)
    metadata = message["metadata"]
    assert isinstance(metadata, dict)
    metadata["response_headers"] = [
        {"name": f"x-header-{index}", "value": "v" * 128} for index in range(128)
    ]
    metadata["response_body"] = dict(descriptor)
    parse_message(message)
    line = json.dumps(message, separators=(",", ":")).encode("utf-8") + b"\n"
    assert len(line) <= MAX_INGEST_LINE_BYTES
    assert parse_ingest_line(line) == message


def test_max_prefix_and_header_boundary_is_accepted_and_one_byte_over_rejected() -> None:
    prefix = b"x" * MAX_INGEST_BODY_PREFIX_BYTES
    ApiServerConfig(max_body_prefix_bytes=MAX_INGEST_BODY_PREFIX_BYTES)

    def line_for_header_bytes(header_bytes: int) -> bytes:
        message = metadata_message(request_body=captured_body(prefix))
        metadata = message["metadata"]
        assert isinstance(metadata, dict)
        header = {"name": "x", "value": "v" * (header_bytes - 1)}
        metadata["request_headers"] = [header]
        metadata["response_body"] = captured_body(prefix)
        metadata["response_headers"] = [header]
        return json.dumps(message, separators=(",", ":")).encode("utf-8") + b"\n"

    accepted_line = line_for_header_bytes(MAX_METADATA_HEADER_BYTES)
    assert len(accepted_line) <= MAX_INGEST_LINE_BYTES
    accepted = parse_ingest_line(accepted_line)
    parse_message(accepted)

    rejected_line = line_for_header_bytes(MAX_METADATA_HEADER_BYTES + 1)
    with pytest.raises(ProtocolError, match="exceeds"):
        parse_message(parse_ingest_line(rejected_line))


def test_serialized_header_overhead_controls_fragmented_header_capacity() -> None:
    def message_with_headers(prefix_size: int, header_count: int) -> dict[str, object]:
        message = metadata_message(request_body=captured_body(b"x" * prefix_size))
        metadata = message["metadata"]
        assert isinstance(metadata, dict)
        headers = [{"name": "x", "value": "v"} for _ in range(header_count)]
        metadata["request_headers"] = headers
        metadata["response_headers"] = headers
        metadata["response_body"] = captured_body(b"x" * prefix_size)
        return message

    # Each side is exactly at the documented raw header-byte bound. The
    # fragmented form still fits while the complete serialized envelope is
    # below the ingest limit.
    accepted = message_with_headers(1024 * 1024, MAX_METADATA_HEADER_BYTES // 2)
    accepted_line = json.dumps(accepted, separators=(",", ":")).encode("utf-8") + b"\n"
    assert len(accepted_line) <= MAX_INGEST_LINE_BYTES
    parse_message(accepted)

    # The same legal raw headers cannot be combined with the largest body
    # prefix once per-header JSON syntax is included in the envelope.
    rejected = message_with_headers(MAX_INGEST_BODY_PREFIX_BYTES, MAX_METADATA_HEADER_BYTES // 2)
    rejected_line = json.dumps(rejected, separators=(",", ":")).encode("utf-8") + b"\n"
    assert len(rejected_line) > MAX_INGEST_LINE_BYTES
    with pytest.raises(ProtocolError, match="ingest line"):
        parse_message(rejected)


def test_surrogateescaped_header_value_is_rejected_at_protocol_boundary() -> None:
    message = metadata_message()
    metadata = message["metadata"]
    assert isinstance(metadata, dict)
    metadata["request_headers"] = [{"name": "x", "value": "bad\udc80"}]

    with pytest.raises(ProtocolError, match="surrogate"):
        parse_message(message)


async def _peer_saw_close(reader: asyncio.StreamReader) -> bool:
    try:
        return await reader.read(1024) == b""
    except ConnectionError:
        return True


def test_close_terminates_connected_websocket_and_ingest_peers(
    socket_dir: Path,
) -> None:
    async def scenario() -> None:
        server = make_server(socket_dir)
        await server.start()
        socket_path = server.config.capture_socket
        assert socket_path is not None
        client, _head = await WsClient.connect(server.bound_port)
        assert client is not None
        for _ in range(3):
            await client.read_message()
        ingest_reader, ingest_writer = await asyncio.open_unix_connection(
            str(socket_path)
        )
        await wait_until(lambda: server.counters["ingest_connections"] == 1)
        assert server.application.subscriber_count == 1

        # Regression: Server.wait_closed() waits for active connection
        # handlers, so close() previously hung forever while a WebSocket
        # client or ingest producer stayed connected.
        await asyncio.wait_for(server.close(), timeout=5.0)

        # wait_closed() releases on transport teardown; handler finalizers
        # (subscriber cleanup) run within the next loop ticks.
        await wait_until(lambda: server.application.subscriber_count == 0)
        assert await asyncio.wait_for(_peer_saw_close(client.reader), timeout=5.0)
        assert await asyncio.wait_for(_peer_saw_close(ingest_reader), timeout=5.0)
        await client.close()
        ingest_writer.close()
        try:
            await ingest_writer.wait_closed()
        except ConnectionError:
            pass

    run_async(scenario)


def test_server_can_restart_and_accept_http_after_close(tmp_path: Path) -> None:
    async def scenario() -> None:
        server = make_server(tmp_path, with_ingest=False)
        await server.start()
        await server.close()

        await server.start()
        try:
            response = await http_request(
                server.bound_port, get("/api/v1/health", server.bound_port)
            )
            assert response.startswith(b"HTTP/1.1 200 ")
        finally:
            await server.close()

    run_async(scenario)


def test_incremental_metadata_projection_matches_full_rescan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The single-flow fast path must stay equivalent to a full projection."""

    import mitm_inspector.api.app as app_module

    projection_calls = 0
    original_projection = app_module.collect_grid_flows

    def spy_projection(store: MemoryStore):
        nonlocal projection_calls
        projection_calls += 1
        return original_projection(store)

    monkeypatch.setattr(app_module, "collect_grid_flows", spy_projection)
    application = make_application(max_items=64)
    scripted = [
        metadata_message("flow-a"),
        metadata_message("flow-b"),
        metadata_message("flow-a", path="/v1/messages/updated"),
        metadata_message("flow-c"),
        metadata_message("flow-b", request_body=captured_body()),
        metadata_message("flow-a", path="/v1/messages/updated"),
    ]
    application.ingest(scripted[0])
    projection_calls = 0
    for message in scripted[1:]:
        application.ingest(message)
        expected = collect_grid_flows(application.store)
        assert application._state.published == expected
        assert list(application._state.published) == list(expected)
    assert projection_calls == 0


def test_incremental_metadata_projection_emits_single_upsert_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mitm_inspector.api.app as app_module

    projection_calls = 0
    original_projection = app_module.collect_grid_flows

    def spy_projection(store: MemoryStore):
        nonlocal projection_calls
        projection_calls += 1
        return original_projection(store)

    monkeypatch.setattr(app_module, "collect_grid_flows", spy_projection)
    application = make_application(max_items=64)
    application.ingest(metadata_message("flow-a"))
    application.ingest(metadata_message("flow-b"))
    projection_calls = 0
    frames: list[str] = []
    application.subscribe(lambda text: frames.append(text) or True)
    frames.clear()
    application.ingest(metadata_message("flow-a", path="/v1/updated"))
    deltas = [json.loads(frame) for frame in frames if '"browser.delta"' in frame]
    assert len(deltas) == 1
    changes = deltas[0]["changes"]
    assert len(changes) == 1
    assert changes[0]["op"] == "upsert"
    assert changes[0]["flow"]["flow_id"] == "flow-a"
    assert changes[0]["flow"]["path"] == "/v1/updated"
    assert projection_calls == 0


def test_duplicate_metadata_ingest_emits_no_delta(monkeypatch: pytest.MonkeyPatch) -> None:
    import mitm_inspector.api.app as app_module

    projection_calls = 0
    original_projection = app_module.collect_grid_flows

    def spy_projection(store: MemoryStore):
        nonlocal projection_calls
        projection_calls += 1
        return original_projection(store)

    monkeypatch.setattr(app_module, "collect_grid_flows", spy_projection)
    application = make_application(max_items=64)
    application.ingest(metadata_message("flow-a"))
    projection_calls = 0
    before = application.cursor
    result = application.ingest(metadata_message("flow-a"))
    assert result.delta_emitted is False
    assert application.cursor == before
    assert projection_calls == 0


def test_eviction_during_metadata_ingest_falls_back_to_full_projection() -> None:
    application = make_application(max_items=4)
    for index in range(8):
        application.ingest(metadata_message(f"flow-{index}"))
        expected = collect_grid_flows(application.store)
        assert application._state.published == expected
        assert list(application._state.published) == list(expected)


def test_close_converges_when_connections_race_the_close_snapshot(
    socket_dir: Path,
) -> None:
    """A handler registered after the close snapshot must not make it stall."""

    async def scenario() -> None:
        server = make_server(socket_dir, with_ingest=False)
        await server.start()
        http_server = server._http_server
        assert http_server is not None

        handler_registered = asyncio.Event()
        handler_started = asyncio.Event()
        release_handler = asyncio.Event()
        snapshot_taken = asyncio.Event()
        wait_closed_gate = asyncio.Event()
        read_forever = asyncio.Event()
        original_handler = server._handle_http_connection

        class FakeTransport:
            def abort(self) -> None:
                return

        class FakeWriter:
            transport = FakeTransport()

            def close(self) -> None:
                handler_registered.set()
                wait_closed_gate.set()

            async def wait_closed(self) -> None:
                return

        class FakeReader:
            async def readuntil(self, _separator: bytes) -> bytes:
                await read_forever.wait()
                return b""

        fake_writer = FakeWriter()
        fake_reader = FakeReader()

        async def gated_handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            handler_started.set()
            await release_handler.wait()
            await original_handler(reader, writer)

        scheduled_after_snapshot = False

        class LateConnectionSet(set[object]):
            def __iter__(self):  # type: ignore[no-untyped-def]
                nonlocal scheduled_after_snapshot
                if not scheduled_after_snapshot:
                    scheduled_after_snapshot = True
                    snapshot_taken.set()
                    asyncio.create_task(gated_handler(fake_reader, fake_writer))
                return super().__iter__()

            def add(self, item: object) -> None:
                handler_registered.set()
                super().add(item)

        server._connections = LateConnectionSet()  # type: ignore[assignment]
        real_wait_closed = http_server.wait_closed

        async def gated_wait_closed() -> None:
            await wait_closed_gate.wait()
            await real_wait_closed()

        http_server.wait_closed = gated_wait_closed  # type: ignore[method-assign]

        # The set's iterator schedules an accepted handler after close() has
        # taken its snapshot.  The handler then blocks before its read loop;
        # the fixed implementation rechecks and sees its closing flag.
        close_task = asyncio.create_task(server.close())
        await asyncio.wait_for(snapshot_taken.wait(), timeout=5.0)
        await asyncio.wait_for(handler_started.wait(), timeout=5.0)
        assert not handler_registered.is_set()
        release_handler.set()
        await asyncio.wait_for(close_task, timeout=5.0)
        assert handler_registered.is_set()

    run_async(scenario)

import base64
import threading
from collections.abc import Iterator, Mapping
from types import SimpleNamespace

import pytest

from mitm_inspector.capture import sink as sink_module
from mitm_inspector.capture.addon import (
    CaptureAddon,
    addons,
    make_addon_from_environment,
)
from mitm_inspector.capture.config import CaptureConfig
from mitm_inspector.capture.sink import BoundedMessageSink
from mitm_inspector.protocol import KnownParsedMessage, ParsedMessageResult, parse_message
from mitm_inspector.store.memory import MemoryStore


class FakeHeaders(dict[str, str]):
    def items(self, multi: bool = False) -> Iterator[tuple[str, str]]:
        return iter(super().items())


def fake_flow(
    flow_id: str = "flow-1",
    *,
    request_body: bytes | None = None,
    response_body: bytes | None = None,
) -> SimpleNamespace:
    request = SimpleNamespace(
        method="POST",
        scheme="https",
        host="api.example.test",
        port=443,
        path="/v1/messages?api_key=do-not-leak",
        headers=FakeHeaders(
            {
                "Host": "api.example.test",
                "Content-Type": "text/event-stream",
                "Authorization": "Bearer request-secret",
                "X-Api-Key": "header-secret",
            }
        ),
        raw_content=request_body,
        stream=False,
    )
    response = SimpleNamespace(
        headers=FakeHeaders(
            {
                "Content-Type": "text/event-stream",
                "Set-Cookie": "session-secret",
            }
        ),
        raw_content=response_body,
        stream=False,
    )
    return SimpleNamespace(id=flow_id, request=request, response=response, error=None)


def payloads(addon: CaptureAddon) -> list[dict[str, object]]:
    result = []
    for message in addon.drain():
        assert isinstance(message, KnownParsedMessage)
        result.append(dict(message.message))
    return result


def lifecycle_states(messages: list[dict[str, object]]) -> list[str]:
    return [
        str(message["state"])
        for message in messages
        if message.get("type") == "flow.lifecycle"
    ]


def test_capture_redacts_query_and_headers_before_sink_ingress() -> None:
    addon = CaptureAddon(source_id="test-source", clock=lambda: "2026-01-01T00:00:00Z")
    flow = fake_flow(request_body=b"{}", response_body=b"ok")

    addon.requestheaders(flow)
    messages = payloads(addon)
    metadata = next(
        message["metadata"] for message in messages if message["type"] == "flow.metadata"
    )
    assert isinstance(metadata, Mapping)
    assert metadata["path"] == "/v1/messages"
    assert "api_key" not in str(metadata)
    headers = metadata["request_headers"]
    assert isinstance(headers, tuple)
    assert {
        header["value"]
        for header in headers
        if str(header["name"]).lower() == "authorization"
    } == {
        "[REDACTED]"
    }
    assert "header-secret" not in str(messages)
    assert "request-secret" not in str(messages)


def test_capture_declares_only_documented_http_hooks() -> None:
    assert CaptureAddon.PUBLIC_HOOKS == {
        "requestheaders",
        "request",
        "responseheaders",
        "response",
        "error",
        "load",
    }
    assert all(callable(getattr(CaptureAddon, name)) for name in CaptureAddon.PUBLIC_HOOKS)


def test_response_can_start_before_request_end_and_duplicate_hooks_are_idempotent() -> None:
    addon = CaptureAddon(source_id="test-source", clock=lambda: "now")
    flow = fake_flow()

    addon.requestheaders(flow)
    addon.requestheaders(flow)
    addon.responseheaders(flow)
    assert flow.response.stream(b"data: first\n\n") == b"data: first\n\n"
    addon.response(flow)
    addon.response(flow)
    flow.request.raw_content = b"request"
    addon.request(flow)
    addon.request(flow)
    messages = payloads(addon)

    states = lifecycle_states(messages)
    assert states.index("response_started") < states.index("request_end")
    assert states.count("request_started") == 1
    assert states.count("response_end") == 1
    assert states.count("flow_completed") == 1


def test_response_completion_is_deferred_until_late_missing_request_end() -> None:
    addon = CaptureAddon(clock=lambda: "now")
    flow = fake_flow(request_body=None)

    addon.requestheaders(flow)
    addon.responseheaders(flow)
    addon.response(flow)
    before_request = payloads(addon)
    assert "flow_completed" not in lifecycle_states(before_request)

    addon.request(flow)
    after_request = payloads(addon)
    states = lifecycle_states(after_request)
    assert states[-4:] == ["request_body", "request_end", "flow_completed"][-4:]
    body_end = [message for message in after_request if message["type"] == "body.end"]
    assert any(message["body_side"] == "request" for message in body_end)


def test_error_with_missing_content_always_finishes_request_before_completion() -> None:
    addon = CaptureAddon(clock=lambda: "now")
    flow = fake_flow(request_body=None)
    flow.error = SimpleNamespace(msg="synthetic failure")

    addon.error(flow)
    states = lifecycle_states(payloads(addon))
    assert states[-4:] == ["request_body", "request_end", "error", "flow_completed"]


def test_empty_headers_are_captured_metadata_not_treated_as_uninitialized() -> None:
    addon = CaptureAddon(clock=lambda: "now")
    flow = fake_flow()
    flow.request.headers = FakeHeaders()

    addon.requestheaders(flow)
    metadata = [
        message["metadata"]
        for message in payloads(addon)
        if message["type"] == "flow.metadata"
    ][0]
    assert isinstance(metadata, Mapping)
    assert metadata["request_headers"] == ()


def test_stream_observation_is_incremental_bounded_and_byte_preserving() -> None:
    addon = CaptureAddon(max_body_prefix_bytes=5, clock=lambda: "now")
    flow = fake_flow()
    addon.requestheaders(flow)
    addon.responseheaders(flow)
    first = b"abc"
    second = b"defgh"
    assert flow.response.stream(first) is first
    assert flow.response.stream(second) is second
    flow.response.raw_content = None
    addon.response(flow)
    messages = payloads(addon)

    chunks = [message for message in messages if message["type"] == "body.chunk"]
    assert [message["data_base64"] for message in chunks] == [
        base64.b64encode(first).decode(),
        base64.b64encode(b"de").decode(),
    ]
    body_end = [message for message in messages if message["type"] == "body.end"][-1]
    assert body_end["total_bytes"] == "8"
    body = body_end["body"]
    assert isinstance(body, Mapping)
    assert body["state"] == "truncated"
    assert body["captured_bytes"] == "5"
    assert base64.b64decode(body["data"]) == b"abcde"


@pytest.mark.parametrize(
    ("body", "state"),
    [(None, "missing"), (b"", "empty"), (b"abc", "captured"), (b"abcd", "truncated")],
)
def test_body_descriptors_distinguish_missing_empty_captured_and_truncated(
    body: bytes | None, state: str
) -> None:
    addon = CaptureAddon(max_body_prefix_bytes=3, clock=lambda: "now")
    flow = fake_flow(request_body=body)
    addon.requestheaders(flow)
    addon.request(flow)
    body_end = [message for message in payloads(addon) if message["type"] == "body.end"][0]
    descriptor = body_end["body"]
    assert isinstance(descriptor, Mapping)
    assert descriptor["state"] == state


def test_error_is_sanitized_lifecycle_and_completes_without_error_text() -> None:
    addon = CaptureAddon(clock=lambda: "now")
    flow = fake_flow(request_body=None)
    flow.error = SimpleNamespace(msg="authorization error-secret")
    addon.requestheaders(flow)
    addon.error(flow)
    messages = payloads(addon)
    assert lifecycle_states(messages)[-2:] == ["error", "flow_completed"]
    assert "error-secret" not in str(messages)


def test_full_sink_is_nonblocking_and_observably_drops() -> None:
    sink = BoundedMessageSink(max_pending=1)
    first = parse_message(
        {"protocol_version": "1", "type": "future.one", "value": "first"}
    )
    second = parse_message(
        {"protocol_version": "1", "type": "future.two", "value": "second"}
    )
    assert sink.offer(first)
    assert not sink.offer(second)
    assert sink.dropped_count == 1
    assert sink.pending_count == 1


def test_sink_lock_contention_returns_immediately_and_emits_gap() -> None:
    sink = BoundedMessageSink(max_pending=4)
    sink._lock.acquire()
    try:
        assert not sink.offer(parse_message({"protocol_version": "1", "type": "future.lock"}))
    finally:
        sink._lock.release()
    assert sink.offer(parse_message({"protocol_version": "1", "type": "future.next"}))
    messages = sink.drain()
    assert [
        (message.message if isinstance(message, KnownParsedMessage) else message.payload)["type"]
        for message in messages
    ] == [
        "stream.gap",
        "future.next",
    ]
    gap = messages[0]
    assert isinstance(gap, KnownParsedMessage)
    assert gap.message["dropped_count"] == "1"


def test_sink_queue_and_body_drops_are_in_band_and_delivery_positioned() -> None:
    sink = BoundedMessageSink(max_pending=1, max_body_bytes=1)
    assert sink.offer(parse_message({"protocol_version": "1", "type": "future.first"}))
    assert not sink.offer(parse_message({"protocol_version": "1", "type": "future.queue"}))
    sink.drain()
    assert not sink.offer(
        parse_message(
            {
                "protocol_version": "1",
                "type": "body.end",
                "flow_id": "f",
                "body_side": "response",
                "total_bytes": "2",
                "body": {
                    "state": "captured",
                    "size_bytes": "2",
                    "encoding": "base64",
                    "data": base64.b64encode(b"ab").decode(),
                },
            }
        )
    )
    assert sink.dropped_count == 2
    assert sink.offer(parse_message({"protocol_version": "1", "type": "future.after"}))
    messages = sink.drain()
    assert [
        (message.message if isinstance(message, KnownParsedMessage) else message.payload)["type"]
        for message in messages
    ] == [
        "stream.gap",
        "future.after",
    ]
    delivered = messages[-1]
    delivered_payload = (
        delivered.message if isinstance(delivered, KnownParsedMessage) else delivered.payload
    )
    assert delivered_payload["delivery_position"] == "4"


def test_callback_cannot_block_stream_forwarding_and_runs_only_on_explicit_drain() -> None:
    delivered: list[ParsedMessageResult] = []
    addon = CaptureAddon(emit=delivered.append, clock=lambda: "now")
    flow = fake_flow()
    addon.requestheaders(flow)
    addon.responseheaders(flow)

    chunk = b"data: event\n\n"
    assert flow.response.stream(chunk) is chunk
    assert delivered == []
    addon.drain()
    assert delivered


def test_hostile_callback_runs_only_on_consumer_thread_after_stream_returns() -> None:
    entered = threading.Event()
    release = threading.Event()

    def blocked_emit(_message: ParsedMessageResult) -> None:
        entered.set()
        release.wait(timeout=5)

    addon = CaptureAddon(emit=blocked_emit, clock=lambda: "now")
    flow = fake_flow()
    addon.requestheaders(flow)
    addon.responseheaders(flow)
    chunk = b"data: callback\n\n"
    assert flow.response.stream(chunk) is chunk
    assert not entered.is_set()
    drain_thread = threading.Thread(target=addon.drain)
    drain_thread.start()
    assert entered.wait(timeout=1)
    release.set()
    drain_thread.join(timeout=1)
    assert not drain_thread.is_alive()


def test_partial_response_error_finishes_response_and_disables_late_stream_chunks() -> None:
    addon = CaptureAddon(clock=lambda: "now")
    flow = fake_flow()
    addon.requestheaders(flow)
    addon.responseheaders(flow)
    callback = flow.response.stream
    chunk = b"data: partial\n\n"
    assert callback(chunk) is chunk
    addon.error(flow)
    late = b"data: late\n\n"
    assert callback(late) is late
    messages = payloads(addon)
    states = lifecycle_states(messages)
    assert states[-6:] == [
        "response_body",
        "request_body",
        "request_end",
        "response_end",
        "error",
        "flow_completed",
    ]
    response_end = next(
        index
        for index, message in enumerate(messages)
        if message.get("type") == "flow.lifecycle" and message.get("state") == "response_end"
    )
    body_end = [
        index
        for index, message in enumerate(messages)
        if message.get("type") == "body.end" and message.get("body_side") == "response"
    ]
    assert body_end and body_end[0] < response_end
    chunks = [
        message
        for message in messages
        if message.get("type") == "body.chunk" and message.get("body_side") == "response"
    ]
    assert len(chunks) == 1
    assert addon.counters["active_prefix_bytes"] == 0


def test_sink_emits_a_final_gap_without_a_later_retained_message() -> None:
    sink = BoundedMessageSink(max_pending=1)
    assert sink.offer(parse_message({"protocol_version": "1", "type": "future.first"}))
    assert not sink.offer(parse_message({"protocol_version": "1", "type": "future.dropped"}))
    first_drain = sink.drain()
    assert [
        (message.message if isinstance(message, KnownParsedMessage) else message.payload)["type"]
        for message in first_drain
    ] == ["future.first", "stream.gap"]
    gap = first_drain[-1]
    assert isinstance(gap, KnownParsedMessage)
    assert gap.message["expected_sequence"] == "1"
    assert gap.message["actual_sequence"] == "3"
    assert sink.drain() == []


def test_contended_offer_does_no_payload_work(monkeypatch: pytest.MonkeyPatch) -> None:
    sink = BoundedMessageSink()
    message = parse_message({"protocol_version": "1", "type": "future.contended"})

    def fail(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("payload work ran on the contended producer path")

    monkeypatch.setattr(sink_module, "require_parsed_message", fail)
    monkeypatch.setattr(sink_module, "_message_body_bytes", fail)
    monkeypatch.setattr(sink_module, "_message_weight", fail)
    sink._lock.acquire()
    try:
        assert not sink.offer(message)
    finally:
        sink._lock.release()
    assert sink.dropped_count == 1


def test_producer_drop_during_drain_is_not_cleared_or_duplicated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = BoundedMessageSink(max_pending=2)
    assert sink.offer(parse_message({"protocol_version": "1", "type": "future.first"}))
    entered = threading.Event()
    release = threading.Event()
    original_positioner = sink_module._with_delivery_position

    def paused_positioner(message: ParsedMessageResult, position: int) -> ParsedMessageResult:
        entered.set()
        assert release.wait(timeout=1)
        return original_positioner(message, position)

    monkeypatch.setattr(sink_module, "_with_delivery_position", paused_positioner)
    first_result: list[ParsedMessageResult] = []
    drain_thread = threading.Thread(target=lambda: first_result.extend(sink.drain()))
    drain_thread.start()
    assert entered.wait(timeout=1)
    sink.record_loss()
    assert sink.offer(parse_message({"protocol_version": "1", "type": "future.after"}))
    release.set()
    drain_thread.join(timeout=1)
    assert not drain_thread.is_alive()

    second_result = sink.drain()
    combined = first_result + second_result
    types = [
        (message.message if isinstance(message, KnownParsedMessage) else message.payload)["type"]
        for message in combined
    ]
    assert types == ["future.first", "stream.gap", "future.after"]
    assert types.count("stream.gap") == 1


def test_concurrent_drop_events_are_emitted_once_in_one_resyncable_gap() -> None:
    sink = BoundedMessageSink(max_pending=1)
    assert sink.offer(parse_message({"protocol_version": "1", "type": "future.first"}))
    workers = [
        threading.Thread(
            target=sink.offer,
            args=(parse_message({"protocol_version": "1", "type": f"future.drop.{index}"}),),
        )
        for index in range(64)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    messages = sink.drain()
    gaps = [
        message
        for message in messages
        if isinstance(message, KnownParsedMessage) and message.message["type"] == "stream.gap"
    ]
    assert len(gaps) == 1
    assert gaps[0].message["dropped_count"] == "64"
    assert sink.dropped_count == 64
    assert sink.drain() == []


def test_active_flow_bounds_evict_zero_body_flood_with_tombstones_and_gaps() -> None:
    config = CaptureConfig(
        source_id="source",
        max_body_prefix_bytes=0,
        max_in_memory_bytes=1,
        max_pending_messages=1,
    )
    addon = CaptureAddon(config=config, max_active_flows=1, clock=lambda: "now")
    for index in range(1_000):
        addon.requestheaders(fake_flow(f"flood-{index}"))
        assert addon.counters["active_flows"] <= 1
        assert addon.counters["active_metadata_bytes"] <= 1
    messages = payloads(addon)
    assert addon.counters["active_flows"] == 0
    assert addon.counters["evicted_flows"] == 1_000
    assert any(message["type"] == "stream.gap" for message in messages)


def test_active_flow_age_bound_tombstones_stale_incomplete_flow() -> None:
    now = [0.0]
    addon = CaptureAddon(
        max_active_age_seconds=5,
        active_clock=lambda: now[0],
        clock=lambda: "now",
    )
    addon.requestheaders(fake_flow("stale"))
    now[0] = 5.0
    messages = payloads(addon)
    assert addon.counters["evicted_flows"] == 1
    assert addon.counters["active_flows"] == 0
    assert "stale" in addon._completed_ids
    assert any(message["type"] == "stream.gap" for message in messages)


def test_redaction_rejects_str_subclass_before_authorization_coercion() -> None:
    called = False

    class HostileString(str):
        def __str__(self) -> str:
            nonlocal called
            called = True
            raise AssertionError("hostile __str__ must not run")

    addon = CaptureAddon(clock=lambda: "now")
    flow = fake_flow()
    flow.request.headers = FakeHeaders({"Authorization": HostileString("secret")})
    with pytest.raises(ValueError, match="exact str or bytes"):
        addon.requestheaders(flow)
    assert not called


def test_huge_integer_is_rejected_before_sink_or_store_retention() -> None:
    huge = parse_message(
        {
            "protocol_version": "1",
            "type": "future.huge-number",
            "value": 1 << 8_000_000,
        }
    )
    with pytest.raises(ValueError, match="bounded protocol numeric range"):
        BoundedMessageSink(max_memory_bytes=512).offer(huge)
    store = MemoryStore(max_memory_bytes=512)
    with pytest.raises(ValueError, match="bounded protocol numeric range"):
        store.append(huge)
    assert store.counters["retained_messages"] == 0


def test_module_addon_load_parses_valid_environment_without_ipc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = {
        "MITM_INSPECTOR_CAPTURE_SOCKET": "/tmp/mitm-inspector.sock",
        "MITM_INSPECTOR_SOURCE_ID": "source-from-env",
        "MITM_INSPECTOR_MAX_BODY_PREFIX_BYTES": "64",
        "MITM_INSPECTOR_MAX_IN_MEMORY_BYTES": "128",
        "MITM_INSPECTOR_MAX_PENDING_MESSAGES": "7",
    }
    for name, value in config.items():
        monkeypatch.setenv(name, value)
    CaptureConfig.from_environment(config)
    addon = CaptureAddon()
    addon.load(object())
    assert addons and isinstance(addons[0], CaptureAddon)
    assert addon.capture_socket == "/tmp/mitm-inspector.sock"
    assert addon.source_id == "source-from-env"
    assert addon.max_body_prefix_bytes == 64
    assert addon.max_in_memory_bytes == 128
    assert addon.sink._max_pending == 7
    factory_addon = make_addon_from_environment()
    assert factory_addon.capture_socket == "/tmp/mitm-inspector.sock"
    assert CaptureAddon.from_environment().source_id == "source-from-env"


@pytest.mark.parametrize(
    "environment",
    [
        {"MITM_INSPECTOR_CAPTURE_SOCKET": "relative.sock"},
        {"MITM_INSPECTOR_MAX_PENDING_MESSAGES": "0"},
        {
            "MITM_INSPECTOR_MAX_BODY_PREFIX_BYTES": "129",
            "MITM_INSPECTOR_MAX_IN_MEMORY_BYTES": "128",
        },
        {"MITM_INSPECTOR_MAX_BODY_PREFIX_BYTES": "not-a-number"},
    ],
)
def test_invalid_environment_is_rejected_clearly(
    environment: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):
        CaptureAddon().load(object())


def completed_flow_message(flow_id: str, sequence: str) -> ParsedMessageResult:
    return parse_message(
        {
            "protocol_version": "1",
            "type": "flow.lifecycle",
            "source_id": "source",
            "flow_id": flow_id,
            "event_id": f"event-{flow_id}",
            "occurred_at": "now",
            "sequence": sequence,
            "state": "flow_completed",
        }
    )


def test_store_eviction_and_mutation_isolation_are_deterministic() -> None:
    store = MemoryStore(max_items=2)
    for index in range(3):
        store.append(completed_flow_message(f"flow-{index}", str(index)))
    assert store.counters["completed_flows"] == 2
    assert store.counters["evicted_flows"] == 1
    retained = list(store.newest_first())
    assert len(retained) == 2
    assert all(isinstance(message, KnownParsedMessage) for message in retained)
    with pytest.raises(TypeError):
        retained[0].message["state"] = "forged"  # type: ignore[index]


def test_store_body_budget_and_expiry_counters_are_observable() -> None:
    now = [0.0]
    store = MemoryStore(max_items=10, max_body_bytes=3, max_age_seconds=10, clock=lambda: now[0])
    store.append(
        parse_message(
            {
                "protocol_version": "1",
                "type": "body.end",
                "flow_id": "body-flow",
                "body_side": "response",
                "total_bytes": "4",
                "body": {
                    "state": "truncated",
                    "size_bytes": "4",
                    "captured_bytes": "4",
                    "encoding": "base64",
                    "data": base64.b64encode(b"data").decode(),
                },
            }
        )
    )
    assert store.counters["body_bytes"] == 0
    assert store.counters["body_budget_drops"] == 1

    store = MemoryStore(max_items=10, max_age_seconds=10, clock=lambda: now[0])
    store.append(completed_flow_message("old", "0"))
    now[0] = 11.0
    assert list(store.newest_first()) == []
    assert store.counters["expired_flows"] == 1


def test_store_bounds_incomplete_flows_per_flow_global_and_standalone_messages() -> None:
    store = MemoryStore(
        max_items=10,
        max_messages=5,
        max_messages_per_flow=2,
        max_standalone_messages=2,
    )
    for index in range(10):
        store.append(
            parse_message(
                {
                    "protocol_version": "1",
                    "type": "flow.lifecycle",
                    "source_id": "s",
                    "flow_id": f"incomplete-{index}",
                    "event_id": f"e-{index}",
                    "occurred_at": "now",
                    "sequence": str(index),
                    "state": "request_started",
                }
            )
        )
    for index in range(10):
        store.append(parse_message({"protocol_version": "1", "type": f"future.{index}"}))
    assert store.counters["retained_messages"] <= 5
    assert store.counters["retained_flows"] <= 5
    assert store.counters["dropped_messages"] == 0
    assert store.counters["message_evictions"] > 0


def flow_metadata_with_body(flow_id: str, data: bytes) -> dict[str, object]:
    request_body: dict[str, str]
    if data:
        request_body = {
            "state": "captured",
            "size_bytes": str(len(data)),
            "encoding": "base64",
            "data": base64.b64encode(data).decode(),
        }
    else:
        request_body = {"state": "empty", "size_bytes": "0"}
    return {
        "flow_id": flow_id,
        "method": "GET",
        "scheme": "https",
        "host": "example.test",
        "port": "443",
        "path": "/",
        "request_headers": [],
        "request_body": request_body,
    }


def test_store_counts_nested_snapshot_and_delta_body_budget() -> None:
    metadata = flow_metadata_with_body("nested", b"body")
    store = MemoryStore(max_body_bytes=4, max_messages=10)
    store.append(
        parse_message(
            {
                "protocol_version": "1",
                "type": "browser.snapshot",
                "snapshot_id": "s",
                "cursor": "0",
                "flows": [metadata],
            }
        )
    )
    store.append(
        parse_message(
            {
                "protocol_version": "1",
                "type": "browser.delta",
                "cursor": "1",
                "changes": [{"op": "upsert", "flow": metadata}],
            }
        )
    )
    assert store.counters["body_bytes"] <= 4
    assert store.counters["body_budget_drops"] > 0


def test_completed_eviction_uses_completion_order_not_insertion_order() -> None:
    store = MemoryStore(max_items=1)
    store.append(completed_flow_message("a", "0"))
    store.append(completed_flow_message("b", "1"))
    # A was inserted first but B completed last and must be retained.
    retained = list(store.newest_first())
    assert all(
        message.message["flow_id"] == "b"
        for message in retained
        if isinstance(message, KnownParsedMessage)
    )


def test_newest_first_tracks_coalesced_message_order_exactly() -> None:
    store = MemoryStore(max_items=10)
    metadata_a = parse_message(
        {
            "protocol_version": "1",
            "type": "flow.metadata",
            "metadata": flow_metadata_with_body("a", b""),
        }
    )
    lifecycle_a = parse_message(
        {
            "protocol_version": "1",
            "type": "flow.lifecycle",
            "source_id": "s",
            "flow_id": "a",
            "event_id": "a-event",
            "occurred_at": "now",
            "sequence": "0",
            "state": "request_started",
        }
    )
    metadata_b = parse_message(
        {
            "protocol_version": "1",
            "type": "flow.metadata",
            "metadata": flow_metadata_with_body("b", b""),
        }
    )
    store.append(metadata_a)
    store.append(lifecycle_a)
    store.append(metadata_b)
    store.append(metadata_a)
    newest = list(store.newest_first())
    assert [
        message.message["type"]
        for message in newest
        if isinstance(message, KnownParsedMessage)
    ] == ["flow.metadata", "flow.metadata", "flow.lifecycle"]


def test_max_messages_one_never_leaves_an_orphaned_active_record() -> None:
    store = MemoryStore(max_messages=1, max_messages_per_flow=10)
    for index in range(20):
        store.append(
            parse_message(
                {
                    "protocol_version": "1",
                    "type": "flow.lifecycle",
                    "source_id": "s",
                    "flow_id": "one",
                    "event_id": f"event-{index}",
                    "occurred_at": "now",
                    "sequence": str(index),
                    "state": "request_started",
                }
            )
        )
        counters = store.counters
        visible = list(store.newest_first())
        assert counters["retained_messages"] == len(visible) == 1
        assert counters["retained_messages"] <= store.max_messages

    terminal_store = MemoryStore(max_messages=1, max_messages_per_flow=1)
    for side in ("request", "response"):
        terminal_store.append(
            parse_message(
                {
                    "protocol_version": "1",
                    "type": "body.end",
                    "flow_id": "one-terminal",
                    "body_side": side,
                    "total_bytes": "0",
                    "body": {"state": "empty", "size_bytes": "0"},
                }
            )
        )
        assert terminal_store.counters["retained_messages"] == 1
        assert len(list(terminal_store.newest_first())) == 1


def test_long_sse_stream_evicts_chunks_but_keeps_terminal_messages() -> None:
    store = MemoryStore(max_messages=20, max_messages_per_flow=3)
    for index in range(40):
        store.append(
            parse_message(
                {
                    "protocol_version": "1",
                    "type": "body.chunk",
                    "flow_id": "sse",
                    "body_side": "response",
                    "chunk_index": str(index),
                    "offset_bytes": str(index),
                    "data_base64": base64.b64encode(f"event:{index}\n\n".encode()).decode(),
                }
            )
        )
    for index, state in enumerate(("request_end", "response_end", "flow_completed")):
        store.append(
            parse_message(
                {
                    "protocol_version": "1",
                    "type": "flow.lifecycle",
                    "source_id": "s",
                    "flow_id": "sse",
                    "event_id": f"terminal-{index}",
                    "occurred_at": "now",
                    "sequence": str(index),
                    "state": state,
                }
            )
        )
    for side in ("request", "response"):
        store.append(
            parse_message(
                {
                    "protocol_version": "1",
                    "type": "body.end",
                    "flow_id": "sse",
                    "body_side": side,
                    "total_bytes": "0",
                    "body": {"state": "empty", "size_bytes": "0"},
                }
            )
        )
    retained = [
        message.message
        for message in store.newest_first()
        if isinstance(message, KnownParsedMessage)
    ]
    assert store.counters["flow_message_evictions"] > 0
    assert store.counters["retained_messages"] <= 20
    assert store.counters["retained_messages"] <= store.max_messages_per_flow
    assert {
        message.get("state")
        for message in retained
        if message.get("type") == "flow.lifecycle"
    } >= {"flow_completed"}
    assert {
        message.get("body_side")
        for message in retained
        if message.get("type") == "body.end"
    } == {"request", "response"}


def test_canonical_memory_budget_counts_unknown_fields_and_nested_snapshots() -> None:
    unknown = parse_message(
        {
            "protocol_version": "1",
            "type": "future.large",
            "payload": {"unknown_blob": "x" * (1024 * 1024)},
        }
    )
    store = MemoryStore(max_memory_bytes=128, max_messages=10)
    store.append(unknown)
    assert store.counters["retained_messages"] == 0
    assert store.counters["memory_bytes"] == 0
    assert store.counters["memory_budget_drops"] == 1

    flows = [flow_metadata_with_body(f"flow-{index}", b"") for index in range(200)]
    snapshot = parse_message(
        {
            "protocol_version": "1",
            "type": "browser.snapshot",
            "snapshot_id": "snapshot",
            "cursor": "0",
            "flows": flows,
        }
    )
    store.append(snapshot)
    assert store.counters["memory_bytes"] <= 128
    assert store.counters["retained_messages"] == 0

    sink = BoundedMessageSink(max_pending=4, max_memory_bytes=128)
    assert not sink.offer(unknown)
    assert not sink.offer(snapshot)
    assert sink.memory_budget_drops == 2

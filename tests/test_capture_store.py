import base64
import gc
import subprocess
import sys
import threading
import time
import weakref
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import SimpleNamespace

import pytest

from mitm_inspector.capture import adapter as adapter_module
from mitm_inspector.capture import gate as gate_module
from mitm_inspector.capture import metrics as metrics_module
from mitm_inspector.capture import sequencer as sequencer_module
from mitm_inspector.capture import sink as sink_module
from mitm_inspector.capture.addon import (
    CaptureAddon,
    addons,
    make_addon_from_environment,
)
from mitm_inspector.capture.config import CaptureConfig
from mitm_inspector.capture.sink import BoundedMessageSink
from mitm_inspector.protocol import (
    MAX_U64,
    KnownParsedMessage,
    ParsedMessageResult,
    parse_message,
)
from mitm_inspector.store import memory as memory_module
from mitm_inspector.store.memory import MemoryStore


class FakeHeaders(dict[str, str]):
    def items(self, multi: bool = False) -> Iterator[tuple[str, str]]:
        return iter(super().items())


def fake_flow(
    flow_id: str = "flow-1",
    *,
    request_body: bytes | None = None,
    response_body: bytes | None = None,
    session_id: str | None = None,
    include_client_conn: bool = True,
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
    flow = SimpleNamespace(id=flow_id, request=request, response=response, error=None)
    if include_client_conn:
        flow.client_conn = SimpleNamespace(id=session_id)
    return flow


def test_capture_emits_client_connection_id_and_maps_missing_connection_to_null() -> None:
    addon = CaptureAddon(source_id="test-source", clock=lambda: "now")
    flow = fake_flow(session_id="client-connection-uuid")
    addon.requestheaders(flow)
    metadata = next(
        message["metadata"]
        for message in payloads(addon)
        if message["type"] == "flow.metadata"
    )
    assert isinstance(metadata, Mapping)
    assert metadata["session_id"] == "client-connection-uuid"

    addon = CaptureAddon(source_id="test-source", clock=lambda: "now")
    addon.requestheaders(fake_flow(flow_id="missing-connection", include_client_conn=False))
    metadata = next(
        message["metadata"]
        for message in payloads(addon)
        if message["type"] == "flow.metadata"
    )
    assert isinstance(metadata, Mapping)
    assert metadata["session_id"] is None


def payloads(addon: CaptureAddon) -> list[dict[str, object]]:
    result = []
    for message in addon.drain():
        assert isinstance(message, KnownParsedMessage)
        result.append(dict(message.message))
    return result


def acked_sink_drain(
    sink: BoundedMessageSink, limit: int | None = None
) -> list[ParsedMessageResult]:
    batch = sink.drain(limit)
    messages = list(batch.messages)
    sink.acknowledge(batch)
    return messages


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


def test_error_flow_metadata_never_carries_a_synthetic_response_status() -> None:
    addon = CaptureAddon(source_id="test-source", clock=lambda: "now")
    flow = fake_flow(flow_id="errored")
    # No responseheaders / response hook fires — this is the failed-connect
    # shape where the proxy observed only the outbound request before the
    # error. Every metadata emission from here must omit response_status.
    flow.response = None
    addon.requestheaders(flow)
    addon.error(flow)
    messages = payloads(addon)
    metadata_messages = [
        message for message in messages if message["type"] == "flow.metadata"
    ]
    assert metadata_messages, "expected at least one flow.metadata payload"
    for message in metadata_messages:
        metadata = message["metadata"]
        assert isinstance(metadata, Mapping)
        assert "response_status" not in metadata, message


def test_response_status_is_emitted_only_for_real_http_status_codes() -> None:
    addon = CaptureAddon(source_id="test-source", clock=lambda: "now")
    good = fake_flow()
    good.response.status_code = 200
    addon.requestheaders(good)
    addon.responseheaders(good)
    addon.response(good)
    addon.request(good)
    good_messages = payloads(addon)
    metadata_with_status = [
        message["metadata"]
        for message in good_messages
        if message["type"] == "flow.metadata"
    ]
    assert metadata_with_status, "expected at least one flow.metadata payload"
    assert any(
        isinstance(metadata, Mapping) and metadata.get("response_status") == "200"
        for metadata in metadata_with_status
    )

    for bogus in (0, 42, 99, 600, 999):
        addon = CaptureAddon(source_id="test-source", clock=lambda: "now")
        flow = fake_flow(flow_id=f"bogus-{bogus}")
        flow.response.status_code = bogus
        addon.requestheaders(flow)
        addon.responseheaders(flow)
        addon.response(flow)
        addon.request(flow)
        for message in payloads(addon):
            if message["type"] != "flow.metadata":
                continue
            metadata = message["metadata"]
            assert isinstance(metadata, Mapping)
            assert "response_status" not in metadata, bogus

    addon = CaptureAddon(source_id="test-source", clock=lambda: "now")
    missing = fake_flow(flow_id="no-status")
    addon.requestheaders(missing)
    addon.responseheaders(missing)
    addon.response(missing)
    addon.request(missing)
    for message in payloads(addon):
        if message["type"] != "flow.metadata":
            continue
        metadata = message["metadata"]
        assert isinstance(metadata, Mapping)
        assert "response_status" not in metadata


def test_successful_completion_does_not_emit_a_capture_loss_gap() -> None:
    addon = CaptureAddon(source_id="test-source", clock=lambda: "now")
    flow = fake_flow(request_body=b"request", response_body=b"response")

    addon.requestheaders(flow)
    addon.responseheaders(flow)
    addon.response(flow)
    addon.request(flow)

    messages = payloads(addon)
    assert not any(message.get("type") == "stream.gap" for message in messages)
    assert addon.sink.dropped_count == 0


def test_offer_post_sequencer_unlock_fault_does_not_wedge_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = BoundedMessageSink()
    original_lock = sink._sequencer.lock

    class ReleaseFaultLock:
        def __init__(self) -> None:
            self.failed = True

        def acquire(self, blocking: bool = True) -> bool:
            return original_lock.acquire(blocking)

        def release(self) -> None:
            original_lock.release()
            if self.failed:
                self.failed = False
                raise KeyboardInterrupt("post-unlock")

        def __enter__(self) -> object:
            self.acquire()
            return self

        def __exit__(self, *args: object) -> None:
            self.release()

    monkeypatch.setattr(sink._sequencer, "lock", ReleaseFaultLock())
    with pytest.raises(KeyboardInterrupt, match="post-unlock"):
        sink.offer(parse_message({"protocol_version": "1", "type": "future.unlock"}))
    monkeypatch.undo()
    assert sink.offer(parse_message({"protocol_version": "1", "type": "future.after-unlock"}))
    batch = sink.drain()
    assert [
        message.message["type"]
        if isinstance(message, KnownParsedMessage)
        else message.payload["type"]
        for message in batch
    ] == ["future.unlock", "future.after-unlock"]
    sink.acknowledge(batch)


def test_drain_post_sequencer_unlock_fault_recovers_committed_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = BoundedMessageSink()
    assert sink.offer(parse_message({"protocol_version": "1", "type": "future.drain-unlock"}))
    original_lock = sink._sequencer.lock

    class ReleaseFaultLock:
        def __init__(self) -> None:
            self.failed = True

        def acquire(self, blocking: bool = True) -> bool:
            return original_lock.acquire(blocking)

        def release(self) -> None:
            original_lock.release()
            if self.failed:
                self.failed = False
                raise KeyboardInterrupt("post-drain-unlock")

        def __enter__(self) -> object:
            self.acquire()
            return self

        def __exit__(self, *args: object) -> None:
            self.release()

    monkeypatch.setattr(sink._sequencer, "lock", ReleaseFaultLock())
    with pytest.raises(KeyboardInterrupt, match="post-drain-unlock") as raised:
        sink.drain()
    assert getattr(raised.value, "capture_committed", False)
    monkeypatch.undo()
    batch = sink.drain()
    assert len(batch) == 1
    sink.acknowledge(batch)


def test_post_commit_lifecycle_fault_reconciles_sequence_before_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    addon = CaptureAddon(clock=lambda: "now")
    state = addon._ensure_flow(fake_flow())
    original_release = addon.sink._lock.release_if_owned

    def release_then_interrupt(owner: object | None = None) -> bool:
        original_release(owner)
        raise KeyboardInterrupt

    monkeypatch.setattr(addon.sink._lock, "release_if_owned", release_then_interrupt)
    with pytest.raises(KeyboardInterrupt) as raised:
        addon._lifecycle(state, "request_started")
    assert getattr(raised.value, "capture_committed", False)
    assert "request_started" in state.lifecycle_states
    assert addon._sequence == 1
    monkeypatch.undo()
    addon._lifecycle(state, "request_started")
    messages = addon.drain()
    lifecycle = [
        message
        for message in messages
        if isinstance(message, KnownParsedMessage)
        and message.message.get("type") == "flow.lifecycle"
    ]
    assert len(lifecycle) == 1
    assert lifecycle[0].message["sequence"] == "0"


def test_post_commit_body_end_fault_reconciles_terminal_state_before_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    addon = CaptureAddon(clock=lambda: "now")
    state = addon._ensure_flow(fake_flow())
    original_release = addon.sink._lock.release_if_owned
    releases = 0

    def release_body_end_then_interrupt(owner: object | None = None) -> bool:
        nonlocal releases
        releases += 1
        released = original_release(owner)
        if releases == 2:
            raise KeyboardInterrupt
        return released

    monkeypatch.setattr(
        addon.sink._lock,
        "release_if_owned",
        release_body_end_then_interrupt,
    )
    with pytest.raises(KeyboardInterrupt) as raised:
        addon._finish_body(state, "request", b"")
    assert getattr(raised.value, "capture_committed", False)
    assert state.request.ended
    assert not state.request.stream_enabled
    monkeypatch.undo()
    assert addon._finish_body(state, "request", b"")
    messages = addon.drain()
    assert sum(
        isinstance(message, KnownParsedMessage)
        and message.message.get("type") == "body.end"
        for message in messages
    ) == 1


def test_post_commit_body_chunk_fault_reconciles_offset_before_next_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    addon = CaptureAddon(clock=lambda: "now")
    state = addon._ensure_flow(fake_flow())
    original_release = addon.sink._lock.release_if_owned
    releases = 0

    def release_body_chunk_then_interrupt(owner: object | None = None) -> bool:
        nonlocal releases
        releases += 1
        released = original_release(owner)
        if releases == 2:
            raise KeyboardInterrupt
        return released

    monkeypatch.setattr(
        addon.sink._lock,
        "release_if_owned",
        release_body_chunk_then_interrupt,
    )
    with pytest.raises(KeyboardInterrupt) as raised:
        addon._observe_chunk(state, "request", b"abc")
    assert getattr(raised.value, "capture_committed", False)
    assert state.request.chunk_index == 1
    monkeypatch.undo()
    addon._observe_chunk(state, "request", b"def")
    chunks = [
        message
        for message in addon.drain()
        if isinstance(message, KnownParsedMessage)
        and message.message.get("type") == "body.chunk"
    ]
    assert [message.message["chunk_index"] for message in chunks] == ["0", "1"]


@pytest.mark.parametrize("side", ["request", "response"])
def test_committed_body_lifecycle_fault_resumes_accounted_chunk(
    monkeypatch: pytest.MonkeyPatch, side: str
) -> None:
    addon = CaptureAddon(clock=lambda: "now")
    state = addon._ensure_flow(fake_flow())
    body = state.request if side == "request" else state.response
    original_release = addon.sink._lock.release_if_owned
    failed = True

    def release_lifecycle_then_interrupt(owner: object | None = None) -> bool:
        nonlocal failed
        released = original_release(owner)
        if failed:
            failed = False
            raise KeyboardInterrupt
        return released

    monkeypatch.setattr(
        addon.sink._lock,
        "release_if_owned",
        release_lifecycle_then_interrupt,
    )
    with pytest.raises(KeyboardInterrupt) as raised:
        addon._observe_chunk(state, side, b"abc")
    assert getattr(raised.value, "capture_committed", False)
    assert body.total_bytes == 3
    assert len(body.prefix) == 3
    assert body.lifecycle_emitted
    assert body.chunk_index == 0
    monkeypatch.undo()

    addon._observe_chunk(state, side, b"abc")
    addon._observe_chunk(state, side, b"def")
    body_chunks = [
        message
        for message in addon.drain()
        if isinstance(message, KnownParsedMessage)
        and message.message.get("type") == "body.chunk"
        and message.message.get("body_side") == side
    ]
    assert [message.message["chunk_index"] for message in body_chunks] == ["0", "1"]
    assert [message.message["offset_bytes"] for message in body_chunks] == ["0", "3"]
    assert body.total_bytes == 6


def test_precommit_body_lifecycle_fault_retries_without_reaccounting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    addon = CaptureAddon(clock=lambda: "now")
    state = addon._ensure_flow(fake_flow())
    original_offer = addon.sink.offer
    failed = True

    def fail_once(message: object) -> bool:
        nonlocal failed
        if failed:
            failed = False
            raise RuntimeError("injected lifecycle admission failure")
        return original_offer(message)  # type: ignore[arg-type]

    monkeypatch.setattr(addon.sink, "offer", fail_once)
    with pytest.raises(RuntimeError, match="lifecycle admission"):
        addon._observe_chunk(state, "request", b"abc")
    assert state.request.total_bytes == 3
    assert not state.request.lifecycle_emitted
    monkeypatch.undo()
    addon._observe_chunk(state, "request", b"abc")
    assert state.request.total_bytes == 3
    assert state.request.chunk_index == 1


def test_committed_body_lifecycle_fault_flushes_pending_chunk_at_terminal_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    addon = CaptureAddon(clock=lambda: "now")
    state = addon._ensure_flow(fake_flow())
    original_release = addon.sink._lock.release_if_owned
    failed = True

    def release_lifecycle_then_interrupt(owner: object | None = None) -> bool:
        nonlocal failed
        released = original_release(owner)
        if failed:
            failed = False
            raise KeyboardInterrupt
        return released

    monkeypatch.setattr(addon.sink._lock, "release_if_owned", release_lifecycle_then_interrupt)
    with pytest.raises(KeyboardInterrupt):
        addon._observe_chunk(state, "request", b"abc")
    monkeypatch.undo()

    assert addon._finish_body(state, "request", b"abc")
    messages = addon.drain()
    chunks = [
        message
        for message in messages
        if isinstance(message, KnownParsedMessage) and message.message.get("type") == "body.chunk"
    ]
    ends = [
        message
        for message in messages
        if isinstance(message, KnownParsedMessage) and message.message.get("type") == "body.end"
    ]
    assert len(chunks) == 1
    assert chunks[0].message["data_base64"] == base64.b64encode(b"abc").decode()
    assert len(ends) == 1


@pytest.mark.parametrize("side", ["request", "response"])
def test_equal_adjacent_body_chunks_are_distinct_without_retry_receipt(side: str) -> None:
    addon = CaptureAddon(clock=lambda: "now")
    state = addon._ensure_flow(fake_flow())
    first = bytes(bytearray(b"equal"))
    second = bytes(bytearray(b"equal"))
    assert first == second
    assert first is not second
    addon._observe_chunk(state, side, first)
    addon._observe_chunk(state, side, second)
    chunks = [
        message
        for message in addon.drain()
        if isinstance(message, KnownParsedMessage)
        and message.message.get("type") == "body.chunk"
    ]
    assert [message.message["chunk_index"] for message in chunks] == ["0", "1"]
    assert [message.message["offset_bytes"] for message in chunks] == ["0", "5"]


@pytest.mark.parametrize("side", ["request", "response"])
def test_pending_chunk_constructor_fault_precedes_all_body_accounting(
    monkeypatch: pytest.MonkeyPatch, side: str
) -> None:
    addon = CaptureAddon(clock=lambda: "now")
    state = addon._ensure_flow(fake_flow())
    body = state.request if side == "request" else state.response

    def fail_constructor(*args: object, **kwargs: object) -> object:
        raise MemoryError("pending receipt construction")

    monkeypatch.setattr(adapter_module, "_PendingChunk", fail_constructor)
    with pytest.raises(MemoryError, match="pending receipt construction"):
        addon._observe_chunk(state, side, b"abc")
    assert body.total_bytes == 0
    assert body.chunk_index == 0
    assert body.prefix == bytearray()
    assert body.pending_chunk is None
    assert addon._captured_prefix_bytes == 0
    monkeypatch.undo()

    addon._observe_chunk(state, side, b"abc")
    assert body.total_bytes == 3
    assert body.chunk_index == 1
    assert bytes(body.prefix) == b"abc"


@pytest.mark.parametrize("side", ["request", "response"])
def test_equal_retry_chunk_uses_bytes_identity_not_reusable_address(
    monkeypatch: pytest.MonkeyPatch, side: str
) -> None:
    addon = CaptureAddon(clock=lambda: "now")
    state = addon._ensure_flow(fake_flow())
    body = state.request if side == "request" else state.response
    first = bytes(bytearray(b"equal"))
    second = bytes(bytearray(b"equal"))
    original_publish = addon._publish_body_chunk
    failed = True

    def fail_receipt(body_arg: object, pending: object, retryable: bool) -> None:
        nonlocal failed
        if failed:
            failed = False
            raise KeyboardInterrupt("retry receipt")
        original_publish(body_arg, pending, retryable)  # type: ignore[arg-type]

    monkeypatch.setattr(addon, "_publish_body_chunk", fail_receipt)
    with pytest.raises(KeyboardInterrupt, match="retry receipt"):
        addon._observe_chunk(state, side, first)
    monkeypatch.undo()
    assert body.retry_chunk is not None
    addon._observe_chunk(state, side, second)
    addon._finish_body(state, side, None)
    chunks = [
        message
        for message in addon.drain()
        if isinstance(message, KnownParsedMessage)
        and message.message.get("type") == "body.chunk"
        and message.message.get("body_side") == side
    ]
    assert [message.message["chunk_index"] for message in chunks] == ["0", "1"]
    assert [message.message["offset_bytes"] for message in chunks] == ["0", "5"]


def test_retained_flow_does_not_retain_released_capture_state() -> None:
    addon = CaptureAddon(clock=lambda: "now")
    flow = fake_flow()
    addon.requestheaders(flow)
    flow.request.stream(b"secret-prefix")
    state = addon._flows[flow.id]
    state_ref = weakref.ref(state)
    callback = flow.request.stream
    assert bytes(state.request.prefix) == b"secret-prefix"

    addon._complete_active(state)
    assert addon.counters["active_prefix_bytes"] == 0
    assert state.request.prefix == bytearray()
    assert state.identity == {}
    del state
    gc.collect()
    assert state_ref() is None
    assert callback(b"still-forwarded") == b"still-forwarded"


@pytest.mark.parametrize("side", ["request", "response"])
def test_body_chunk_publication_fault_keeps_receipt_and_equal_next_chunk(
    monkeypatch: pytest.MonkeyPatch, side: str
) -> None:
    addon = CaptureAddon(clock=lambda: "now")
    state = addon._ensure_flow(fake_flow())
    first = bytes(bytearray(b"equal"))
    second = bytes(bytearray(b"equal"))
    original_publish = addon._publish_body_chunk
    failed = True

    def publish_then_interrupt(body: object, pending: object, retryable: bool) -> None:
        nonlocal failed
        if failed:
            failed = False
            raise KeyboardInterrupt("body receipt publication")
        original_publish(body, pending, retryable)  # type: ignore[arg-type]

    monkeypatch.setattr(addon, "_publish_body_chunk", publish_then_interrupt)
    with pytest.raises(KeyboardInterrupt, match="body receipt publication"):
        addon._observe_chunk(state, side, first)
    monkeypatch.undo()
    addon._observe_chunk(state, side, first)
    addon._observe_chunk(state, side, second)
    chunks = [
        message
        for message in addon.drain()
        if isinstance(message, KnownParsedMessage)
        and message.message.get("type") == "body.chunk"
    ]
    assert [message.message["chunk_index"] for message in chunks] == ["0", "1"]
    assert [message.message["offset_bytes"] for message in chunks] == ["0", "5"]


@pytest.mark.parametrize("side", ["request", "response"])
def test_body_lifecycle_and_end_publication_faults_are_resumable(
    monkeypatch: pytest.MonkeyPatch, side: str
) -> None:
    addon = CaptureAddon(clock=lambda: "now")
    state = addon._ensure_flow(fake_flow())
    original_lifecycle = addon._publish_body_lifecycle
    lifecycle_failed = True

    def lifecycle_then_interrupt(body: object) -> None:
        nonlocal lifecycle_failed
        if lifecycle_failed:
            lifecycle_failed = False
            raise KeyboardInterrupt("body lifecycle publication")
        original_lifecycle(body)  # type: ignore[arg-type]

    monkeypatch.setattr(addon, "_publish_body_lifecycle", lifecycle_then_interrupt)
    chunk = bytes(bytearray(b"abc"))
    with pytest.raises(KeyboardInterrupt, match="body lifecycle publication"):
        addon._observe_chunk(state, side, chunk)
    monkeypatch.undo()
    addon._observe_chunk(state, side, chunk)

    original_end = addon._publish_body_end
    end_failed = True

    def end_then_interrupt(body: object) -> None:
        nonlocal end_failed
        if end_failed:
            end_failed = False
            raise KeyboardInterrupt("body end publication")
        original_end(body)  # type: ignore[arg-type]

    monkeypatch.setattr(addon, "_publish_body_end", end_then_interrupt)
    with pytest.raises(KeyboardInterrupt, match="body end publication"):
        assert not addon._finish_body(state, side, None)
    monkeypatch.undo()
    assert addon._finish_body(state, side, None)
    messages = addon.drain()
    assert sum(
        isinstance(message, KnownParsedMessage)
        and message.message.get("type") == "body.chunk"
        for message in messages
    ) == 1
    assert sum(
        isinstance(message, KnownParsedMessage)
        and message.message.get("type") == "body.end"
        for message in messages
    ) == 1


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


def test_body_prefix_seals_after_budget_skip_and_reclamation() -> None:
    config = CaptureConfig(
        source_id="source",
        max_body_prefix_bytes=64,
        max_in_memory_bytes=4_096,
    )
    addon = CaptureAddon(config=config, clock=lambda: "now")
    flow = fake_flow()
    addon.requestheaders(flow)
    addon.responseheaders(flow)
    metadata_weight = addon._active_metadata_bytes
    addon.max_in_memory_bytes = metadata_weight + 5
    assert flow.response.stream(b"FIRST") == b"FIRST"
    assert flow.response.stream(b"SECOND") == b"SECOND"
    addon.max_in_memory_bytes = metadata_weight + 1024
    assert flow.response.stream(b"THIRD") == b"THIRD"
    addon.response(flow)

    body_end = [
        message
        for message in payloads(addon)
        if message.get("type") == "body.end" and message.get("body_side") == "response"
    ][-1]
    body = body_end["body"]
    assert isinstance(body, Mapping)
    assert body["captured_bytes"] == "5"
    assert base64.b64decode(body["data"]) == b"FIRST"
    assert body_end["total_bytes"] == "16"


def test_body_prefix_partial_chunk_is_sealed_as_true_prefix() -> None:
    config = CaptureConfig(
        source_id="source",
        max_body_prefix_bytes=64,
        max_in_memory_bytes=4_096,
    )
    addon = CaptureAddon(config=config, clock=lambda: "now")
    flow = fake_flow()
    addon.requestheaders(flow)
    addon.responseheaders(flow)
    addon.max_in_memory_bytes = addon._active_metadata_bytes + 6
    flow.response.stream(b"FIRST")
    flow.response.stream(b"SECOND")
    addon.max_in_memory_bytes += 1024
    addon.response(flow)

    body_end = [
        message
        for message in payloads(addon)
        if message.get("type") == "body.end" and message.get("body_side") == "response"
    ][-1]
    body = body_end["body"]
    assert isinstance(body, Mapping)
    assert body["captured_bytes"] == "6"
    assert base64.b64decode(body["data"]) == b"FIRSTS"
    assert body_end["total_bytes"] == "11"


def test_capture_uint64_boundaries_fail_before_body_or_lifecycle_mutation() -> None:
    addon = CaptureAddon(clock=lambda: "now")
    flow = fake_flow()
    addon.requestheaders(flow)
    state = addon._flows[flow.id]

    addon._sequence = MAX_U64
    with pytest.raises(OverflowError, match="lifecycle sequence exhausted"):
        addon._lifecycle(state, "synthetic")
    assert addon._sequence == MAX_U64
    assert "synthetic" not in state.lifecycle_states

    addon._sequence = 0
    state.request.total_bytes = MAX_U64
    with pytest.raises(OverflowError, match="body byte offset exhausted"):
        addon._observe_chunk(state, "request", b"x")
    assert state.request.total_bytes == MAX_U64
    assert not state.request.observed
    state.request.total_bytes = 0
    state.request.chunk_index = MAX_U64
    with pytest.raises(OverflowError, match="body chunk index exhausted"):
        addon._observe_chunk(state, "request", b"")
    assert state.request.chunk_index == MAX_U64
    assert not state.request.observed


def test_stream_body_exhaustion_returns_original_chunk_and_discards_flow() -> None:
    addon = CaptureAddon(clock=lambda: "now")
    flow = fake_flow()
    addon.requestheaders(flow)
    state = addon._flows[flow.id]
    state.request.total_bytes = MAX_U64
    chunk = b"forwarded-without-capture"

    assert flow.request.stream(chunk) is chunk
    assert state.discarded
    assert state.tombstone
    assert flow.id not in addon._flows
    assert addon.counters["active_flows"] == 0
    assert addon.sink.dropped_count == 1
    assert flow.request.stream(chunk) is chunk
    assert addon.sink.dropped_count == 1


def test_flow_completion_exhaustion_discards_before_sticky_completion() -> None:
    addon = CaptureAddon(clock=lambda: "now")
    flow = fake_flow()
    addon.requestheaders(flow)
    state = addon._flows[flow.id]
    state.request.ended = True
    state.terminal_observed = True
    addon._sequence = MAX_U64

    addon._finalize(state)

    assert not state.completed
    assert state.discarded
    assert state.tombstone
    assert flow.id not in addon._flows
    assert addon.sink.dropped_count == 1

def test_body_end_failure_does_not_tombstone_body_before_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    addon = CaptureAddon(clock=lambda: "now")
    flow = fake_flow()
    addon.requestheaders(flow)
    state = addon._flows[flow.id]
    original_send = addon._send

    def fail_body_end(message: dict[str, object]) -> None:
        if message.get("type") == "body.end":
            raise ValueError("injected body-end validation failure")
        original_send(message)

    monkeypatch.setattr(addon, "_send", fail_body_end)
    with pytest.raises(ValueError, match="body-end validation failure"):
        addon._finish_body(state, "request", b"body")
    assert not state.request.ended
    assert state.request.stream_enabled
    monkeypatch.undo()
    addon._finish_body(state, "request", None)
    assert state.request.ended
    assert any(
        message["type"] == "body.end"
        for message in payloads(addon)
        if message.get("body_side") == "request"
    )


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


def test_source_hello_retries_after_full_sink_rejection() -> None:
    sink = BoundedMessageSink(max_pending=1)
    assert sink.offer(parse_message({"protocol_version": "1", "type": "future.blocker"}))
    addon = CaptureAddon(sink=sink, clock=lambda: "now")

    addon._announce_source()
    assert not addon._source_announced
    first = addon.drain()
    assert not any(
        (message.message if isinstance(message, KnownParsedMessage) else message.payload)["type"]
        == "source.hello"
        for message in first
    )

    addon._announce_source()
    assert addon._source_announced
    second = addon.drain()
    assert any(
        (message.message if isinstance(message, KnownParsedMessage) else message.payload)["type"]
        == "source.hello"
        for message in second
    )


def test_source_hello_retries_after_injected_send_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    addon = CaptureAddon(clock=lambda: "now")
    original_send = addon._send
    failed = True

    def fail_once(message: dict[str, object]) -> bool:
        nonlocal failed
        if failed:
            failed = False
            raise RuntimeError("injected hello failure")
        return original_send(message)

    monkeypatch.setattr(addon, "_send", fail_once)
    with pytest.raises(RuntimeError, match="injected hello failure"):
        addon._announce_source()
    assert not addon._source_announced
    addon._announce_source()
    assert addon._source_announced


def test_source_hello_postcommit_cleanup_failure_does_not_duplicate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = BoundedMessageSink()
    addon = CaptureAddon(sink=sink, clock=lambda: "now")
    original_release = sink._lock.release_if_owned
    releases = 0

    def release_then_interrupt(owner: object | None = None) -> bool:
        nonlocal releases
        releases += 1
        released = original_release(owner)
        if releases == 1:
            raise KeyboardInterrupt
        return released

    monkeypatch.setattr(sink._lock, "release_if_owned", release_then_interrupt)
    with pytest.raises(KeyboardInterrupt):
        addon._announce_source()
    assert addon._source_announced
    monkeypatch.undo()

    messages = addon.drain()
    assert [
        (message.message if isinstance(message, KnownParsedMessage) else message.payload)["type"]
        for message in messages
    ] == ["source.hello"]


def test_source_hello_precommit_cleanup_failure_remains_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = BoundedMessageSink()
    addon = CaptureAddon(sink=sink, clock=lambda: "now")
    def fail_before_admission(_message: object) -> bool:
        raise RuntimeError("injected precommit failure")

    monkeypatch.setattr(sink, "offer", fail_before_admission)
    with pytest.raises(RuntimeError, match="injected precommit failure"):
        addon._announce_source()
    assert not addon._source_announced
    monkeypatch.undo()
    assert sink.pending_count == 0

    addon._announce_source()
    assert addon._source_announced
    messages = addon.drain()
    assert len(messages) == 1
    assert isinstance(messages[0], KnownParsedMessage)
    assert messages[0].message["type"] == "source.hello"


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


def test_loss_admission_is_nonblocking_while_position_lock_is_held() -> None:
    sink = BoundedMessageSink(max_pending=1)
    assert sink.offer(parse_message({"protocol_version": "1", "type": "future.first"}))
    held = threading.Event()
    release = threading.Event()

    def hold_sequencer() -> None:
        with sink._sequencer.lock:
            held.set()
            release.wait(timeout=2)

    holder = threading.Thread(target=hold_sequencer)
    holder.start()
    assert held.wait(timeout=1)
    started = time.perf_counter()
    assert sink.record_loss()
    assert not sink.offer(parse_message({"protocol_version": "1", "type": "future.full"}))
    assert time.perf_counter() - started < 0.05
    release.set()
    holder.join(timeout=1)
    assert not holder.is_alive()
    assert sink.dropped_count == 2
    messages = sink.drain()
    assert [
        (message.message if isinstance(message, KnownParsedMessage) else message.payload)["type"]
        for message in messages
    ] == ["future.first", "stream.gap"]
    gap = messages[-1]
    assert isinstance(gap, KnownParsedMessage)
    assert gap.message["dropped_count"] == "2"


def test_loss_admission_is_nonblocking_while_sequencer_is_held() -> None:
    sink = BoundedMessageSink(max_pending=1)
    assert sink.offer(parse_message({"protocol_version": "1", "type": "future.first"}))

    sink._sequencer.lock.acquire()
    try:
        started = time.perf_counter()
        assert sink.record_loss()
        assert not sink.offer(
            parse_message({"protocol_version": "1", "type": "future.full"})
        )
        assert time.perf_counter() - started < 0.05
    finally:
        sink._sequencer.lock.release()

    assert sink.dropped_count == 2
    drained = sink.drain()
    assert [
        (item.message if isinstance(item, KnownParsedMessage) else item.payload)["type"]
        for item in drained
    ] == ["future.first", "stream.gap"]
    gap = drained[-1]
    assert isinstance(gap, KnownParsedMessage)
    assert gap.message["dropped_count"] == "2"


def test_contended_loss_ticket_is_folded_once_after_two_public_admissions() -> None:
    sink = BoundedMessageSink(max_pending=1)
    entered = threading.Event()
    release = threading.Event()

    def hold_sequencer() -> None:
        with sink._sequencer.lock:
            entered.set()
            release.wait(timeout=1)

    holder = threading.Thread(target=hold_sequencer)
    holder.start()
    assert entered.wait(timeout=1)
    results: list[bool] = []
    workers = [
        threading.Thread(target=lambda: results.append(sink.record_loss()))
        for _ in range(2)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=1)
    release.set()
    holder.join(timeout=1)
    assert results == [True, True]
    assert sink.dropped_count == 2
    gaps = sink.drain()
    assert len(gaps) == 1
    assert isinstance(gaps[0], KnownParsedMessage)
    assert gaps[0].message["dropped_count"] == "2"


def test_post_append_fault_reports_committed_offer_without_duplicate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = BoundedMessageSink()
    original_append = sink._sequencer.append_locked

    def append_then_interrupt(*args: object, **kwargs: object) -> bool:
        original_append(*args, **kwargs)
        raise KeyboardInterrupt

    monkeypatch.setattr(sink._sequencer, "append_locked", append_then_interrupt)
    with pytest.raises(KeyboardInterrupt) as raised:
        sink.offer(parse_message({"protocol_version": "1", "type": "future.committed"}))
    assert getattr(raised.value, "capture_committed", False)
    assert sink.accepted_count == 1
    assert sink.pending_count == 1
    monkeypatch.undo()
    drained = sink.drain()
    assert len(drained) == 1
    sink.acknowledge(drained)
    assert acked_sink_drain(sink) == []


def test_drain_cleanup_fault_keeps_committed_snapshot_visible_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = BoundedMessageSink()
    assert sink.offer(parse_message({"protocol_version": "1", "type": "future.drain"}))
    original_release = sink._lock.release_if_owned

    def release_then_interrupt(owner: object | None = None) -> bool:
        original_release(owner)
        raise KeyboardInterrupt

    monkeypatch.setattr(sink._lock, "release_if_owned", release_then_interrupt)
    batch = sink.drain()
    assert len(batch) == 1
    monkeypatch.undo()
    assert sink.pending_count == 0
    assert len(sink.drain()) == 1
    sink.acknowledge(batch)
    assert acked_sink_drain(sink) == []


def test_drain_post_commit_fault_keeps_durable_batch_for_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = BoundedMessageSink()
    assert sink.offer(parse_message({"protocol_version": "1", "type": "future.drain-fault"}))
    original_drain = sink._sequencer.drain

    def drain_then_interrupt(*args: object, **kwargs: object) -> list[ParsedMessageResult]:
        original_drain(*args, **kwargs)
        raise KeyboardInterrupt

    monkeypatch.setattr(sink._sequencer, "drain", drain_then_interrupt)
    with pytest.raises(KeyboardInterrupt) as raised:
        sink.drain()
    assert getattr(raised.value, "capture_committed", False)
    assert sink.pending_count == 0
    monkeypatch.undo()
    recovered = sink.drain()
    assert len(recovered) == 1
    sink.acknowledge(recovered)
    assert acked_sink_drain(sink) == []


def test_drain_publication_fault_leaves_queue_and_batch_uncommitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = BoundedMessageSink()
    assert sink.offer(parse_message({"protocol_version": "1", "type": "future.publish"}))
    original_batch = sequencer_module.DrainBatch

    def fail_publication(*args: object, **kwargs: object) -> object:
        raise MemoryError("injected batch publication failure")

    monkeypatch.setattr(sequencer_module, "DrainBatch", fail_publication)
    with pytest.raises(MemoryError, match="batch publication"):
        sink.drain()
    assert sink.pending_count == 1
    monkeypatch.setattr(sequencer_module, "DrainBatch", original_batch)
    batch = sink.drain()
    assert len(batch) == 1
    sink.acknowledge(batch)


def test_ack_fault_after_commit_is_idempotently_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = BoundedMessageSink()
    assert sink.offer(parse_message({"protocol_version": "1", "type": "future.ack"}))
    batch = sink.drain()
    original_ack = sink._sequencer.acknowledge

    def ack_then_interrupt(candidate: object) -> None:
        original_ack(candidate)  # type: ignore[arg-type]
        raise KeyboardInterrupt

    monkeypatch.setattr(sink._sequencer, "acknowledge", ack_then_interrupt)
    with pytest.raises(KeyboardInterrupt):
        sink.acknowledge(batch)
    monkeypatch.undo()
    sink.acknowledge(batch)
    assert acked_sink_drain(sink) == []


def test_stale_batch_object_cannot_acknowledge_a_new_generation() -> None:
    sink = BoundedMessageSink(max_pending=2)
    assert sink.offer(parse_message({"protocol_version": "1", "type": "future.first"}))
    first = sink.drain()
    sink.acknowledge(first)
    assert sink.offer(parse_message({"protocol_version": "1", "type": "future.second"}))
    second = sink.drain()
    assert second.token == first.token + 1
    with pytest.raises(ValueError, match="not current"):
        sink.acknowledge(first)
    assert sink.drain() is second
    sink.acknowledge(second)


def test_drain_recovery_after_caller_return_fault_reuses_same_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = BoundedMessageSink()
    assert sink.offer(parse_message({"protocol_version": "1", "type": "future.return"}))
    original_drain = sink.drain

    def return_then_interrupt(*args: object, **kwargs: object) -> object:
        original_drain(*args, **kwargs)
        raise KeyboardInterrupt

    monkeypatch.setattr(sink, "drain", return_then_interrupt)
    with pytest.raises(KeyboardInterrupt):
        sink.drain()
    monkeypatch.undo()
    recovered = sink.drain()
    assert recovered.token == 1
    sink.acknowledge(recovered)


def test_drain_baseexception_restores_one_atomic_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = BoundedMessageSink(max_pending=2)
    assert sink.offer(parse_message({"protocol_version": "1", "type": "future.atomic"}))
    before = (sink.pending_count, sink.accepted_count, sink.dropped_count, sink._memory_bytes)

    def interrupt(_message: ParsedMessageResult, _position: int) -> ParsedMessageResult:
        raise KeyboardInterrupt

    monkeypatch.setattr(sink_module, "_with_delivery_position", interrupt)
    with pytest.raises(KeyboardInterrupt):
        sink.drain()
    assert (
        sink.pending_count,
        sink.accepted_count,
        sink.dropped_count,
        sink._memory_bytes,
    ) == before
    monkeypatch.undo()
    batch = sink.drain()
    assert len(batch) == 1
    sink.acknowledge(batch)
    assert sink._memory_bytes == 0


def test_unacknowledged_drain_batch_remains_inside_admission_budgets() -> None:
    sink = BoundedMessageSink(max_pending=2, max_body_bytes=8, max_memory_bytes=4_096)
    message = parse_message(
        {
            "protocol_version": "1",
            "type": "body.end",
            "flow_id": "budget",
            "body_side": "response",
            "total_bytes": "8",
            "body": {
                "state": "captured",
                "size_bytes": "8",
                "encoding": "base64",
                "data": base64.b64encode(b"12345678").decode(),
            },
        }
    )
    assert sink.offer(message)
    retained_memory = sink._memory_bytes
    batch = sink.drain()
    assert len(batch) == 1
    assert sink._memory_bytes == retained_memory
    assert sink._body_bytes == 8
    assert not sink.offer(message)
    assert sink.pending_count == 0
    sink.acknowledge(batch)
    assert sink._memory_bytes == 0
    assert sink._body_bytes == 0
    assert sink.offer(message)

    count_sink = BoundedMessageSink(max_pending=1, max_body_bytes=8, max_memory_bytes=4_096)
    assert count_sink.offer(message)
    count_batch = count_sink.drain()
    assert not count_sink.offer(message)
    count_sink.acknowledge(count_batch)


def test_sink_queue_and_body_drops_are_in_band_and_delivery_positioned() -> None:
    sink = BoundedMessageSink(max_pending=1, max_body_bytes=1)
    assert sink.offer(parse_message({"protocol_version": "1", "type": "future.first"}))
    assert not sink.offer(parse_message({"protocol_version": "1", "type": "future.queue"}))
    acked_sink_drain(sink)
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
    sink.acknowledge(messages)
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


def test_active_metadata_and_prefix_share_one_memory_budget() -> None:
    config = CaptureConfig(
        source_id="source",
        max_body_prefix_bytes=1_024,
        max_in_memory_bytes=1_024,
    )
    addon = CaptureAddon(config=config, clock=lambda: "now")
    flow = fake_flow()
    addon.requestheaders(flow)
    addon.responseheaders(flow)
    addon.max_in_memory_bytes = addon._active_metadata_bytes + 6
    flow.response.stream(b"123456789")
    counters = addon.counters
    assert counters["active_metadata_bytes"] + counters["active_prefix_bytes"] <= (
        addon.max_in_memory_bytes
    )
    assert counters["active_prefix_bytes"] == 6


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


def test_reentrant_callback_drain_is_single_flight_and_keeps_suffix_order() -> None:
    observed: list[str] = []
    reentrant_results: list[list[ParsedMessageResult]] = []
    addon: CaptureAddon

    def emit(message: ParsedMessageResult) -> None:
        payload = message.message if isinstance(message, KnownParsedMessage) else message.payload
        message_type = str(payload["type"])
        observed.append(message_type)
        if message_type == "future.one":
            addon.sink.offer(parse_message({"protocol_version": "1", "type": "future.three"}))
            reentrant_results.append(addon.drain())

    addon = CaptureAddon(emit=emit, clock=lambda: "now")
    assert addon.sink.offer(parse_message({"protocol_version": "1", "type": "future.one"}))
    assert addon.sink.offer(parse_message({"protocol_version": "1", "type": "future.two"}))
    first = addon.drain()
    assert len(first) == 2
    assert reentrant_results == [[]]
    assert observed == ["future.one", "future.two"]
    addon.drain()
    assert observed == ["future.one", "future.two", "future.three"]


def test_callback_failure_attempts_entire_batch_without_stranding_suffix() -> None:
    observed: list[str] = []
    failed = True

    def emit(message: ParsedMessageResult) -> None:
        payload = message.message if isinstance(message, KnownParsedMessage) else message.payload
        message_type = str(payload["type"])
        observed.append(message_type)
        nonlocal failed
        if message_type == "future.one" and failed:
            failed = False
            raise RuntimeError("injected callback failure")

    addon = CaptureAddon(emit=emit, clock=lambda: "now")
    assert addon.sink.offer(parse_message({"protocol_version": "1", "type": "future.one"}))
    assert addon.sink.offer(parse_message({"protocol_version": "1", "type": "future.two"}))
    with pytest.raises(RuntimeError, match="injected callback failure"):
        addon.drain()
    assert observed == ["future.one"]
    retry = addon.drain()
    assert len(retry) == 2
    assert observed == ["future.one", "future.one", "future.two"]


def test_callback_progress_receipt_survives_publication_fault(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[str] = []
    failed = True

    def emit(message: ParsedMessageResult) -> None:
        payload = message.message if isinstance(message, KnownParsedMessage) else message.payload
        observed.append(str(payload["type"]))

    addon = CaptureAddon(emit=emit, clock=lambda: "now")
    assert addon.sink.offer(parse_message({"protocol_version": "1", "type": "future.one"}))
    assert addon.sink.offer(parse_message({"protocol_version": "1", "type": "future.two"}))
    original_publish = addon._publish_callback_progress

    def fail_once(batch: object, next_index: int) -> None:
        nonlocal failed
        if failed:
            failed = False
            raise KeyboardInterrupt("callback receipt publication")
        original_publish(batch, next_index)  # type: ignore[arg-type]

    monkeypatch.setattr(addon, "_publish_callback_progress", fail_once)
    with pytest.raises(KeyboardInterrupt, match="callback receipt publication"):
        addon.drain()
    monkeypatch.undo()
    assert addon.drain()
    assert observed == ["future.one", "future.two"]


def test_capture_drain_returns_stable_result_after_post_ack_fault(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    addon = CaptureAddon(clock=lambda: "now")
    assert addon.sink.offer(parse_message({"protocol_version": "1", "type": "future.ack-return"}))
    original_ack = addon.sink.acknowledge

    def ack_then_interrupt(batch: object) -> None:
        original_ack(batch)  # type: ignore[arg-type]
        raise KeyboardInterrupt("after acknowledgement")

    monkeypatch.setattr(addon.sink, "acknowledge", ack_then_interrupt)
    result = addon.drain()
    assert len(result) == 1
    payload = result[0].message if isinstance(result[0], KnownParsedMessage) else result[0].payload
    assert payload["type"] == "future.ack-return"


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
    sink.acknowledge(first_drain)
    assert acked_sink_drain(sink) == []


def test_contended_offer_does_no_payload_work(monkeypatch: pytest.MonkeyPatch) -> None:
    sink = BoundedMessageSink()
    message = {"protocol_version": "1", "type": "future.contended"}

    def fail(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("payload work ran on the contended producer path")

    monkeypatch.setattr(sink_module, "require_parsed_message", fail)
    monkeypatch.setattr(sink_module, "parse_message", fail)
    monkeypatch.setattr(sink_module, "_message_body_bytes", fail)
    monkeypatch.setattr(sink_module, "_message_weight", fail)
    sink._reservation_lock.acquire()
    try:
        assert not sink.offer(message)
    finally:
        sink._reservation_lock.release()
    assert sink.dropped_count == 1


def test_full_offer_rejects_hostile_mapping_before_payload_traversal() -> None:
    sink = BoundedMessageSink(max_pending=1)
    assert sink.offer(parse_message({"protocol_version": "1", "type": "future.full"}))

    class HostileMapping(dict[str, object]):
        def items(self) -> Iterator[tuple[str, object]]:
            raise AssertionError("full queue must not inspect the payload")

    assert not sink.offer(HostileMapping())
    assert sink.pending_count == 1
    assert sink.dropped_count == 1


def test_loss_ranges_stay_bounded_for_25000_contiguous_drops() -> None:
    sink = BoundedMessageSink(max_pending=1)
    assert sink.offer(parse_message({"protocol_version": "1", "type": "future.first"}))
    message = parse_message({"protocol_version": "1", "type": "future.drop"})
    for _ in range(25_000):
        assert not sink.offer(message)
    assert sink.dropped_count == 25_000
    assert sink.loss_range_count == 1
    messages = sink.drain()
    gaps = [
        message
        for message in messages
        if isinstance(message, KnownParsedMessage) and message.message["type"] == "stream.gap"
    ]
    assert len(gaps) == 1
    assert gaps[0].message["dropped_count"] == "25000"
    assert sink.loss_range_count == 0


def test_noncontiguous_losses_preserve_every_accepted_entry() -> None:
    sink = BoundedMessageSink(max_pending=512)
    message = parse_message({"protocol_version": "1", "type": "future.keep"})
    for _ in range(300):
        assert sink.offer(message)
        sink.record_loss()
    assert sink.loss_range_count == 300
    messages = sink.drain()
    retained = [
        message
        for message in messages
        if (message.message if isinstance(message, KnownParsedMessage) else message.payload)[
            "type"
        ] == "future.keep"
    ]
    gaps = [
        message
        for message in messages
        if isinstance(message, KnownParsedMessage) and message.message["type"] == "stream.gap"
    ]
    assert len(retained) == 300
    assert len(gaps) == 300
    assert sink.dropped_count == 300


def test_prepared_offer_has_no_payload_work_when_queue_lock_loses_between_stages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = BoundedMessageSink()
    message = parse_message({"protocol_version": "1", "type": "future.prepared"})
    def fail(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("payload helper ran after prepared admission")

    monkeypatch.setattr(sink_module, "require_parsed_message", fail)
    monkeypatch.setattr(sink_module, "_message_body_bytes", fail)
    monkeypatch.setattr(sink_module, "_message_weight", fail)
    sink._lock.acquire()
    result: list[bool] = []
    producer = threading.Thread(
        target=lambda: result.append(sink.offer(message))
    )
    producer.start()
    producer.join(timeout=1)
    sink._lock.release()
    assert not producer.is_alive()
    assert result == [False]


def test_admission_preparation_failure_releases_reservation_without_position(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = BoundedMessageSink()
    message = parse_message({"protocol_version": "1", "type": "future.failure"})

    def fail(_message: ParsedMessageResult) -> ParsedMessageResult:
        raise RuntimeError("injected preparation failure")

    monkeypatch.setattr(sink_module, "require_parsed_message", fail)
    with pytest.raises(RuntimeError, match="injected preparation failure"):
        sink.offer(message)
    assert sink.pending_count == 0
    assert sink.dropped_count == 0
    assert sink._next_position_value == 1


def test_pending_marker_commit_survives_allocate_fault_without_phantom_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = BoundedMessageSink()
    original = sink._sequencer._allocate_position

    def fail(_state: object) -> tuple[object, int]:
        raise KeyboardInterrupt

    monkeypatch.setattr(sink._sequencer, "_allocate_position", fail)
    with pytest.raises(KeyboardInterrupt):
        sink.offer(parse_message({"protocol_version": "1", "type": "future.fault"}))
    monkeypatch.setattr(sink._sequencer, "_allocate_position", original)
    assert sink.offer(parse_message({"protocol_version": "1", "type": "future.retry"}))
    drained = sink.drain()
    assert len(drained) == 1
    payload = (
        drained[0].payload
        if not isinstance(drained[0], KnownParsedMessage)
        else drained[0].message
    )
    assert payload["type"] == "future.retry"
    assert payload["delivery_position"] == "1"
    assert sink.dropped_count == 0


def test_pending_loss_fold_fault_retries_snapshot_without_double_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = BoundedMessageSink()
    assert sink._sequencer.note_loss_without_lock()
    original = sink._sequencer._append_loss_count
    failed = True

    def fail_once(state: object, count: int) -> object:
        nonlocal failed
        if failed:
            failed = False
            raise MemoryError("injected loss fold failure")
        return original(state, count)  # type: ignore[arg-type]

    monkeypatch.setattr(sink._sequencer, "_append_loss_count", fail_once)
    with pytest.raises(MemoryError, match="injected loss fold failure"):
        sink.offer(parse_message({"protocol_version": "1", "type": "future.fold-fault"}))
    monkeypatch.undo()
    assert sink.offer(parse_message({"protocol_version": "1", "type": "future.after-fold"}))
    drained = sink.drain()
    assert [
        (
            message.message
            if isinstance(message, KnownParsedMessage)
            else message.payload
        )["type"]
        for message in drained
    ] == ["stream.gap", "future.after-fold"]
    gap = drained[0]
    assert isinstance(gap, KnownParsedMessage)
    assert gap.message["dropped_count"] == "1"


def test_pending_reader_snapshot_fault_does_not_consume_loss_ticket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = BoundedMessageSink()
    assert sink._sequencer.note_loss_without_lock()
    original_replace = sequencer_module.replace
    failed = True

    def fail_snapshot(*args: object, **kwargs: object) -> object:
        nonlocal failed
        if failed:
            failed = False
            raise MemoryError("injected snapshot publication failure")
        return original_replace(*args, **kwargs)

    monkeypatch.setattr(sequencer_module, "replace", fail_snapshot)
    with pytest.raises(MemoryError, match="snapshot publication"):
        sink.offer(parse_message({"protocol_version": "1", "type": "future.snapshot-fault"}))
    monkeypatch.undo()
    assert sink.offer(parse_message({"protocol_version": "1", "type": "future.snapshot-retry"}))
    drained = acked_sink_drain(sink)
    assert [
        (
            message.message
            if isinstance(message, KnownParsedMessage)
            else message.payload
        )["type"]
        for message in drained
    ] == ["stream.gap", "future.snapshot-retry"]


def test_reader_watermark_snapshot_does_not_consume_producer_counter() -> None:
    sink = BoundedMessageSink()
    assert sink._sequencer.note_loss_without_lock()
    before = sink._sequencer._producer_watermark()
    sink._sequencer.flush_for_read()
    after = sink._sequencer._producer_watermark()
    assert after == before == 1
    assert sink.dropped_count == 1
    assert sink.loss_range_count == 1


def test_reservation_acquire_baseexception_after_ownership_is_recoverable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = BoundedMessageSink()
    original_acquire = sink._reservation_lock.acquire

    def interrupt_after_acquire(
        blocking: bool = True, owner: object | None = None
    ) -> bool:
        acquired = original_acquire(blocking, owner)
        if acquired:
            raise KeyboardInterrupt
        return acquired

    monkeypatch.setattr(sink._reservation_lock, "acquire", interrupt_after_acquire)
    with pytest.raises(KeyboardInterrupt):
        sink.offer(parse_message({"protocol_version": "1", "type": "future.acquire"}))
    monkeypatch.undo()

    assert sink.pending_count == 0
    assert sink.dropped_count == 0
    assert sink._reservation_lock.acquire(False)
    sink._reservation_lock.release()
    assert sink.offer(parse_message({"protocol_version": "1", "type": "future.after"}))


def test_interrupted_acquire_cleanup_cannot_release_another_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = BoundedMessageSink()
    holder = sink_module._ReservationLease(sink._reservation_lock)
    waiter = sink_module._ReservationLease(sink._reservation_lock)
    assert holder.acquire()
    original_acquire = sink._reservation_lock.acquire

    def interrupt_before_acquire(
        blocking: bool = False, owner: object | None = None
    ) -> bool:
        raise KeyboardInterrupt

    monkeypatch.setattr(sink._reservation_lock, "acquire", interrupt_before_acquire)
    with pytest.raises(KeyboardInterrupt):
        waiter.acquire()
    monkeypatch.undo()

    assert holder.close() is None
    assert original_acquire(False)
    sink._reservation_lock.release()


def test_gate_notification_retry_wakes_real_waiting_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = BoundedMessageSink()
    holder = sink_module._ReservationLease(sink._lock)
    waiter = sink_module._ReservationLease(sink._lock)
    assert holder.acquire()
    entered = threading.Event()
    acquired = threading.Event()
    waiter_errors: list[BaseException] = []

    def wait_for_gate() -> None:
        try:
            entered.set()
            assert waiter.acquire(blocking=True)
            acquired.set()
        except BaseException as error:
            waiter_errors.append(error)

    waiter_thread = threading.Thread(target=wait_for_gate)
    waiter_thread.start()
    assert entered.wait(timeout=1)
    time.sleep(0.01)
    original_notify = sink._lock._condition.notify
    calls = 0

    def notify_once(*args: object, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise KeyboardInterrupt
        original_notify(*args, **kwargs)

    monkeypatch.setattr(sink._lock._condition, "notify", notify_once)
    close_error = holder.close()
    assert isinstance(close_error, KeyboardInterrupt)
    assert acquired.wait(timeout=1)
    waiter_thread.join(timeout=1)
    assert not waiter_thread.is_alive()
    assert waiter_errors == []
    assert waiter.close() is None


def test_reservation_release_baseexception_before_and_after_clear_is_recoverable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = BoundedMessageSink(max_pending=2)
    original_release = sink._reservation_lock.release_if_owned
    calls = 0

    def interrupt_once(owner: object | None = None) -> bool:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise KeyboardInterrupt
        return original_release(owner)

    monkeypatch.setattr(sink._reservation_lock, "release_if_owned", interrupt_once)
    with pytest.raises(KeyboardInterrupt):
        sink.offer(parse_message({"protocol_version": "1", "type": "future.before"}))
    monkeypatch.undo()
    assert sink.pending_count == 1

    original_release = sink._reservation_lock.release_if_owned

    def release_then_interrupt(owner: object | None = None) -> bool:
        original_release(owner)
        raise KeyboardInterrupt

    monkeypatch.setattr(sink._reservation_lock, "release_if_owned", release_then_interrupt)
    with pytest.raises(KeyboardInterrupt):
        sink.offer(parse_message({"protocol_version": "1", "type": "future.after"}))
    monkeypatch.undo()
    assert sink.pending_count == 2
    assert len(sink.drain()) == 2


def test_reservation_close_keeps_uncleared_owner_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = BoundedMessageSink()
    lease = sink_module._ReservationLease(sink._lock)
    assert lease.acquire()
    original = sink._lock.release_if_owned
    attempts = 0

    def fail_before_clear(owner: object | None = None) -> bool:
        nonlocal attempts
        attempts += 1
        if attempts <= 2:
            raise KeyboardInterrupt
        return original(owner)

    monkeypatch.setattr(sink._lock, "release_if_owned", fail_before_clear)
    error = lease.close()
    assert isinstance(error, KeyboardInterrupt)
    assert lease.phase.name == "RELEASING"
    assert sink._lock.is_owned(lease)
    monkeypatch.undo()
    assert lease.close() is None
    assert not sink._lock.is_owned(lease)
    assert sink.offer(parse_message({"protocol_version": "1", "type": "future.after-release"}))


def test_public_drain_retains_lease_after_two_preclear_cleanup_faults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = BoundedMessageSink()
    assert sink.offer(parse_message({"protocol_version": "1", "type": "future.lease"}))
    original = sink._lock.release_if_owned
    attempts = 0

    def fail_twice(owner: object | None = None) -> bool:
        nonlocal attempts
        attempts += 1
        if attempts <= 2:
            raise KeyboardInterrupt
        return original(owner)

    monkeypatch.setattr(sink._lock, "release_if_owned", fail_twice)
    batch = sink.drain()
    assert len(batch) == 1
    monkeypatch.undo()
    recovered = sink.drain()
    assert recovered.token == batch.token
    sink.acknowledge(recovered)
    assert acked_sink_drain(sink) == []


def test_cleanup_obligation_is_published_before_finish_helper_fault(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = BoundedMessageSink(max_pending=2)

    def interrupt_before_finish(_reservation: object) -> None:
        raise KeyboardInterrupt("before cleanup helper")

    monkeypatch.setattr(sink, "_finish_reservation", interrupt_before_finish)
    with pytest.raises(KeyboardInterrupt) as raised:
        sink.offer(parse_message({"protocol_version": "1", "type": "future.helper"}))
    assert getattr(raised.value, "capture_committed", False)
    monkeypatch.undo()

    # The next producer retries the gate-owned RELEASING lease before taking
    # admission.  The committed first message remains visible and a waiter
    # cannot be stranded behind the interrupted helper.
    assert sink.offer(parse_message({"protocol_version": "1", "type": "future.after"}))
    batch = sink.drain()
    assert len(batch) == 2
    sink.acknowledge(batch)


def test_gate_retries_two_failed_notifications_for_waiter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = BoundedMessageSink()
    holder = sink_module._ReservationLease(sink._lock)
    assert holder.acquire()
    waiter = sink_module._ReservationLease(sink._lock)
    entered = threading.Event()
    waiting = threading.Event()
    acquired = threading.Event()

    def wait_for_gate() -> None:
        entered.set()
        assert waiter.acquire(blocking=True)
        acquired.set()
        assert waiter.close() is None

    original_wait = sink._lock._condition.wait

    def mark_waiting(*args: object, **kwargs: object) -> bool:
        waiting.set()
        return original_wait(*args, **kwargs)

    monkeypatch.setattr(sink._lock._condition, "wait", mark_waiting)
    waiter_thread = threading.Thread(target=wait_for_gate)
    waiter_thread.start()
    assert entered.wait(timeout=1)
    assert waiting.wait(timeout=1)
    original = sink._lock._condition.notify
    calls = 0

    def fail_twice(*args: object, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        if calls <= 2:
            raise KeyboardInterrupt
        original(*args, **kwargs)

    monkeypatch.setattr(sink._lock._condition, "notify", fail_twice)
    close_error = holder.close()
    assert isinstance(close_error, KeyboardInterrupt)
    assert holder.phase.name == "RELEASING"
    assert not acquired.wait(timeout=0.05)
    monkeypatch.undo()
    assert holder.close() is None
    assert acquired.wait(timeout=1)
    waiter_thread.join(timeout=1)
    assert not waiter_thread.is_alive()


def test_gate_owner_clear_fault_keeps_notification_obligation_for_waiter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = BoundedMessageSink()
    holder = sink_module._ReservationLease(sink._lock)
    waiter = sink_module._ReservationLease(sink._lock)
    assert holder.acquire()
    acquired = threading.Event()

    def wait_for_gate() -> None:
        assert waiter.acquire(blocking=True)
        acquired.set()

    waiter_thread = threading.Thread(target=wait_for_gate)
    waiter_thread.start()
    original_replace = gate_module.replace
    failures = 0

    def fail_owner_clear(state: object, **changes: object) -> object:
        nonlocal failures
        if changes.get("owner") is None and changes.get("released_generation") is not None:
            failures += 1
            if failures <= 2:
                raise KeyboardInterrupt("owner clear publication")
        return original_replace(state, **changes)  # type: ignore[arg-type]

    monkeypatch.setattr(gate_module, "replace", fail_owner_clear)
    error = holder.close()
    assert isinstance(error, KeyboardInterrupt)
    assert not acquired.wait(timeout=0.05)
    assert sink._lock._state.notify_pending
    monkeypatch.undo()
    assert holder.close() is None
    assert acquired.wait(timeout=1)
    waiter_thread.join(timeout=1)
    assert not waiter_thread.is_alive()
    assert waiter.close() is None


def test_queue_lock_ownership_boundaries_do_not_strand_slot_or_drain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = BoundedMessageSink(max_pending=2)
    original_acquire = sink._lock.acquire

    def interrupt_after_queue_acquire(
        blocking: bool = True, owner: object | None = None
    ) -> bool:
        acquired = original_acquire(blocking, owner)
        if acquired:
            raise KeyboardInterrupt
        return acquired

    monkeypatch.setattr(sink._lock, "acquire", interrupt_after_queue_acquire)
    with pytest.raises(KeyboardInterrupt):
        sink.offer(parse_message({"protocol_version": "1", "type": "future.queue"}))
    monkeypatch.undo()
    assert sink.pending_count == 0

    original_release = sink._lock.release_if_owned

    def release_queue_then_interrupt(owner: object | None = None) -> bool:
        original_release(owner)
        raise KeyboardInterrupt

    monkeypatch.setattr(sink._lock, "release_if_owned", release_queue_then_interrupt)
    with pytest.raises(KeyboardInterrupt):
        sink.offer(parse_message({"protocol_version": "1", "type": "future.release"}))
    monkeypatch.undo()
    # The queue commit succeeded before cleanup failed; the surfaced error is
    # therefore a committed outcome and retrying would duplicate the message.
    assert sink.pending_count == 1
    assert len(sink.drain()) == 1


def test_concurrent_loss_admissions_cover_every_reserved_position() -> None:
    sink = BoundedMessageSink(max_pending=1)
    barrier = threading.Barrier(8)
    results: list[bool] = []
    results_lock = threading.Lock()

    def record() -> None:
        barrier.wait()
        result = sink.record_loss()
        with results_lock:
            results.append(result)

    workers = [threading.Thread(target=record) for _ in range(8)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=1)
    assert all(not worker.is_alive() for worker in workers)
    assert results == [True] * 8
    assert sink.dropped_count == 8
    gaps = sink.drain()
    assert len(gaps) == 1
    assert isinstance(gaps[0], KnownParsedMessage)
    assert gaps[0].message["dropped_count"] == "8"


def test_natural_gate_contention_keeps_all_60000_admissions_accounted() -> None:
    sink = BoundedMessageSink(max_pending=64)
    message = parse_message({"protocol_version": "1", "type": "future.contention"})
    stop = threading.Event()
    errors: list[BaseException] = []
    total = 12 * 5_000

    def produce() -> None:
        try:
            for _ in range(5_000):
                sink.offer(message)
        except BaseException as error:
            errors.append(error)

    def consume() -> None:
        try:
            while not stop.is_set() or sink.pending_count:
                batch = sink.drain()
                sink.acknowledge(batch)
        except BaseException as error:
            errors.append(error)

    consumer = threading.Thread(target=consume)
    workers = [threading.Thread(target=produce) for _ in range(12)]
    consumer.start()
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=5)
    stop.set()
    consumer.join(timeout=5)
    assert all(not worker.is_alive() for worker in workers)
    assert not consumer.is_alive()
    assert errors == []
    assert sink.accepted_count + sink.dropped_count == total


def test_keyboard_interrupt_during_drain_restores_detached_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = BoundedMessageSink()
    assert sink.offer(parse_message({"protocol_version": "1", "type": "future.interrupt"}))

    def interrupt(_message: ParsedMessageResult, _position: int) -> ParsedMessageResult:
        raise KeyboardInterrupt

    monkeypatch.setattr(sink_module, "_with_delivery_position", interrupt)
    with pytest.raises(KeyboardInterrupt):
        sink.drain()
    assert sink.pending_count == 1
    assert sink._last_delivered_position == 0
    monkeypatch.undo()
    assert len(sink.drain()) == 1


def test_sink_does_not_expose_fabricable_prepared_metric_seam() -> None:
    sink = BoundedMessageSink()
    assert not hasattr(sink, "prepare")
    assert not hasattr(sink, "offer_prepared")


def test_uint64_exhaustion_is_stable_without_constructing_next_position() -> None:
    sink = BoundedMessageSink(max_pending=1)
    sink._next_position_value = MAX_U64 - 1
    assert sink.offer(parse_message({"protocol_version": "1", "type": "future.last"}))
    assert not sink.offer(parse_message({"protocol_version": "1", "type": "future.drop"}))
    assert sink.exhausted
    assert sink.dropped_count == 1
    pending = sink.pending_count
    ranges = sink.loss_range_count
    for _ in range(3):
        assert not sink.offer(parse_message({"protocol_version": "1", "type": "future.repeat"}))
        assert not sink.record_loss()
    assert sink.pending_count == pending
    assert sink.loss_range_count == ranges
    drained = sink.drain()
    assert len(drained) == 1
    drained_payload = (
        drained[0].message
        if isinstance(drained[0], KnownParsedMessage)
        else drained[0].payload
    )
    assert drained_payload["type"] == "future.last"
    assert drained_payload["delivery_position"] == str(MAX_U64 - 1)
    assert sink.pending_count == 0
    assert sink.loss_range_count == ranges
    sink.acknowledge(drained)
    assert acked_sink_drain(sink) == []
    assert sink.loss_range_count == ranges


def test_terminal_loss_splits_representable_prefix_from_terminal_position() -> None:
    sink = BoundedMessageSink(max_pending=2)
    sink._next_position_value = MAX_U64 - 2
    assert sink.offer(parse_message({"protocol_version": "1", "type": "future.edge"}))
    assert sink.record_loss()
    assert sink.record_loss()
    drained = sink.drain()
    assert len(drained) == 2
    assert isinstance(drained[1], KnownParsedMessage)
    assert drained[1].message["type"] == "stream.gap"
    assert drained[1].message["actual_sequence"] == str(MAX_U64)
    assert sink.loss_range_count == 1
    sink.acknowledge(drained)
    assert acked_sink_drain(sink) == []


def test_drain_failure_restores_detached_queue_and_cursor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = BoundedMessageSink()
    message = parse_message({"protocol_version": "1", "type": "future.rollback"})
    assert sink.offer(message)

    def fail(_message: ParsedMessageResult, _position: int) -> ParsedMessageResult:
        raise RuntimeError("injected drain failure")

    monkeypatch.setattr(sink_module, "_with_delivery_position", fail)
    with pytest.raises(RuntimeError, match="injected drain failure"):
        sink.drain()
    assert sink.pending_count == 1
    assert sink._last_delivered_position == 0
    monkeypatch.undo()
    delivered = sink.drain()
    assert len(delivered) == 1
    assert sink.pending_count == 0


def test_drain_gap_failure_restores_loss_range_and_detached_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = BoundedMessageSink(max_pending=1)
    assert sink.offer(parse_message({"protocol_version": "1", "type": "future.keep"}))
    assert not sink.offer(parse_message({"protocol_version": "1", "type": "future.drop"}))

    def fail(_expected: int, _loss_end: int) -> ParsedMessageResult:
        raise RuntimeError("injected gap failure")

    monkeypatch.setattr(sink_module, "_gap_after_loss", fail)
    with pytest.raises(RuntimeError, match="injected gap failure"):
        sink.drain()
    assert sink.pending_count == 1
    assert sink.loss_range_count == 1
    assert sink._last_delivered_position == 0
    monkeypatch.undo()
    drained = sink.drain()
    assert [
        (item.message if isinstance(item, KnownParsedMessage) else item.payload)["type"]
        for item in drained
    ] == ["future.keep", "stream.gap"]


def test_producer_losses_during_drain_are_not_cleared_or_duplicated(
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
    first_batches: list[object] = []
    drain_thread = threading.Thread(target=lambda: first_batches.append(sink.drain()))
    drain_thread.start()
    assert entered.wait(timeout=1)
    sink.record_loss()
    assert not sink.offer(parse_message({"protocol_version": "1", "type": "future.after"}))
    release.set()
    drain_thread.join(timeout=1)
    assert not drain_thread.is_alive()

    first_batch = first_batches[0]
    sink.acknowledge(first_batch)
    second_batch = sink.drain()
    combined = list(first_batch) + list(second_batch)  # type: ignore[arg-type]
    types = [
        (message.message if isinstance(message, KnownParsedMessage) else message.payload)["type"]
        for message in combined
    ]
    assert types == ["future.first", "stream.gap"]
    assert types.count("stream.gap") == 1
    sink.acknowledge(second_batch)


def test_overlapping_drains_are_single_flight_and_keep_delivery_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = BoundedMessageSink(max_pending=4)
    assert sink.offer(parse_message({"protocol_version": "1", "type": "future.one"}))
    assert sink.offer(parse_message({"protocol_version": "1", "type": "future.two"}))
    entered = threading.Event()
    release = threading.Event()
    second_done = threading.Event()
    first_result: list[object] = []
    second_result: list[object] = []
    original_positioner = sink_module._with_delivery_position

    def paused_positioner(message: ParsedMessageResult, position: int) -> ParsedMessageResult:
        entered.set()
        assert release.wait(timeout=1)
        return original_positioner(message, position)

    monkeypatch.setattr(sink_module, "_with_delivery_position", paused_positioner)
    first = threading.Thread(target=lambda: first_result.append(sink.drain()))

    def run_second() -> None:
        second_result.append(sink.drain())
        second_done.set()

    second = threading.Thread(target=run_second)
    first.start()
    assert entered.wait(timeout=1)
    second.start()
    assert not second_done.wait(timeout=0.05)
    release.set()
    first.join(timeout=1)
    second.join(timeout=1)
    assert not first.is_alive()
    assert not second.is_alive()
    assert len(first_result) == 1
    assert len(second_result) == 1
    first_batch = first_result[0]
    second_batch = second_result[0]
    assert first_batch.token == second_batch.token  # type: ignore[union-attr]
    assert [
        (item.message if isinstance(item, KnownParsedMessage) else item.payload)["type"]
        for item in first_batch  # type: ignore[union-attr]
    ] == ["future.one", "future.two"]
    sink.acknowledge(first_batch)  # type: ignore[arg-type]


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
    batch = sink.drain()
    sink.acknowledge(batch)
    assert acked_sink_drain(sink) == []


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


def test_repeated_completed_flows_release_all_active_accounting() -> None:
    addon = CaptureAddon(clock=lambda: "now")
    for index in range(5):
        flow = fake_flow(f"fake-{index}", request_body=b"req", response_body=b"resp")
        addon.requestheaders(flow)
        addon.request(flow)
        addon.responseheaders(flow)
        addon.response(flow)
        assert addon.counters["active_flows"] == 0
        assert addon.counters["active_metadata_bytes"] == 0
        assert addon.counters["active_prefix_bytes"] == 0


def test_repeated_error_completions_release_all_active_accounting() -> None:
    addon = CaptureAddon(clock=lambda: "now")
    for index in range(5):
        flow = fake_flow(f"err-{index}", request_body=b"req")
        flow.error = SimpleNamespace(msg="synthetic failure")
        addon.requestheaders(flow)
        addon.error(flow)
        assert addon.counters["active_flows"] == 0
        assert addon.counters["active_metadata_bytes"] == 0
        assert addon.counters["active_prefix_bytes"] == 0


def test_repeated_deferred_completions_release_all_active_accounting() -> None:
    addon = CaptureAddon(clock=lambda: "now")
    for index in range(5):
        flow = fake_flow(f"late-{index}", request_body=None, response_body=b"resp")
        addon.requestheaders(flow)
        addon.responseheaders(flow)
        addon.response(flow)
        assert addon.counters["active_flows"] == 1
        addon.request(flow)
        assert addon.counters["active_flows"] == 0
        assert addon.counters["active_metadata_bytes"] == 0
        assert addon.counters["active_prefix_bytes"] == 0


def test_repeated_real_httpflow_completions_release_all_active_accounting() -> None:
    from mitmproxy.test import tflow

    addon = CaptureAddon(clock=lambda: "now")
    for _ in range(5):
        flow = tflow.tflow(resp=True)
        addon.requestheaders(flow)
        addon.request(flow)
        addon.responseheaders(flow)
        addon.response(flow)
        assert addon.counters["active_flows"] == 0
        assert addon.counters["active_metadata_bytes"] == 0
        assert addon.counters["active_prefix_bytes"] == 0


def test_tight_budget_flow_still_captures_after_many_completions() -> None:
    config = CaptureConfig(
        source_id="source",
        max_body_prefix_bytes=64,
        max_in_memory_bytes=4_096,
    )
    addon = CaptureAddon(config=config, clock=lambda: "now")
    probe = fake_flow("probe", request_body=None, response_body=b"data")
    addon.requestheaders(probe)
    addon.request(probe)
    addon.responseheaders(probe)
    single_flow_weight = addon._active_metadata_bytes
    addon.response(probe)

    for index in range(50):
        flow = fake_flow(f"churn-{index}", request_body=None, response_body=b"data")
        addon.requestheaders(flow)
        addon.request(flow)
        addon.responseheaders(flow)
        addon.response(flow)
        payloads(addon)
    assert addon.counters["active_metadata_bytes"] == 0

    addon.max_in_memory_bytes = single_flow_weight + 8
    late = fake_flow("tight", request_body=None, response_body=b"data")
    addon.requestheaders(late)
    addon.request(late)
    addon.responseheaders(late)
    addon.response(late)
    assert addon.counters["evicted_flows"] == 0
    body_end = [
        message
        for message in payloads(addon)
        if message.get("type") == "body.end"
        and message.get("body_side") == "response"
        and message.get("flow_id") == "tight"
    ][-1]
    body = body_end["body"]
    assert isinstance(body, Mapping)
    assert body["state"] == "captured"
    assert base64.b64decode(body["data"]) == b"data"


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


@pytest.mark.parametrize(
    ("value", "accepted"),
    [
        (-(1 << 63), True),
        (-(1 << 63) - 1, False),
        (MAX_U64, True),
        (MAX_U64 + 1, False),
    ],
)
def test_shared_metric_boundary_corpus_matches_sink_and_store(
    value: int, accepted: bool
) -> None:
    parsed = parse_message(
        {
            "protocol_version": "1",
            "type": "future.metric-boundary",
            "nested": {"value": value},
        }
    )
    sink = BoundedMessageSink(max_memory_bytes=512)
    store = MemoryStore(max_memory_bytes=512)
    payload = parsed.payload if not isinstance(parsed, KnownParsedMessage) else parsed.message
    if accepted:
        assert metrics_module.canonical_weight(payload) == memory_module._message_weight(payload)
        assert sink.offer(parsed)
        store.append(parsed)
        assert sink.pending_count == 1
        assert store.counters["retained_messages"] == 1
    else:
        with pytest.raises(ValueError, match="bounded protocol numeric range"):
            metrics_module.canonical_weight(payload)
        with pytest.raises(ValueError, match="bounded protocol numeric range"):
            memory_module._message_weight(payload)
        with pytest.raises(ValueError, match="bounded protocol numeric range"):
            sink.offer(parsed)
        with pytest.raises(ValueError, match="bounded protocol numeric range"):
            store.append(parsed)
        assert sink.pending_count == 0
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


def test_stock_mitmdump_12_2_3_loads_capture_script_without_dataclass_failure() -> None:
    mitmdump = Path(sys.executable).with_name("mitmdump")
    if not mitmdump.exists():
        pytest.skip("mitmdump is not installed beside the test interpreter")
    repository = Path(__file__).parents[1]
    script = repository / "src" / "mitm_inspector" / "capture" / "addon.py"
    version = subprocess.run(
        [str(mitmdump), "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "12.2.3" in version.stdout
    process = subprocess.Popen(
        [
            str(mitmdump),
            "-q",
            "-s",
            str(script),
            "--listen-host",
            "127.0.0.1",
            "--listen-port",
            "0",
        ],
        cwd=repository,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=1.5)
    except subprocess.TimeoutExpired:
        process.terminate()
        stdout, stderr = process.communicate(timeout=3)
        assert "error in script" not in stderr
        assert "Traceback" not in stderr
    else:
        pytest.fail(f"mitmdump exited during startup ({process.returncode}): {stdout}{stderr}")
    assert "dataclass" not in stderr


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


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_in_memory_bytes": 1.5},
        {"max_pending_messages": True},
        {"max_body_prefix_bytes": MAX_U64 + 1},
        {"max_in_memory_bytes": MAX_U64 + 1},
        {"max_pending_messages": MAX_U64 + 1},
    ],
)
def test_direct_capture_config_rejects_non_exact_or_out_of_range_uint64(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="exact uint64"):
        CaptureConfig(**kwargs)  # type: ignore[arg-type]


def test_invalid_direct_config_cannot_leave_addon_source_or_flow_state() -> None:
    with pytest.raises(ValueError, match="exact uint64"):
        CaptureAddon(config=CaptureConfig(max_in_memory_bytes=MAX_U64 + 1))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_pending": 1.0},
        {"max_pending": True},
        {"max_pending": float("nan")},
        {"max_body_bytes": MAX_U64 + 1},
        {"max_memory_bytes": MAX_U64 + 1},
    ],
)
def test_sink_rejects_non_integral_or_unbounded_limits(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="exact bounded integer"):
        BoundedMessageSink(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("limit", [True, 0, 1.5, float("nan"), float("inf"), MAX_U64 + 1])
def test_sink_drain_rejects_invalid_limit_before_sequencing(limit: object) -> None:
    sink = BoundedMessageSink()
    assert sink.offer(parse_message({"protocol_version": "1", "type": "future.limit"}))
    with pytest.raises(ValueError):
        sink.drain(limit)  # type: ignore[arg-type]
    assert sink.pending_count == 1


@pytest.mark.parametrize("limit", [True, 0, 1.5, float("nan"), float("inf"), MAX_U64 + 1])
def test_addon_drain_rejects_invalid_limit_before_active_purge(limit: object) -> None:
    now = [0.0]
    addon = CaptureAddon(
        active_clock=lambda: now[0],
        max_active_age_seconds=1,
        clock=lambda: "now",
    )
    addon.requestheaders(fake_flow("limit-flow"))
    now[0] = 2.0
    with pytest.raises(ValueError):
        addon.drain(limit)  # type: ignore[arg-type]
    assert "limit-flow" in addon._flows


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_items": 1.0},
        {"max_items": True},
        {"max_items": float("nan")},
        {"max_messages": MAX_U64 + 1},
        {"max_body_bytes": MAX_U64 + 1},
        {"max_memory_bytes": MAX_U64 + 1},
        {"max_messages_per_flow": MAX_U64 + 1},
        {"max_standalone_messages": MAX_U64 + 1},
        {"max_age_seconds": float("nan")},
        {"max_age_seconds": float("inf")},
        {"max_age_seconds": -1},
        {"max_age_seconds": True},
    ],
)
def test_store_rejects_non_integral_or_unbounded_limits(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        MemoryStore(**kwargs)  # type: ignore[arg-type]


def test_store_rejects_overflowing_exact_integer_age_before_append() -> None:
    with pytest.raises(ValueError, match="bounded"):
        MemoryStore(max_age_seconds=10**400)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_active_flows": True},
        {"max_active_flows": 1.0},
        {"max_active_flows": float("nan")},
        {"max_active_flows": float("inf")},
        {"max_active_flows": MAX_U64 + 1},
        {"max_active_age_seconds": True},
        {"max_active_age_seconds": float("nan")},
        {"max_active_age_seconds": float("inf")},
    ],
)
def test_active_flow_limits_reject_nonfinite_or_unbounded_values(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        CaptureAddon(**kwargs)  # type: ignore[arg-type]


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


def test_store_expiry_is_incremental_and_append_skips_debug_full_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [0.0]
    store = MemoryStore(max_messages=2_000, max_age_seconds=10, clock=lambda: now[0])

    def forbidden_full_scan() -> None:
        raise AssertionError("append must not run exhaustive invariant scans")

    monkeypatch.setattr(store, "_assert_invariants", forbidden_full_scan)
    for index in range(1_000):
        store.append(parse_message({"protocol_version": "1", "type": f"future.{index}"}))
    assert len(store._expiry_heap) == 1_000
    now[0] = 11.0
    assert store.counters["retained_messages"] == 0
    assert not store._expiry_heap


def test_saturated_default_scale_append_uses_incremental_eviction_candidates() -> None:
    store = MemoryStore(max_items=2_000, max_messages=32_000)
    for index in range(2_001):
        store.append(completed_flow_message(f"completed-{index}", str(index)))
    completion_visits = store._completion_candidate_visits
    store._completion_candidate_visits = 0
    store.append(completed_flow_message("completed-final", "9000"))
    assert store._completion_candidate_visits <= 2
    assert store._completion_candidate_visits + completion_visits < 2_100

    for index in range(31_999):
        store.append(parse_message({"protocol_version": "1", "type": f"future.{index}"}))
    store._eviction_candidate_visits = 0
    store.append(parse_message({"protocol_version": "1", "type": "future.final"}))
    assert store.counters["retained_messages"] <= store.max_messages
    assert store._eviction_candidate_visits <= 2


def test_eviction_heaps_do_not_retain_evicted_payloads_or_unbounded_stale_entries() -> None:
    completed = MemoryStore(max_items=1)
    completed.append(completed_flow_message("old", "0"))
    old_record = completed._flows["old"]
    old_ref = weakref.ref(old_record)
    completed.append(completed_flow_message("new", "1"))
    del old_record
    gc.collect()
    assert old_ref() is None

    standalone = MemoryStore(max_messages=1)
    standalone.append(parse_message({"protocol_version": "1", "type": "future.old"}))
    old_stored = standalone._standalone[0]
    old_stored_ref = weakref.ref(old_stored)
    standalone.append(parse_message({"protocol_version": "1", "type": "future.new"}))
    del old_stored
    gc.collect()
    assert old_stored_ref() is None

    body = MemoryStore(max_body_bytes=1)
    body.append(
        parse_message(
            {
                "protocol_version": "1",
                "type": "body.end",
                "flow_id": "body",
                "body_side": "response",
                "total_bytes": "1",
                "body": {
                    "state": "captured",
                    "size_bytes": "1",
                    "encoding": "base64",
                    "data": base64.b64encode(b"x").decode(),
                },
            }
        )
    )
    body_ref = weakref.ref(body._flows["body"].messages[0])
    body.append(
        parse_message(
            {
                "protocol_version": "1",
                "type": "body.end",
                "flow_id": "body-next",
                "body_side": "response",
                "total_bytes": "2",
                "body": {
                    "state": "captured",
                    "size_bytes": "2",
                    "encoding": "base64",
                    "data": base64.b64encode(b"yy").decode(),
                },
            }
        )
    )
    gc.collect()
    assert body_ref() is None

    memory = MemoryStore(max_memory_bytes=64)
    memory.append(parse_message({"protocol_version": "1", "type": "future.small", "value": "x"}))
    small_ref = weakref.ref(memory._standalone[0])
    memory.append(
        parse_message(
            {"protocol_version": "1", "type": "future.large", "value": "x" * 100}
        )
    )
    gc.collect()
    assert small_ref() is None

    now = [0.0]
    aged = MemoryStore(max_age_seconds=1, clock=lambda: now[0])
    aged.append(parse_message({"protocol_version": "1", "type": "future.aged"}))
    aged_ref = weakref.ref(aged._standalone[0])
    now[0] = 2.0
    assert aged.counters["retained_messages"] == 0
    gc.collect()
    assert aged_ref() is None
    assert len(aged._message_heap) <= 64
    assert len(aged._weight_heap) <= 64
    assert len(aged._body_heap) <= 64


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

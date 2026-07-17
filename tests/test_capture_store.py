import base64
from collections.abc import Iterator, Mapping
from types import SimpleNamespace

import pytest

from mitm_inspector.capture.addon import CaptureAddon
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

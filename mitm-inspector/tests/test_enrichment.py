from __future__ import annotations

import base64
import gzip
import json
import logging
from pathlib import Path

import pytest

from mitm_inspector.api.bodies import (
    MAX_DECODED_BODY_BYTES,
    body_content_encoding,
    decoded_body_bytes,
    decoded_body_descriptor,
)
from mitm_inspector.api.projection import collect_grid_flows, enriched_flow, grid_flow
from mitm_inspector.protocol import parse_message
from mitm_inspector.store.memory import MemoryStore
from mitm_inspector.store.sqlite import SQLiteFlowStorage


def descriptor(data: bytes, *, content_type: str = "application/json") -> dict[str, object]:
    return {
        "state": "captured",
        "size_bytes": str(len(data)),
        "content_type": content_type,
        "encoding": "base64",
        "data": base64.b64encode(data).decode("ascii"),
    }


def anthropic_metadata(
    request: object,
    response: bytes = b"",
    *,
    flow_id: str = "flow-1",
    path: str = "/v1/messages",
    response_encoding: str | None = None,
) -> dict[str, object]:
    request_bytes = json.dumps(request).encode()
    response_headers: list[dict[str, str]] = [{"name": "content-type", "value": "application/json"}]
    if response_encoding is not None:
        response_headers.append({"name": "content-encoding", "value": response_encoding})
    metadata: dict[str, object] = {
        "flow_id": flow_id,
        "method": "POST",
        "scheme": "https",
        "host": "api.anthropic.com",
        "port": "443",
        "path": path,
        "request_headers": [{"name": "content-type", "value": "application/json"}],
        "response_headers": response_headers,
        "response_status": "200",
        "request_body": descriptor(request_bytes),
        "response_body": descriptor(response),
    }
    return metadata


def metadata_message(metadata: dict[str, object]) -> dict[str, object]:
    return {"protocol_version": "1", "type": "flow.metadata", "metadata": metadata}


def lifecycle(flow_id: str, sequence: int, state: str, occurred_at: str) -> dict[str, object]:
    return {
        "protocol_version": "1",
        "type": "flow.lifecycle",
        "source_id": "source",
        "flow_id": flow_id,
        "event_id": f"event-{sequence}",
        "occurred_at": occurred_at,
        "sequence": str(sequence),
        "state": state,
    }


def test_messages_summary_strips_reminders_and_reads_gzip_sse_usage() -> None:
    sse = b"\n".join(
        [
            b'data: {"type":"message_start","message":{"usage":'
            b'{"input_tokens":12,"cache_read_input_tokens":4}}}',
            b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
            b'"usage":{"output_tokens":7,"output_tokens_details":'
            b'{"thinking_tokens":2}}}',
            b"data: [DONE]",
        ]
    )
    request = {
        "model": "claude-test",
        "stream": True,
        "messages": [
            {
                "role": "user",
                "content": (
                    "before <system-reminder>private\ncontext</system-reminder> after   text"
                ),
            }
        ],
    }
    metadata = anthropic_metadata(
        request,
        gzip.compress(sse),
        response_encoding="gzip",
    )

    projected = grid_flow(metadata)

    assert projected["request_body_size"] == str(len(json.dumps(request).encode()))
    assert projected["response_body_size"] == str(len(gzip.compress(sse)))
    assert projected["content_encoding"] == {"response": "gzip"}
    assert projected["summary"] == {
        "kind": "anthropic_messages",
        "model": "claude-test",
        "message_count": "1",
        "stream": True,
        "preview": {"source": "user_text", "text": "before after text"},
        "input_tokens": "12",
        "cache_read_input_tokens": "4",
        "stop_reason": "end_turn",
        "output_tokens": "7",
        "thinking_tokens": "2",
    }


def test_unsupported_content_encoding_logs_visible_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="mitm_inspector.api.bodies"):
        assert body_content_encoding(
            {"response_headers": [{"name": "content-encoding", "value": "br"}]},
            "response",
        ) is None
    assert "unsupported content-encoding 'br'" in caplog.text


def test_messages_summary_resolves_tool_result_only_turn_and_degrades_malformed() -> None:
    request = {
        "messages": [
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "tool-1", "name": "Read"}],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "tool-1"}],
            },
        ]
    }
    projected = enriched_flow(anthropic_metadata(request, b"not-json"))
    assert projected["summary"] == {
        "kind": "anthropic_messages",
        "message_count": "2",
        "preview": {"source": "tool_result", "tool_name": "Read"},
    }

    malformed = anthropic_metadata({}, b"{broken")
    malformed["request_body"] = descriptor(b"{broken")
    assert enriched_flow(malformed)["summary"] == {"kind": "anthropic_messages"}


def test_count_tokens_summary_reads_plain_json_result() -> None:
    metadata = anthropic_metadata(
        {"model": "claude-test", "messages": []},
        json.dumps({"input_tokens": 321}).encode(),
        path="/v1/messages/count_tokens",
    )
    assert enriched_flow(metadata)["summary"] == {
        "kind": "anthropic_count_tokens",
        "model": "claude-test",
        "message_count": "0",
        "preview": {"source": "none"},
        "count_tokens_result": "321",
    }


def test_gzip_body_is_decoded_in_metadata_and_original_encoding_is_exposed() -> None:
    response = json.dumps({"stop_reason": "end_turn", "usage": {"output_tokens": 3}}).encode()
    compressed = gzip.compress(response)
    metadata = anthropic_metadata({}, compressed, response_encoding="gzip")

    projected = enriched_flow(metadata)

    decoded = base64.b64decode(projected["response_body"]["data"])
    assert decoded == response
    assert projected["response_body"]["size_bytes"] == str(len(response))
    assert projected["response_body_size"] == str(len(compressed))
    assert projected["content_encoding"] == {"response": "gzip"}


def test_content_decoding_is_bounded_and_over_limit_descriptor_falls_back_raw() -> None:
    compressed = gzip.compress(b"x" * (MAX_DECODED_BODY_BYTES + 1))
    original = descriptor(compressed)

    decoded, was_decoded = decoded_body_bytes(original, "gzip")
    served, descriptor_was_decoded = decoded_body_descriptor(original, "gzip")

    assert was_decoded
    assert decoded == b"x" * MAX_DECODED_BODY_BYTES
    assert not descriptor_was_decoded
    assert served == original


def test_concatenated_gzip_members_are_all_decoded() -> None:
    compressed = gzip.compress(b"first-member") + gzip.compress(b"second-member")
    served, was_decoded = decoded_body_descriptor(descriptor(compressed), "gzip")
    assert was_decoded
    assert base64.b64decode(served["data"]) == b"first-membersecond-member"


def test_gzip_trailing_garbage_preserves_raw_descriptor() -> None:
    compressed = gzip.compress(b"valid-member") + b"not-another-member"
    original = descriptor(compressed)
    served, was_decoded = decoded_body_descriptor(original, "gzip")
    assert not was_decoded
    assert served == original


def test_invalid_gzip_is_attempted_once_for_descriptor_and_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mitm_inspector.api.bodies as bodies_module

    calls = 0
    original_decode = bodies_module._decode_content

    def count_decode(data: bytes, encoding: str) -> bodies_module.ContentDecodeResult:
        nonlocal calls
        calls += 1
        return original_decode(data, encoding)

    monkeypatch.setattr(bodies_module, "_decode_content", count_decode)
    compressed = gzip.compress(b"invalid trailer") + b"garbage"
    metadata = anthropic_metadata({}, compressed, response_encoding="gzip")

    enriched_flow(metadata)

    assert calls == 1


def test_truncated_compressed_prefix_preserves_encoded_size_and_data() -> None:
    compressed = gzip.compress(b"decoded" * 100_000)
    prefix = compressed[: len(compressed) // 2]
    original = descriptor(prefix)
    original.update(
        {
            "state": "truncated",
            "size_bytes": str(len(compressed)),
            "captured_bytes": str(len(prefix)),
        }
    )
    served, was_decoded = decoded_body_descriptor(original, "gzip")
    assert not was_decoded
    assert served == original


def test_empty_compressed_capture_preserves_raw_descriptor() -> None:
    original = descriptor(b"")
    served, was_decoded = decoded_body_descriptor(original, "gzip")
    assert not was_decoded
    assert served == original


def test_empty_derived_strings_remain_valid_protocol_values() -> None:
    response = json.dumps({"stop_reason": ""}).encode()
    metadata = anthropic_metadata({"model": "", "messages": []}, response)
    request_body = metadata["request_body"]
    response_body = metadata["response_body"]
    assert isinstance(request_body, dict)
    assert isinstance(response_body, dict)
    request_body["content_type"] = ""
    response_body["content_type"] = ""
    projected = grid_flow(metadata)
    parse_message(
        {
            "protocol_version": "1",
            "type": "browser.snapshot",
            "snapshot_id": "empty-values",
            "cursor": "0",
            "flows": [projected],
        }
    )
    assert projected["request_content_type"] == ""
    assert projected["response_content_type"] == ""
    assert projected["summary"]["model"] == ""
    assert projected["summary"]["stop_reason"] == ""


def test_restart_replay_preserves_enriched_projection(tmp_path: Path) -> None:
    request = {"model": "claude-test", "messages": [{"role": "user", "content": "hello"}]}
    response = gzip.compress(b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}\n')
    metadata = anthropic_metadata(request, response, response_encoding="gzip")
    messages = [
        metadata_message(metadata),
        lifecycle("flow-1", 1, "request_started", "2026-07-20T12:00:00Z"),
        lifecycle("flow-1", 2, "flow_completed", "2026-07-20T12:00:01Z"),
    ]
    live_store = MemoryStore(32)
    storage = SQLiteFlowStorage(tmp_path / "flows.sqlite")
    try:
        for message in messages:
            parsed = parse_message(message)
            live_store.append(parsed)
            storage.offer(parsed)
        storage.flush()
        restarted_store = MemoryStore(32)
        storage.replay_into(restarted_store)

        assert collect_grid_flows(restarted_store) == collect_grid_flows(live_store)
        replayed = collect_grid_flows(restarted_store)["flow-1"]
        assert replayed["started_at"] == "2026-07-20T12:00:00Z"
        assert replayed["ended_at"] == "2026-07-20T12:00:01Z"
    finally:
        storage.close()

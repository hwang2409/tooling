import base64
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from mitm_inspector.capture.redaction import REDACTED, sanitize_header, sanitize_path
from mitm_inspector.protocol import ProtocolError, parse_message

ROOT = Path(__file__).parents[1]
STREAM_PATH = ROOT / "contracts" / "fixtures" / "stream.json"
CONFORMANCE_PATH = ROOT / "contracts" / "fixtures" / "conformance.json"
SCHEMA_PATH = ROOT / "contracts" / "protocol-v1.schema.json"


def fixture_messages() -> list[object]:
    return json.loads(STREAM_PATH.read_text())


def conformance() -> dict[str, list[dict[str, object]]]:
    return json.loads(CONFORMANCE_PATH.read_text())


def schema_validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text())
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def body_descriptors(message: dict[str, object]) -> list[dict[str, object]]:
    message_type = message.get("type")
    if message_type == "flow.metadata":
        metadata = message["metadata"]
        assert isinstance(metadata, dict)
        result = [metadata["request_body"]]
        if "response_body" in metadata:
            result.append(metadata["response_body"])
        return result
    if message_type == "body.end":
        return [message["body"]]
    if message_type == "browser.snapshot":
        result = []
        for flow in message["flows"]:
            result.extend(body_descriptors({"type": "flow.metadata", "metadata": flow}))
        return result
    return []


def test_schema_is_authoritative_for_positive_and_negative_conformance() -> None:
    validator = schema_validator()
    positive = [*fixture_messages(), *conformance()["valid"]]
    for message in positive:
        errors = list(validator.iter_errors(message))
        assert errors == [], errors
    expected_invalid = {case["name"] for case in conformance()["invalid"]}
    assert len(expected_invalid) == len(conformance()["invalid"])
    for case in conformance()["invalid"]:
        errors = list(validator.iter_errors(case["message"]))
        assert errors, case["name"]


def test_python_accepts_positive_and_rejects_every_negative_case() -> None:
    for message in [*fixture_messages(), *conformance()["valid"]]:
        parse_message(message)
    for case in conformance()["invalid"]:
        with pytest.raises(ProtocolError):
            parse_message(case["message"])


def test_shared_fixture_preserves_ordered_duplicates_and_lifecycle_order() -> None:
    messages = [parse_message(message) for message in fixture_messages()]
    metadata = messages[1]["metadata"]
    assert metadata["request_headers"][1:3] == [
        {"name": "x-trace", "value": "first"},
        {"name": "x-trace", "value": "second"},
    ]
    lifecycle = [message for message in messages if message["type"] == "flow.lifecycle"]
    assert [message["state"] for message in lifecycle] == ["response_started", "request_end"]


def test_body_counts_and_states_are_distinct() -> None:
    messages = [parse_message(message) for message in fixture_messages()]
    bodies = [body for message in messages for body in body_descriptors(message)]
    assert [bodies[0]["state"], bodies[1]["state"], bodies[2]["state"]] == [
        "captured",
        "missing",
        "truncated",
    ]
    for body in bodies:
        state = body["state"]
        if state == "captured":
            decoded = base64.b64decode(body["data"], validate=True)
            assert len(decoded) == int(body["size_bytes"])
        elif state == "truncated":
            decoded = base64.b64decode(body["data"], validate=True)
            assert len(decoded) <= int(body["captured_bytes"]) <= int(body["size_bytes"])
        elif state == "empty":
            assert body["size_bytes"] == "0"
        else:
            assert set(body) <= {"state", "content_type"}


def test_truncated_prefix_cannot_exceed_total_or_captured_counts() -> None:
    message = {
        "protocol_version": "1",
        "type": "body.end",
        "flow_id": "f",
        "body_side": "response",
        "total_bytes": "3",
        "body": {
            "state": "truncated",
            "size_bytes": "3",
            "captured_bytes": "4",
            "encoding": "base64",
            "data": "AQID",
        },
    }
    with pytest.raises(ProtocolError, match="exceeds"):
        parse_message(message)


def test_gap_order_and_dropped_count_are_authoritative_runtime_invariants() -> None:
    for expected, actual, dropped in (("2", "1", None), ("9", "11", "2")):
        message = {
            "protocol_version": "1",
            "type": "stream.gap",
            "expected_sequence": expected,
            "actual_sequence": actual,
        }
        if dropped is not None:
            message["dropped_count"] = dropped
        with pytest.raises(ProtocolError):
            parse_message(message)


def test_unknown_types_and_additive_fields_are_tolerated_without_numeric_coercion() -> None:
    message = parse_message(fixture_messages()[-1])
    assert message["type"] == "future.message"
    assert message["sequence"] == "9007199254740993"
    future = parse_message(conformance()["valid"][-1])
    assert future["type"] == "future.additive"
    assert future["sequence"] == "18446744073709551616"


def test_fixtures_have_no_secret_canaries() -> None:
    serialized = json.dumps({"stream": fixture_messages(), "conformance": conformance()}).lower()
    for canary in (
        "authorization-canary",
        "x-api-key-canary",
        "cookie-canary",
        "query-secret-canary",
        "contract-secret",
    ):
        assert canary not in serialized


def test_redaction_is_case_and_separator_insensitive_for_credential_names() -> None:
    for name in (
        "Authorization",
        "X_Api_Key",
        "X-Auth_Token",
        "X-Amz-Security_Token",
        "X-Custom_Secret",
    ):
        assert sanitize_header(name, "credential-canary")[1] == REDACTED
    assert sanitize_header("X-Trace", "safe-value") == ("x-trace", "safe-value")
    assert sanitize_path("/v1/messages?query-secret-canary") == "/v1/messages"

import json
from pathlib import Path

import pytest

from mitm_inspector.capture.redaction import REDACTED, sanitize_header, sanitize_path
from mitm_inspector.protocol import ProtocolError, parse_message

FIXTURE_PATH = Path(__file__).parents[1] / "contracts" / "fixtures" / "stream.json"


def fixture_messages() -> list[object]:
    return json.loads(FIXTURE_PATH.read_text())


def test_shared_fixture_is_accepted_and_preserves_ordered_duplicates() -> None:
    messages = [parse_message(message) for message in fixture_messages()]
    metadata = messages[1]["metadata"]
    assert metadata["request_headers"][1:3] == [
        {"name": "x-trace", "value": "first"},
        {"name": "x-trace", "value": "second"},
    ]
    lifecycle = [message for message in messages if message["type"] == "flow.lifecycle"]
    assert [message["state"] for message in lifecycle] == ["response_started", "request_end"]


def test_body_states_are_distinct() -> None:
    messages = [parse_message(message) for message in fixture_messages()]
    bodies = [
        messages[1]["metadata"]["request_body"],
        messages[1]["metadata"]["response_body"],
        messages[5]["body"],
    ]
    assert [body["state"] for body in bodies] == ["captured", "missing", "truncated"]


def test_unknown_types_and_additive_fields_are_tolerated() -> None:
    message = parse_message(fixture_messages()[-1])
    assert message["type"] == "future.message"
    assert message["sequence"] == "9007199254740993"
    assert message["future_additive_field"] is True


def test_decimal_values_are_strings() -> None:
    with pytest.raises(ProtocolError, match="decimal string"):
        parse_message(
            {
                "protocol_version": "1",
                "type": "stream.gap",
                "expected_sequence": 1,
                "actual_sequence": "2",
            }
        )


def test_fixtures_have_no_secret_canaries() -> None:
    serialized = json.dumps(fixture_messages()).lower()
    for canary in (
        "authorization-canary",
        "x-api-key-canary",
        "cookie-canary",
        "query-secret-canary",
        "contract-secret",
    ):
        assert canary not in serialized


def test_redaction_happens_before_transport() -> None:
    assert sanitize_header("Authorization", "Bearer authorization-canary") == (
        "authorization",
        REDACTED,
    )
    assert sanitize_header("X-Trace", "safe-value") == ("x-trace", "safe-value")
    assert sanitize_path("/v1/messages?query-secret-canary") == "/v1/messages"

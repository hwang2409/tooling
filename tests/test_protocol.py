import base64
import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest
from jsonschema import Draft202012Validator

from mitm_inspector import protocol as protocol_module
from mitm_inspector.api.transport import encode_for_browser
from mitm_inspector.capture.redaction import REDACTED, sanitize_header, sanitize_path
from mitm_inspector.protocol import (
    FrozenJsonObject,
    KnownParsedMessage,
    OpaqueParsedMessage,
    ParsedMessage,
    ProtocolError,
    parse_message,
)
from mitm_inspector.store.memory import MemoryStore

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


def known(message: object) -> FrozenJsonObject:
    parsed = parse_message(message)
    assert isinstance(parsed, KnownParsedMessage)
    return parsed.message


def known_fixture_messages() -> list[object]:
    return [message for message in fixture_messages() if message["type"] != "future.message"]


def body_chunk_message() -> dict[str, object]:
    return {
        "protocol_version": "1",
        "type": "body.chunk",
        "flow_id": "f",
        "body_side": "request",
        "chunk_index": "0",
        "offset_bytes": "0",
        "data_base64": "",
        "extension": {"nested": [{"value": "before"}]},
    }


class HostileString(str):
    """A string whose Python-level behavior disagrees with its stored text."""

    def __eq__(self, other: object) -> bool:
        return other == "body.chunk"

    def __hash__(self) -> int:
        return hash("body.chunk")

    def __str__(self) -> str:
        return "spoofed-text"

    def lower(self) -> str:
        return "content-type"


class DuplicateTextKey(str):
    def __eq__(self, _other: object) -> bool:
        return False

    def __hash__(self) -> int:
        return hash("different-key")


class HostileInt(int):
    def __int__(self) -> int:
        return 999


class HostileFloat(float):
    def __float__(self) -> float:
        return 999.0


def assert_deep_plain_json(value: object) -> None:
    if isinstance(value, dict):
        assert all(type(key) is str for key in value)
        for item in value.values():
            assert_deep_plain_json(item)
    elif isinstance(value, list):
        for item in value:
            assert_deep_plain_json(item)
    else:
        assert value is None or type(value) in {bool, float, int, str}


def body_descriptors(message: Mapping[str, object]) -> list[Mapping[str, object]]:
    message_type = message.get("type")
    if message_type == "flow.metadata":
        metadata = message["metadata"]
        assert isinstance(metadata, Mapping)
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
    messages = [known(message) for message in known_fixture_messages()]
    metadata = messages[1]["metadata"]
    assert metadata["request_headers"][1:3] == (
        {"name": "x-trace", "value": "first"},
        {"name": "x-trace", "value": "second"},
    )
    conformance_metadata = known(conformance()["valid"][1])["metadata"]
    assert conformance_metadata["request_headers"][2]["value"] == ""
    assert conformance_metadata["request_body"]["content_type"] == ""
    lifecycle = [message for message in messages if message["type"] == "flow.lifecycle"]
    assert [message["state"] for message in lifecycle] == ["response_started", "request_end"]


def test_body_counts_and_states_are_distinct() -> None:
    messages = [known(message) for message in known_fixture_messages()]
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
    assert isinstance(message, OpaqueParsedMessage)
    assert message.original_type == "future.message"
    assert message.original_type == message.payload["type"]
    assert message.payload["sequence"] == "9007199254740993"
    future = parse_message(conformance()["valid"][-1])
    assert isinstance(future, OpaqueParsedMessage)
    assert future.original_type == "future.additive"
    assert future.original_type == future.payload["type"]
    assert future.payload["sequence"] == "18446744073709551616"


def test_string_subclasses_are_normalized_before_type_discrimination_and_emission() -> None:
    raw = body_chunk_message()
    raw["type"] = HostileString("future.evil")

    parsed = parse_message(raw)
    assert isinstance(parsed, OpaqueParsedMessage)
    assert type(parsed.original_type) is str
    assert parsed.original_type == "future.evil"
    assert type(parsed.payload["type"]) is str

    store = MemoryStore()
    store.append(parsed)
    wire = encode_for_browser(next(store.newest_first()))
    assert_deep_plain_json(wire)
    assert wire["type"] == "future.evil"
    assert json.loads(json.dumps(wire)) == wire


def test_string_mapping_keys_and_values_are_exact_builtins_before_validation() -> None:
    raw = body_chunk_message()
    del raw["type"]
    raw[HostileString("type")] = "body.chunk"
    raw["flow_id"] = HostileString("f")

    parsed = parse_message(raw)
    assert isinstance(parsed, KnownParsedMessage)
    wire = encode_for_browser(parsed)
    assert_deep_plain_json(wire)
    assert wire["type"] == "body.chunk"
    assert wire["flow_id"] == "f"
    assert json.loads(json.dumps(wire)) == wire


def test_keys_that_collide_after_string_normalization_are_rejected() -> None:
    raw: dict[str, object] = {
        "protocol_version": "1",
        "type": "future.message",
    }
    raw[DuplicateTextKey("type")] = "future.evil"

    with pytest.raises(ProtocolError, match="duplicate keys"):
        parse_message(raw)


@pytest.mark.parametrize("scalar", [HostileInt(1), HostileFloat(1.5)])
def test_numeric_scalar_subclasses_are_rejected(scalar: object) -> None:
    with pytest.raises(ProtocolError, match="scalar subclasses"):
        parse_message(
            {
                "protocol_version": "1",
                "type": "future.message",
                "extension": scalar,
            }
        )


def test_parsed_messages_are_nominal_non_overlapping_and_recursively_immutable() -> None:
    raw_known = body_chunk_message()
    known_message = parse_message(raw_known)
    assert isinstance(known_message, KnownParsedMessage)
    assert not hasattr(known_message, "payload")

    raw_known["flow_id"] = "mutated"
    raw_known["extension"]["nested"][0]["value"] = "after"
    assert known_message.message["flow_id"] == "f"
    extension = known_message.message["extension"]
    assert isinstance(extension, Mapping)
    nested = extension["nested"]
    assert isinstance(nested, tuple)
    nested_value = nested[0]
    assert isinstance(nested_value, Mapping)
    assert nested_value["value"] == "before"
    with pytest.raises(TypeError):
        known_message.message["flow_id"] = "forged"  # type: ignore[index]
    with pytest.raises(TypeError):
        nested_value["value"] = "forged"  # type: ignore[index]

    raw_opaque = {
        "protocol_version": "1",
        "type": "future.message",
        "extension": {"values": ["before"]},
    }
    opaque_message = parse_message(raw_opaque)
    assert isinstance(opaque_message, OpaqueParsedMessage)
    assert not hasattr(opaque_message, "message")
    raw_opaque["type"] = "body.chunk"
    raw_opaque["extension"]["values"][0] = "after"
    assert opaque_message.original_type == opaque_message.payload["type"] == "future.message"
    opaque_extension = opaque_message.payload["extension"]
    assert isinstance(opaque_extension, Mapping)
    assert opaque_extension["values"] == ("before",)


def test_opaque_messages_reject_non_json_cycles() -> None:
    extension: dict[str, object] = {}
    extension["self"] = extension
    with pytest.raises(ProtocolError, match="cycles"):
        parse_message(
            {"protocol_version": "1", "type": "future.message", "extension": extension}
        )


def test_ingress_rejects_token_constructed_invalid_known_wrapper() -> None:
    invalid_payload = body_chunk_message()
    invalid_payload["flow_id"] = ""
    constructed = KnownParsedMessage(
        cast(FrozenJsonObject, invalid_payload),
        _token=protocol_module._PARSE_TOKEN,
    )
    with pytest.raises(ProtocolError, match="flow_id"):
        MemoryStore().append(constructed)
    with pytest.raises(ProtocolError, match="flow_id"):
        encode_for_browser(constructed)


def test_ingress_canonicalizes_object_new_wrapper_and_breaks_nested_aliases() -> None:
    mutable_payload = body_chunk_message()
    forged = object.__new__(KnownParsedMessage)
    object.__setattr__(forged, "_message", mutable_payload)

    store = MemoryStore()
    store.append(forged)
    stored = next(store.newest_first())
    assert isinstance(stored, KnownParsedMessage)
    assert stored is not forged

    mutable_payload["flow_id"] = "mutated"
    mutable_payload["extension"]["nested"][0]["value"] = "after"
    assert stored.message["flow_id"] == "f"
    extension = stored.message["extension"]
    assert isinstance(extension, Mapping)
    assert extension["nested"][0]["value"] == "before"


def test_ingress_revalidates_payload_replaced_after_parse() -> None:
    parsed = parse_message(body_chunk_message())
    assert isinstance(parsed, KnownParsedMessage)
    tampered_payload = body_chunk_message()
    object.__setattr__(parsed, "_message", tampered_payload)

    store = MemoryStore()
    store.append(parsed)
    stored = next(store.newest_first())
    assert stored is not parsed
    tampered_payload["extension"]["nested"][0]["value"] = "after"
    assert isinstance(stored, KnownParsedMessage)
    extension = stored.message["extension"]
    assert isinstance(extension, Mapping)
    assert extension["nested"][0]["value"] == "before"


def test_ingress_canonicalizes_valid_opaque_wrapper() -> None:
    mutable_payload = {
        "protocol_version": "1",
        "type": "future.message",
        "extension": {"values": ["before"]},
    }
    forged = object.__new__(OpaqueParsedMessage)
    object.__setattr__(forged, "_original_type", "future.message")
    object.__setattr__(forged, "_payload", mutable_payload)

    store = MemoryStore()
    store.append(forged)
    stored = next(store.newest_first())
    assert isinstance(stored, OpaqueParsedMessage)
    assert stored is not forged
    mutable_payload["extension"]["values"][0] = "after"
    extension = stored.payload["extension"]
    assert isinstance(extension, Mapping)
    assert extension["values"] == ("before",)
    assert encode_for_browser(stored) == {
        "protocol_version": "1",
        "type": "future.message",
        "extension": {"values": ["before"]},
    }


@pytest.mark.parametrize(
    ("original_type", "payload_type"),
    [
        ("future.message", "different.future"),
        ("future.message", "body.chunk"),
        ("body.chunk", "body.chunk"),
    ],
)
def test_ingress_rejects_object_new_spoofed_opaque_wrappers(
    original_type: str,
    payload_type: str,
) -> None:
    spoofed = object.__new__(OpaqueParsedMessage)
    object.__setattr__(spoofed, "_original_type", original_type)
    object.__setattr__(
        spoofed,
        "_payload",
        {"protocol_version": "1", "type": payload_type},
    )
    with pytest.raises(ProtocolError):
        MemoryStore().append(spoofed)
    with pytest.raises(ProtocolError):
        encode_for_browser(spoofed)


def test_transport_returns_independent_deep_plain_json() -> None:
    raw = body_chunk_message()
    parsed = parse_message(raw)
    assert isinstance(parsed, KnownParsedMessage)

    wire = encode_for_browser(parsed)
    assert_deep_plain_json(wire)
    assert wire == raw
    assert json.loads(json.dumps(wire)) == wire

    wire["flow_id"] = "wire-mutated"
    wire["extension"]["nested"][0]["value"] = "wire-mutated"
    assert parsed.message["flow_id"] == "f"
    parsed_extension = parsed.message["extension"]
    assert isinstance(parsed_extension, Mapping)
    assert parsed_extension["nested"][0]["value"] == "before"

    store = MemoryStore()
    store.append(parsed)
    stored = next(store.newest_first())
    assert stored is not parsed
    stored_wire = encode_for_browser(stored)
    stored_wire["extension"]["nested"][0]["value"] = "after"
    assert encode_for_browser(stored)["extension"]["nested"][0]["value"] == "before"


def test_store_reads_never_expose_retained_wrappers() -> None:
    store = MemoryStore()
    store.append(parse_message(body_chunk_message()))

    first_read = next(store.newest_first())
    replacement_raw = body_chunk_message()
    replacement_raw["flow_id"] = "valid-replacement"
    replacement = parse_message(replacement_raw)
    assert isinstance(first_read, KnownParsedMessage)
    assert isinstance(replacement, KnownParsedMessage)
    object.__setattr__(first_read, "_message", replacement.message)
    assert encode_for_browser(first_read)["flow_id"] == "valid-replacement"

    second_read = next(store.newest_first())
    assert second_read is not first_read
    assert encode_for_browser(second_read)["flow_id"] == "f"
    object.__setattr__(second_read, "_message", replacement.message)
    assert encode_for_browser(next(store.newest_first()))["flow_id"] == "f"


def test_transport_json_roundtrips_every_valid_known_and_opaque_message() -> None:
    for raw in [*fixture_messages(), *conformance()["valid"]]:
        wire = encode_for_browser(parse_message(raw))
        assert_deep_plain_json(wire)
        assert json.loads(json.dumps(wire)) == raw


@pytest.mark.parametrize(
    "forged",
    [
        {"kind": "known", "message": {"protocol_version": "1", "type": "source.hello"}},
        {
            "kind": "unknown",
            "original_type": "future.message",
            "payload": {"protocol_version": "1", "type": "different.future"},
        },
        {
            "kind": "unknown",
            "original_type": "future.message",
            "payload": {"protocol_version": "1", "type": "body.chunk"},
        },
    ],
)
def test_store_and_transport_reject_forged_or_spoofed_envelopes(
    forged: dict[str, object],
) -> None:
    structural_lookalike = cast(ParsedMessage, forged)
    with pytest.raises(ProtocolError, match="parsed wrapper"):
        MemoryStore().append(structural_lookalike)
    with pytest.raises(ProtocolError, match="parsed wrapper"):
        encode_for_browser(structural_lookalike)


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


@pytest.mark.parametrize(
    "name",
    [
        "Accept",
        "ACCEPT-ENCODING",
        "accept-language",
        "Cache-Control",
        "Content-Encoding",
        "Content-Length",
        "Content-Range",
        "Content-Type",
        "Date",
        "ETag",
        "Expires",
        "Host",
        "Last-Modified",
        "Range",
        "Server",
        "Traceparent",
        "Transfer-Encoding",
        "User-Agent",
        "X-B3-TraceId",
        "X-Correlation-ID",
        "X-Request-ID",
        "X-Trace-ID",
    ],
)
def test_redaction_exposes_only_explicit_safe_header_values(name: str) -> None:
    assert sanitize_header(name, "safe-value") == (name, "safe-value")


@pytest.mark.parametrize(
    "name",
    [
        "Authorization",
        "X_Access_Token",
        "XAccessToken",
        "X-Refresh-Token",
        "XRefreshToken",
        "X-ID-Token",
        "XIDToken",
        "X-Bearer-Token",
        "XBearerToken",
        "X-Token",
        "XToken",
        "X-ApiKey",
        "XApiKey",
        "X.Token",
        "X.Access.Token",
        "X.Api.Key",
        "X+Api+Key",
        "XAuthToken",
        "X_Credential",
        "X-Custom_Secret",
        "X-Signature",
        "Cookie",
        "Content-Key",
        "X-Trace",
        "X-Token-Count",
        "X-Key-ID",
        "X-Secret-Version",
        "X-Credential-Type",
        "X-Signature-Version",
        "X-Cookie-State",
        "X-Auth-Mode",
        "X.Token.Count",
        "X+Key+ID",
        "X.Secret.Version",
        "X.Api.Key.Version",
        "X-Api-Key-Suffix",
        "XApiKeySuffix",
        "X-Request-ID-Suffix",
        "Unknown-Custom-Header",
    ],
)
def test_redaction_redacts_all_unlisted_prior_and_suffix_variants(name: str) -> None:
    assert sanitize_header(name, "credential-canary") == (name, REDACTED)


@pytest.mark.parametrize("punctuation", list("!#$%&'*+-.^_`|~"))
def test_redaction_redacts_every_punctuation_variant(punctuation: str) -> None:
    name = f"X{punctuation}Api{punctuation}Key"
    assert sanitize_header(name, "credential-canary") == (name, REDACTED)


@pytest.mark.parametrize(
    "name",
    [
        "",
        " Content-Type",
        "Content-Type ",
        "Content\x00Type",
        "Content\nType",
        "Contént-Type",
        "Ｃontent-Type",
        "🔥",
    ],
)
def test_redaction_preserves_but_never_exposes_invalid_header_names(name: str) -> None:
    assert sanitize_header(name, "credential-canary") == (name, REDACTED)


def test_redaction_preserves_header_name_order() -> None:
    headers = [
        ("X-Custom", "secret"),
        ("Content-Type", "application/json"),
        ("X-Custom", "second-secret"),
    ]
    assert [sanitize_header(name, value) for name, value in headers] == [
        ("X-Custom", REDACTED),
        ("Content-Type", "application/json"),
        ("X-Custom", REDACTED),
    ]


def test_redaction_normalizes_hostile_string_subclasses_before_allowlist_comparison() -> None:
    name, value = sanitize_header(
        HostileString("Authorization"),
        HostileString("credential-canary"),
    )
    assert type(name) is type(value) is str
    assert (name, value) == ("Authorization", REDACTED)

    safe_name, safe_value = sanitize_header(
        HostileString("Content-Type"),
        HostileString("application/json"),
    )
    assert type(safe_name) is type(safe_value) is str
    assert (safe_name, safe_value) == ("Content-Type", "application/json")


def test_query_material_is_dropped() -> None:
    assert sanitize_path("/v1/messages?query-secret-canary") == "/v1/messages"

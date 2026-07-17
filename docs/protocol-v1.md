# Protocol v1

Protocol-v1 is a JSON message stream owned by this repository. Every message
has `protocol_version: "1"` and a string `type`. The JSON Schema,
`src/mitm_inspector/protocol.py`, and `web/src/protocol.ts` share the same
known-message vocabulary and invariants. Unknown types use the base schema
branch; additive fields on known or unknown messages are retained.

Parsing deep-copies protocol input into nominal, recursively immutable values:
`{kind:"known", message:<KnownMessage>}` for known types, or
`{kind:"unknown", original_type, payload}` for an opaque unknown type. Known and
opaque values are non-overlapping, `original_type` always equals `payload.type`,
and a known vocabulary name cannot enter the opaque branch. Store and transport
boundaries verify the nominal parsed value at runtime; raw structural lookalikes
are accepted only by the parser and cannot mutate retained content by alias.

## Numeric and ordering rules

All unsigned 64-bit values are decimal strings in the inclusive range
`0..18446744073709551615`. This includes `port`, source limits, lifecycle
`sequence`, body counts, chunk indexes/offsets, cursors, gap values, and
`dropped_count`. Python uses bounded integer conversion; TypeScript uses
`BigInt`, never JavaScript `number`. A `stream.gap` has
`actual_sequence > expected_sequence`; when present,
`dropped_count = actual_sequence - expected_sequence - 1`.

Lifecycle events are independent observations, not a request/response state
machine. Consumers must not assume request completion precedes response start;
`response_started` may arrive before `request_end`. Valid states are:
`request_started`, `request_headers`, `request_body`, `request_end`,
`response_started`, `response_headers`, `response_body`, `response_end`,
`error`, and `flow_completed`.

## Message shapes

- `source.hello`: `source_id`, `occurred_at`, capabilities
  `{body_chunks:boolean, redaction:"headers-and-query"}`, and limits
  `{max_body_prefix_bytes, max_in_memory_bytes}`.
- `flow.metadata`: a sanitized `metadata` flow with non-empty identity fields,
  `scheme` `http|https`, decimal-string `port`, and ordered header arrays.
- `flow.lifecycle`: `source_id`, `flow_id`, `event_id`, `occurred_at`, bounded
  decimal-string `sequence`, and one lifecycle state above.
- `body.chunk`: `flow_id`, `body_side` `request|response`, bounded decimal-string
  `chunk_index`/`offset_bytes`, and valid base64 `data_base64`.
- `body.end`: `flow_id`, `body_side`, bounded `total_bytes`, and a body
  descriptor whose non-missing `size_bytes` equals `total_bytes`.
- `stream.gap`: bounded `expected_sequence`, `actual_sequence`, and optional
  bounded `dropped_count` following the arithmetic rule above.
- `browser.snapshot`: `snapshot_id`, bounded `cursor`, and complete flow list.
- `browser.delta`: bounded `cursor` and changes with only `upsert {flow}` or
  `remove {flow_id}` operations.
- `browser.resync`: `reason` `cursor_gap|history_evicted|initial_connect` and
  bounded `requested_cursor`.

## Body states

Body descriptors distinguish:

- `missing`: no body was observed or retained; no counts, encoding, or data.
- `empty`: observed and exactly zero bytes; `size_bytes` must be `"0"`.
- `captured`: complete retained data; `size_bytes` equals decoded base64 bytes.
- `truncated`: total `size_bytes` is known, `captured_bytes <= size_bytes`,
  and decoded base64 prefix length does not exceed `captured_bytes`.

`content_type` is an optional string on every body state. Captured data is
base64 only; `hex` and arbitrary body-side/state values are invalid. The
policy permits empty header values, empty base64 data where a zero-byte chunk
or body makes that meaningful, and an explicitly present empty `content_type`.
Identifiers, header names, paths, directions, and enum values remain
non-empty/validated.
The shared `contracts/fixtures/conformance.json` contains positive and adversarial
negative cases exercised by the schema, Python, and TypeScript tests. JSON
Schema enforces the types, vocabularies, and bounds; the Python/TypeScript
boundary validators enforce cross-field arithmetic that JSON Schema cannot
express over decimal strings.

Body data is sensitive and is exposed only when a user selects a flow in later
work. S0 has no raw export or replay endpoint.

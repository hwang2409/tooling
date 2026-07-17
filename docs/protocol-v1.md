# Protocol v1

Protocol-v1 is a JSON message stream owned by this repository. Every message
has `protocol_version: "1"` and a string `type`. Unknown additive fields and
unknown message types must be retained/ignored by clients so a newer source can
talk to an older browser. The machine-readable baseline is
[`contracts/protocol-v1.schema.json`](../contracts/protocol-v1.schema.json).

All unsigned 64-bit values are decimal strings, including cursors, sequence
numbers, byte counts, offsets, and chunk indexes. Clients must never parse them
through JavaScript `number`; the fixture includes a value above `2^53`.

## Message families

- `source.hello`: source identity, protocol version, capabilities, and limits.
- `flow.metadata`: sanitized request/response metadata. Headers are ordered
  `{name, value}` arrays, not maps, so duplicate headers survive unchanged.
- `flow.lifecycle`: independent lifecycle events keyed by `flow_id` and
  sequenced with decimal strings. Consumers must not assume request completion
  precedes response start; the fixture intentionally models response-before-
  request-end ordering.
- `body.chunk`: incremental base64 bytes for a request or response side.
- `body.end`: total bytes plus a body descriptor.
- `stream.gap`: source sequence discontinuity, requiring a browser resync.
- `browser.snapshot`: complete bounded read model at a cursor.
- `browser.delta`: ordered upsert/remove changes from the previous cursor.
- `browser.resync`: a browser request for a fresh snapshot after a gap.

## Body states

Body descriptors distinguish:

- `missing`: no body was observed or retained.
- `empty`: body was observed and has exactly zero bytes.
- `captured`: complete retained prefix/body, with base64 data.
- `truncated`: total size is known but only a bounded captured prefix is present.

Body data is sensitive and is exposed only when a user selects a flow in later
work. S0 has no raw export or replay endpoint.

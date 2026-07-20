# API boundary

The `api` boundary delivers protocol-v1 messages from the bounded in-memory
store to the browser over a versioned loopback HTTP/WebSocket surface, and
accepts capture messages on the private Unix ingest socket allocated by the
runtime.  It owns no capture decisions and no browser rendering.

`python -m mitm_inspector.api.server` accepts exactly the argv contract the
runtime reserves for the app child (`--host`, `--port`, `--proxy-port`,
`--max-retained-flows`, `--retention-seconds`, `--max-body-bytes`,
`--max-body-prefix-bytes`, `--capture-socket`, `--capture-source-id`, and the
`--capture-max-*` mirrors).  The host must be loopback; a non-loopback bind is
refused at configuration time, never repaired.

## HTTP surface (`/api/v1`)

| Path | Meaning |
| --- | --- |
| `GET /` | Minimal informational page for the browser-open affordance |
| `GET /api/v1/health` | Readiness/health JSON with application and store counters |
| `GET /api/v1/counters` | Top-level counter object with the same shape and keys as `/api/v1/health.counters` |
| `GET /api/v1/snapshot` | One validated `browser.snapshot` message |
| `GET /api/v1/stream` | WebSocket upgrade for the live session |
| `GET /api/v1/flows/<flow_id>` | Every retained message for one selected flow, oldest first |
| `GET /api/v1/search?q=<text>&limit=<n>` | Durable body matches, newest first (default 50, maximum 200) |

Every request must carry exactly one loopback `Host` header (DNS-rebinding
defense); request bodies, non-GET methods, oversized heads, folded headers,
and malformed request lines are rejected.  Responses are `Connection: close`
with `Cache-Control: no-store` and `X-Content-Type-Options: nosniff`.

## WebSocket session

The handshake requires version 13, a valid 16-byte key, and — when an
`Origin` header is present — a loopback `http(s)` origin; cross-site pages
cannot open the stream.  An absent origin (non-browser client) is accepted.

On connect the server sends, in order: a synthesized `source.hello`
(source id and limits from configuration), `browser.resync` with reason
`initial_connect` at the current cursor, and a `browser.snapshot`.  After
that the session receives:

- `browser.delta` messages whose cursor increments by exactly one per delta,
  shared across all sessions.  Changes are `upsert`/`remove` operations
  derived from the store's newest `flow.metadata` per flow; store eviction
  and age expiry surface as `remove` operations (an idle sweep task publishes
  expiry without traffic).
- Relayed `flow.lifecycle`, `stream.gap`, `source.hello`, and unknown-type
  messages, byte-independent copies with additive fields retained.
- Nothing else: `body.chunk`, `body.end`, and raw `flow.metadata` are retained
  in the store but never broadcast.

A client may send only `browser.resync` requests (text frames, one JSON
message per frame); the server answers with a `browser.resync` echo followed
by a fresh snapshot at the current cursor.  Any other client payload, any
unmasked or malformed frame, and any message over 64 KiB closes the session
with a protocol close code.  A slow consumer whose bounded send queue (256
frames) overflows is disconnected; reconnecting yields a coherent snapshot.
The shared cursor is a bounded u64; exhaustion is a stable terminal state
that drops live streams instead of emitting an unrepresentable cursor.

Every outbound frame is round-tripped through full protocol validation and
serialized once, so no mapping proxy, tuple, or mutable alias reaches a
transport, and mutating ingest input after the fact cannot alter retained or
published state.

## Body redaction on the grid stream

Bodies are sensitive and are delivered only on selection.  Snapshot and delta
flow entries pass through a body redaction projection: `captured` and
`truncated` descriptors become schema-valid `truncated` descriptors with
`captured_bytes: "0"` and empty `data`, preserving `size_bytes` and
`content_type` for the grid.  `missing` and `empty` descriptors pass through
unchanged.  The only surface that carries body bytes is
`GET /api/v1/flows/<flow_id>`, which returns the full retained messages
(metadata, chunks, body ends, lifecycle) for one explicitly selected flow.
Gzip/deflate metadata and terminal body descriptors are decoded on this
surface; compressed chunk messages are omitted when the decoded terminal
descriptor is available. Stored sqlite bytes are never rewritten.

`GET /api/v1/search` performs a case-insensitive substring search over durable
sqlite request bodies and decodable response bodies. It returns at most the
requested number of `{flow_id, field, snippet, flow}` objects, with whitespace-
collapsed snippets bounded to 160 characters and a `truncated` flag when more
matches exist. `flow` is the same enriched, body-redacted projection used by
snapshot rows, so matches older than the in-memory snapshot remain renderable;
the three original match fields remain stable for older clients. Search runs
against a persisted trigram text projection, materializes at most `limit + 1`
candidates, and a newer request cooperatively cancels an older scan. A missing
or empty `q` is rejected with HTTP 400.

## Capture ingest socket

When `--capture-socket` is provided the server binds the runtime-allocated
endpoint (mode `0600` inside the mode-`0700` run directory) and accepts
newline-delimited JSON, one protocol-v1 message per line, at most 8 MiB per
line.  Both the runtime and the app configuration reject a
`max_body_prefix_bytes` whose two base64 body prefixes plus the metadata
envelope could exceed one bounded line (`api/limits.py`), so a legal capture
configuration can never produce lines the listener would drop.  Parsing is
strict: duplicate object keys and non-finite numbers are rejected.  Ingest is fail-closed — a malformed line, an invalid protocol
message, or an out-of-bounds number drops that producer connection and
increments `rejected_ingest_lines`; accepted messages are revalidated,
deep-copied, appended to the bounded store, relayed per the rules above, and
reflected in the grid projection.  A pre-existing non-socket file at the
endpoint refuses startup; a stale socket is replaced.

The proxy-side writer that drains `CaptureAddon` into this socket is
integration work (I1); B2's addon exposes the endpoint configuration and the
durable drain/acknowledge contract this listener is built for.

## Bounds and observability

The store keeps the configured flow/age/body/memory bounds
(`docs/capture.md`).  `GET /api/v1/health` exposes application counters
(`ingested_messages`, `relayed_messages`, `emitted_deltas`,
`emitted_snapshots`, `dropped_subscribers`, `subscribers_partial_history`,
`resync_responses`, `cursor`,
`cursor_exhausted`, `subscribers`, `published_flows`), server counters
(`rejected_ingest_lines`, `ingest_connections`, `http_requests`,
`websocket_connections`, `sweep_failures`), and the nested store counters.

`subscribers_partial_history` counts subscribers whose historical replay was
truncated because the transport queue could not hold every retained lifecycle
frame; it is not incremented for the normal case or for initial-frame
refusals.

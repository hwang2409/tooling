# Capture and bounded retention

`CaptureAddon` is a small adapter over the documented mitmproxy 12.2.x HTTP
hooks: `requestheaders`, `request`, `responseheaders`, `response`, and `error`.
Its documented `load` hook parses configuration and the module exports
`addons = [CaptureAddon()]`, so `mitmdump -s` loads a real addon. It imports
only `mitmproxy.http`; it does not use mitmweb, view, proxy-layer, or
dynamic-loading APIs.

Configuration is parsed at addon load, not by an import-time connection or
thread. The supported environment variables are
`MITM_INSPECTOR_CAPTURE_SOCKET` (optional absolute POSIX Unix-socket path),
`MITM_INSPECTOR_SOURCE_ID`, `MITM_INSPECTOR_MAX_BODY_PREFIX_BYTES`,
`MITM_INSPECTOR_MAX_IN_MEMORY_BYTES`, and
`MITM_INSPECTOR_MAX_PENDING_MESSAGES`. Defaults are a 1 MiB body prefix, 128
MiB in-memory budget, and 4,096 pending messages. The parsed endpoint is
exposed as `CaptureAddon.capture_socket`; IPC transport belongs to B3 and is
not opened here. A caller can inject a `BoundedMessageSink` directly.

At the header hooks the addon copies request identity and headers into owned
values, strips the query from the path, applies the fail-closed header policy,
and installs the public `Message.stream` callback. The callback copies only a
bounded prefix for inspection, emits project-owned `body.chunk` messages, and
returns the original chunk unchanged. A slow consumer cannot block forwarding
when the default `BoundedMessageSink` is used: `offer` is `put_nowait`-style,
and a full queue increments `dropped_count`.

The terminal `request` and `response` hooks finalize body descriptors and emit
`body.end`; `error` emits the error lifecycle state without exposing error text.
The terminal hooks also emit `flow_completed`. If response completion arrives
before request completion, `flow_completed` is deferred until request body/end
observation is emitted; an error always emits request body/end first, including
when content is missing. Lifecycle observations are deduplicated per flow, but
ordering is not imposed: a response can start and stream before the request end
observation.

Body descriptors distinguish `missing`, `empty`, `captured`, and `truncated`.
Counts are decimal uint64 strings and captured data is base64. The default
captured prefix is 1 MiB per side. Query values and non-reviewed header values
are irreversibly removed before any message reaches a sink or store; bodies
are intentionally retained only as bounded prefixes for local selection.

Every queued capture message gets an additive monotonic `delivery_position`.
Queue-full, body-budget, memory-budget, and lock-contention drops are published
as synchronized delivery events; `drain` coalesces them into valid
`stream.gap` messages before the next retained message and also flushes a final
gap when no later retained message exists. Callback delivery, when configured,
happens only during explicit `CaptureAddon.drain`, never from a mitmproxy
stream callback.

`MemoryStore` retains project-owned parsed messages grouped by flow. It keeps
the newest 2,000 completed flows or 30 minutes, whichever evicts first, and
accounts decoded body-prefix bytes against a 128 MiB global budget, including
incomplete flows, standalone messages, and body descriptors nested in browser
snapshot/delta envelopes. A separate canonical-memory counter includes all
nested and additive fields, so arbitrary envelopes cannot bypass the same
bound. Global and per-flow message caps prevent lifecycle, zero-byte, or
arbitrary envelope floods from escaping bounds; terminal body/lifecycle
messages are retained while older SSE chunks are evicted. Active capture flows
are independently bounded by count, age, and copied metadata weight.
Metadata and terminal body updates are coalesced. Evictions, expiry, sink
drops, and body/memory-budget drops are observable through
`MemoryStore.counters`, `CaptureAddon.counters`, and
`BoundedMessageSink.dropped_count`.

Tests use synthetic fake flows and assert that the original input objects are
never emitted or retained. They do not start mitmproxy, capture traffic, or
use credentials.

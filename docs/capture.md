# Capture and bounded retention

`CaptureAddon` is a small adapter over the documented mitmproxy 12.2.x HTTP
hooks: `requestheaders`, `request`, `responseheaders`, `response`, and `error`.
It imports only `mitmproxy.http`; it does not use mitmweb, view, proxy-layer,
or dynamic-loading APIs.

At the header hooks the addon copies request identity and headers into owned
values, strips the query from the path, applies the fail-closed header policy,
and installs the public `Message.stream` callback. The callback copies only a
bounded prefix for inspection, emits project-owned `body.chunk` messages, and
returns the original chunk unchanged. A slow consumer cannot block forwarding
when the default `BoundedMessageSink` is used: `offer` is `put_nowait`-style,
and a full queue increments `dropped_count`.

The terminal `request` and `response` hooks finalize body descriptors and emit
`body.end`; `error` emits the error lifecycle state without exposing error text.
The terminal hooks also emit `flow_completed`. Lifecycle observations are
deduplicated per flow, but ordering is not imposed: a response can start and
stream before the request end observation.

Body descriptors distinguish `missing`, `empty`, `captured`, and `truncated`.
Counts are decimal uint64 strings and captured data is base64. The default
captured prefix is 1 MiB per side. Query values and non-reviewed header values
are irreversibly removed before any message reaches a sink or store; bodies
are intentionally retained only as bounded prefixes for local selection.

`MemoryStore` retains project-owned parsed messages grouped by flow. It keeps
the newest 2,000 completed flows or 30 minutes, whichever evicts first, and
accounts decoded body-prefix bytes against a 128 MiB global budget. Metadata
and terminal body updates are coalesced. Evictions, expiry, sink drops, and
body-budget drops are observable through `MemoryStore.counters` and
`BoundedMessageSink.dropped_count`.

Tests use synthetic fake flows and assert that the original input objects are
never emitted or retained. They do not start mitmproxy, capture traffic, or
use credentials.

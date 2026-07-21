# MITMWEB-B5 read-path audit

measurement date: 2026-07-21. source: Henry's live `flows.sqlite` (117 flows,
53.4 MiB decoded in-memory body accounting, 105 MiB sqlite file). The live
backend was left running; it was not restarted.

## timings

| stage | `c47a3c1d` (2 flows) | `2ce6b5ec` (24 flows) | `ff8b5ca9` (27 flows) |
| --- | ---: | ---: | ---: |
| sqlite `SELECT` / body fetch | 1.3 ms | 2.2 ms | 2.2 ms |
| durable message build + protocol parse | 530.5 ms | 938.2 ms | 942.5 ms |
| of that: metadata projection | 326.7 ms | 12.5 ms | 9.5 ms |
| of that: `parse_message` validation/copy | 200.1 ms | 891.3 ms | 899.6 ms |
| browser HTTP / JSON / React | unavailable | unavailable | unavailable |

The sqlite numbers are direct reads from a copy of the live database. The
message numbers call `SQLiteFlowStorage.flow_messages` for every flow, with
the existing 16 MiB per-detail bound. The HTTP stage could not be completed
reliably: `GET /api/v1/flows/<id>` produced no bytes within 10 seconds and
blocked the live event loop. A Playwright probe also observed the existing
WebSocket handshake closing before establishment through both Vite and the
backend. No backend restart was performed, so there are no fabricated client
or React timings.

## findings and cut list

1. sqlite is not the bottleneck: fetching all 27 target rows took 2.2 ms.
2. protocol validation/copy is the dominant durable cost: 891–900 ms for the
   24/27-flow sessions. `parsed_message_to_plain_json()` calls
   `require_parsed_message()`, which reparses and validates the complete body
   descriptor, including base64, before copying it again.
3. the in-memory detail path scans and sorts every retained message for every
   selected flow (`ApiApplication.flow_detail_text` -> `MemoryStore.newest_first`)
   and then performs the same wrapper revalidation. This explains why a live
   flow request can monopolize the event loop even though the underlying store
   is already resident.
4. `body_chunks` cannot be removed: the live database has 2,082 chunks across
   101 flows, including flows whose response body is still missing from the
   one-shot column. It is a real compatibility path, not dead abstraction.

The implementation cut is therefore limited to the measured hot path: resolve
one flow directly from the memory store, and serialize already-validated
retained messages without a second full protocol parse. The protocol schema,
redaction, sqlite schema, WebSocket stream, and body-chunk compatibility path
remain unchanged.

## post-fix replay measurement

Using the same copied live database replayed into `MemoryStore`, the selected
detail path changed as follows:

| session | before: scan/sort/revalidate | after: indexed trusted copy |
| --- | ---: | ---: |
| `c47a3c1d` (2 flows) | 11,632.3 ms | 17.8 ms |
| `2ce6b5ec` (24 flows) | not completed within the 30 s probe | 92.0 ms |
| `ff8b5ca9` (27 flows) | not completed within the 30 s probe | 98.9 ms |

The post-fix totals include the existing detail projection and JSON build for
each flow. The live server remained unavailable after the earlier request
wedged its event loop, so browser network/React timings must be re-captured
against a restarted backend by the operator.

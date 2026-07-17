# Architecture

S0 establishes a local-only, read-only inspector around stock mitmproxy. The
proxy remains the forwarding engine; the app does not wrap mitmweb or import
private mitmproxy modules.

## Boundaries

| Boundary | Owns | Does not own |
| --- | --- | --- |
| `runtime` | Process lifecycle and CLI composition | Proxy internals or UI state |
| `capture` | Documented mitmproxy hooks, sanitization, bounded emission | API routes, browser concerns, raw `Flow` retention |
| `store` | Bounded in-memory project-owned messages | Disk persistence or raw traffic |
| `api` | Future HTTP/WebSocket delivery of protocol-v1 messages | Capture decisions or browser rendering |
| `web/src/protocol.ts` | Browser validation and protocol typing | Server state or mitmproxy types |
| web shell | Developer-tool shell and connection affordance | Flow grid, inspector, replay/edit/intercept controls |
| tests | Contract, architecture, and future integration/e2e seams | Real proxy traffic |

The later delivery graph is `S0 → {runtime, capture, web shell} → {API,
flow workspace, inspector} → integration/e2e and packaging`.

## Data path

1. Stock `mitmdump` calls the addon's documented hooks.
2. The capture boundary removes sensitive header values and query material
   before a message can leave the proxy process.
3. A bounded store retains only project-owned sanitized metadata and bounded
   body prefixes in memory.
4. The future API emits lifecycle messages and browser snapshot/delta envelopes.
5. The web client validates protocol-v1 and renders read-only state.

S0 contains only the seams and inert hook adapter. It does not proxy, open
sockets, persist traffic, or capture real flows.

## Compatibility guard

`tests/test_architecture.py` parses every source Python module and rejects
imports from `mitmproxy.tools.web`, `mitmproxy.addons.view`,
`mitmproxy.proxy.layers`, and the private `mitmweb` namespace generally. Public
hooks are the only allowed mitmproxy integration surface.

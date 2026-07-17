# Security boundary

The app is designed for loopback-only use and read-only inspection. S0 does
not start a listener, authenticate a browser, proxy traffic, or persist data.

Sensitive values are removed before crossing the capture-to-app boundary. S0
uses a deliberately fail-closed safe-header value allowlist:

- Header names and duplicate ordering are preserved unchanged, but every value
  is `[REDACTED]` unless its case-insensitive name is explicitly allowlisted.
- The reviewed MVP allowlist covers content type/length/encoding, the accept
  family, cache metadata, date/etag/last-modified, host, user-agent, server,
  range, transfer/connection metadata, and explicit request/trace ID headers.
- Unknown, custom, credential-shaped, suffixed, malformed, Unicode, and control
  character names always retain their name but lose their value.
- S0 has no configuration or escape hatch for revealing another header value.

This intentionally favors privacy over visibility. Expanding the allowlist
requires a code and test change that judges the specific header value safe.
- Query material is dropped from paths until a query-aware allow-list exists.
- Raw mitmproxy `Flow` objects never enter the store, API, or browser contract.
- Body bytes are bounded prefixes in the future store; body data is not shown
  in list metadata.

The shared fixtures deliberately contain no authorization, API-key, cookie, or
query-secret canary values. `tests/test_protocol.py` enforces that property.
The architecture guard in `tests/test_architecture.py` protects against
accidental adoption of private mitmweb APIs or dynamic import machinery.

Future integration work must add loopback binding, browser authentication, and
explicit tests for redaction-before-transport before enabling live traffic.

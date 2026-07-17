# Security boundary

The app is designed for loopback-only use and read-only inspection. S0 does
not start a listener, authenticate a browser, proxy traffic, or persist data.

Sensitive values are removed before crossing the capture-to-app boundary:

- `authorization`, `proxy-authorization`, `cookie`, `set-cookie`, and
  `x-api-key` values become `[REDACTED]`.
- Query material is dropped from paths until a query-aware allow-list exists.
- Raw mitmproxy `Flow` objects never enter the store, API, or browser contract.
- Body bytes are bounded prefixes in the future store; body data is not shown
  in list metadata.

The shared fixtures deliberately contain no authorization, API-key, cookie, or
query-secret canary values. `tests/test_protocol.py` enforces that property.
The architecture guard in `tests/test_architecture.py` protects against
accidental adoption of private mitmweb APIs.

Future integration work must add loopback binding, browser authentication, and
explicit tests for redaction-before-transport before enabling live traffic.

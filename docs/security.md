# Security boundary

The app is designed for loopback-only use and read-only inspection. S0 does
not start a listener, authenticate a browser, proxy traffic, or persist data.

Sensitive values are removed before crossing the capture-to-app boundary. The
header policy is deliberately fail-closed for credential-shaped names:

- `authorization`, `proxy-authorization`, `cookie`, `set-cookie`, `api-key`,
  `x-api-key`, `x-auth-token`, `x-amz-security-token`, access/refresh/id
  tokens, and client secrets become `[REDACTED]`.
- Classification is case-insensitive and camel-case aware, and tokenizes every
  punctuation character allowed in an HTTP field name. Header names themselves
  are preserved unchanged. Credential-shaped terminal terms (`token`, `key`,
  `secret`, `credential`, `cookie`, `signature`) and auth/bearer segments are redacted;
  explicit harmless shape exceptions such as `x-token-count`, `x-key-id`,
  `x-secret-version`, and `x-signature-version` remain visible. This catches
  variants such as `X_Api_Key`, `XApiKey`, and `X-Auth_Token` without treating
  ordinary metadata headers as credentials.
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

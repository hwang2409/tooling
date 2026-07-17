# mitm-inspector

`mitm-inspector` is a local, read-only traffic-inspection app built around stock
mitmproxy. S0 establishes the ownership boundaries and a project-owned protocol
so capture, backend/API, and frontend work can proceed independently.

## Quick architecture

```text
stock mitmdump + public-hook addon
        │ bounded, sanitized protocol-v1 messages
        ▼
runtime / capture-redaction / bounded store
        │ API and browser snapshot/delta envelopes
        ▼
React + TypeScript web shell
```

The proxy process owns capture and irreversible header/query redaction. The
backend owns bounded in-memory state and API/WebSocket delivery. The browser is
read-only. The protocol never exposes mitmproxy `Flow` objects or private
mitmweb APIs. See [docs/architecture.md](docs/architecture.md) and
[docs/protocol-v1.md](docs/protocol-v1.md).

## Development

Requirements: Python 3.12+ and Node `^20.19.0 || ^22.13.0 || >=24` (the
checked-in commands were run with Python 3.12.13 and Node 26.0.0).

```sh
# Python: creates/uses only .venv in this checkout
uv sync --python 3.12
uv run ruff check .
uv run mypy
uv run pytest

# Web: creates/uses only web/node_modules in this checkout
cd web
npm ci
npm run lint
npm run typecheck
npm test -- --run
npm run build
```

No command starts a proxy, opens a socket, persists traffic, or installs a
global dependency. The fixture suite uses synthetic sanitized messages only.

## Ownership map

- `src/mitm_inspector/runtime`: process and CLI lifecycle boundary.
- `src/mitm_inspector/capture`: public mitmproxy hooks and redaction boundary.
- `src/mitm_inspector/store`: bounded memory retention boundary.
- `src/mitm_inspector/api`: future HTTP/WebSocket transport boundary.
- `web/src/protocol.ts`: browser-side protocol client boundary.
- `web/src/`: minimal shell; future `flow-workspace` and `inspector` features stay separate.
- `tests/`, `web/src/*.test.ts`: contract, architecture, and integration/e2e seams.

## License

MIT. See [LICENSE](LICENSE).

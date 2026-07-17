# Runtime boundary

The runtime owns local process topology and command composition. It does not
import mitmproxy internals, retain `Flow` objects, open a listener itself, or
make the future API server part of S0.

## Configuration

`RuntimeConfig` is frozen and validates its values at construction time. The
defaults are intentionally local and bounded:

- app: `127.0.0.1:8000`
- proxy: `127.0.0.1:8080`
- reverse upstream: `https://api.anthropic.com`
- retention: 2,000 completed flows or 30 minutes
- body budget: 128 MiB globally, with a 1 MiB captured prefix per side

Both app and proxy hosts must be loopback addresses (`localhost`, `127.0.0.1`,
or `::1`/another literal loopback address). Ports must be distinct and in the
TCP range 1–65535. Reverse upstreams must be absolute HTTP(S) URLs without
credentials, queries, or fragments. These checks fail closed before any child
can be spawned.

## Planning before B3

The safe surface available before the app server exists is:

```sh
mitm-inspector plan \
  --reverse-upstream https://api.anthropic.com \
  --app-port 8000
```

It prints the eventual two argv vectors and their lifecycle order. The
equivalent dry-run is `mitm-inspector run --dry-run`; neither command starts a
proxy, app server, listener, browser, or shell. The existing `--version`
entrypoint remains available. A live `run` intentionally exits with an
explanation until B3 provides the app server.

The planned commands are equivalent to:

```text
python -m mitm_inspector.api.server --host 127.0.0.1 --port 8000 --proxy-port 8080 ...
mitmdump --mode reverse:https://api.anthropic.com --listen-host 127.0.0.1 --listen-port 8080 -s src/mitm_inspector/capture/addon.py
```

The first child is started and made ready before the proxy child is spawned.
On shutdown, the proxy is asked to stop before the app. Each child receives a
bounded graceful shutdown request; still-running children are killed and
then waited on for up to a second for reaping. Unexpected child exit is
propagated as a `ChildExitedError` after the sibling is cleaned up. SIGINT, SIGTERM, and
`KeyboardInterrupt` all use the same idempotent cleanup path.

Readiness is an injected seam. S0's default probe only detects an immediate
child exit; B3 can provide a health/readiness probe without changing the
supervisor or its tests. Process creation is also injected, and production
creation always receives a list of argv strings with `shell=False`.

The browser is opened only when `open_browser` is explicitly enabled and both
children have passed readiness. Tests inject the browser opener, so test runs
never open a browser or start real processes.

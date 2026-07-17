# Runtime boundary

The runtime owns local process topology and command composition. It does not
import mitmproxy internals, retain `Flow` objects, open a listener itself, or
make the future API server part of S0.

## Configuration and preflight

`RuntimeConfig` and `CaptureIPCConfig` are frozen. Defaults are relocatable
from the checkout working directory:

- app executable: `sys.executable`
- proxy executable: the sibling `mitmdump` beside that interpreter
- addon: the absolute package-relative `src/mitm_inspector/capture/addon.py`
- app: `127.0.0.1:8000`
- proxy: `127.0.0.1:8080`
- reverse upstream: `https://api.anthropic.com`
- retention: 2,000 completed flows or 30 minutes
- body budget: 128 MiB globally, with a 1 MiB captured prefix per side
- pending-message budget: 4,096 queued capture messages

All uint64 limits are positive except `max_body_prefix_bytes`, which may be
zero. This matches B2's environment parser: emitted body memory and pending
message values are never zero.

Both app and proxy hosts must be loopback addresses. Ports must be distinct
and in the TCP range 1–65535. The reverse target accepts the pinned
mitmdump 12.2.3 authority grammar only: lowercase `http`/`https` (uppercase
schemes are rejected, never repaired), hostname or IP, optional port 1–65535,
no credentials, path, query, fragment, or empty port.

Live startup preflights both executable files, executable permissions, the
addon file, and the writable runtime base directory before spawning a child;
the private per-run directory is then allocated and validated before the first
child starts.
`plan` and `run --dry-run` do not fail just because a path is absent; they
include a `preflight` report listing every issue.

## Shared capture IPC contract

`CaptureIPCConfig` is the durable seam shared by proxy and future app. Every
live run allocates a collision-resistant private directory under the system
temporary directory with mode `0700`; its endpoint is
`<run-directory>/capture.sock`. Existing directories/endpoints are checked
with `lstat` (including ownership, mode, type, and symlink rejection) before
use. The socket and directory are removed after both children stop, including
failure paths; unexpected files or adversarial replacements fail closed.
The same explicit five-variable environment overlay is attached to both
`ProcessSpec` values:

```text
MITM_INSPECTOR_CAPTURE_SOCKET=/absolute/path/to/<run-directory>/capture.sock
MITM_INSPECTOR_SOURCE_ID=mitm-inspector
MITM_INSPECTOR_MAX_BODY_PREFIX_BYTES=1048576
MITM_INSPECTOR_MAX_IN_MEMORY_BYTES=134217728
MITM_INSPECTOR_MAX_PENDING_MESSAGES=4096
```

The future app argv additionally has these reserved names for B3:

```text
--capture-socket <absolute path>
--capture-source-id <source id>
--capture-max-body-prefix-bytes <bytes>
--capture-max-in-memory-bytes <bytes>
--capture-max-pending-messages <count>
```

Stock mitmdump receives the shared values through the environment because
project-specific flags cannot be added to its parser before the `-s` addon is
loaded. B2's addon can consume the exact environment names and remain
loadable through the ordinary `-s <absolute addon path>` argument.

## Planning before B3

The safe surface available before the app server exists is:

```sh
mitm-inspector plan \
  --reverse-upstream https://api.anthropic.com \
  --app-port 8000
```

It prints the eventual argv vectors, IPC contract, preflight report, and
lifecycle order. The equivalent dry-run is `mitm-inspector run --dry-run`;
neither command starts a proxy, app server, listener, browser, or shell. A
live `run` intentionally exits with an explanation until B3 provides the app
server.

The planned commands are equivalent to:

```text
<sys.executable> -m mitm_inspector.api.server --host 127.0.0.1 --port 8000 --proxy-port 8080 ...
<sibling>/mitmdump --mode reverse:https://api.anthropic.com --listen-host 127.0.0.1 --listen-port 8080 -s <absolute addon path>
```

The app child is started and made ready before the proxy child. On shutdown,
the proxy is terminated, fully waited/killed and group-verified before the app
is touched. Cleanup always attempts both children and aggregates exceptions.
A surviving process group after the kill deadline leaves the supervisor in
`FAILED` and raises `CleanupError`; it is never reported as successful
`STOPPED` cleanup.

SIGINT and SIGTERM handlers are installed before either child starts and are
removed only after cleanup. They map to exit statuses 130 and 143. A
`KeyboardInterrupt` maps to 130. Running the supervisor off the main thread
requires the explicit `SignalPolicy.DISABLED_FOR_TEST` injection; otherwise it
fails before startup.

Readiness is an injected seam. S0's default probe only detects an immediate
child exit; B3 can provide a health/readiness probe without changing the
supervisor or its tests. Process creation is also injected, and production
creation always receives a `ProcessSpec` with direct argv and `shell=False`.

The browser is convenience-only. It is disabled by default; a false return or
exception from an explicitly enabled opener is nonfatal and sent to the
injected warning sink (the default emits a Python runtime warning).

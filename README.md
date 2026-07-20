# tooling

Personal tooling monorepo.

## Subprojects

- **`mitm-inspector/`** — live proxy traffic inspector for Anthropic API (Python + React/TS)
- **`pufferclone/`** — pufferpanel clone (Python)
- **`tix/`** — CLI ticket tracker with atomic durable storage (Python)
- **`gauge/`** — time-series DB v0 with terminal charts (Rust)

Each subproject retains its own build, tests, and history (imported via `git subtree`). Development happens per-subproject; PRs open against this repo with paths scoped to the relevant subproject.

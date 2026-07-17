# S0 validation record

Validation was run from the repository root with project-local environments.
No global package or Python environment was modified.

## Tool versions

```text
Python 3.12.13
uv 0.11.13
mitmproxy 12.2.3
ruff 0.12.12
mypy 1.20.2
jsonschema 4.26.0
pytest 8.4.2
Node v26.0.0
npm 11.12.1
Vite 7.3.6
Vitest 3.2.7
Ajv 8.20.0
Git 2.54.0
```

## Commands and results

```sh
uv sync --python /opt/homebrew/bin/python3.12       # pass; local .venv
uv run ruff check .                                 # pass
uv run mypy                                          # pass, 11 source files
uv run pytest                                        # pass, 17 tests

npm install --prefix web                             # pass; generated lockfile
npm ci --prefix web                                  # pass; 221 packages
npm run lint --prefix web                            # pass
npm run typecheck --prefix web                       # pass; app + node projects explicitly checked
npm test --prefix web -- --run                       # pass, 7 tests
npm run build --prefix web                           # pass; Vite production bundle

python3 -m json.tool contracts/protocol-v1.schema.json >/dev/null  # pass
```

The Python and TypeScript suites both parse shared positive and negative
fixtures in `contracts/fixtures/`. They cover ordered duplicate headers,
missing/empty/truncated body states and decoded byte counts,
response-before-request-end ordering, bounded decimal-string values including
u64 overflow, unknown additive fields/types, redaction canaries, gap/body
arithmetic invariants, and private-mitmweb import bypasses.

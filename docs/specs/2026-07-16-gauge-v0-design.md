# gauge v0 — Design

Prometheus-lite metrics TSDB for the local-cloud stack. Motivation (Henry 2026-07-16, "daily utility first"): 5+ long-lived local services (wiki backend, supervisor, fleet, pufferclone, future GPU inference) with zero observability; fleet monitoring today is hand-rolled shell polling; a bench review round was wasted on host contention that CPU metrics would have answered in seconds. CLI-first: no web UI.

Out of scope for v0: alerting, remote write/read, downsampling, full PromQL, clustering, auth (localhost bind only until the tailnet-daemonization arc).

## Shape

- Rust workspace, repo `~/me/fun/misc/gauge`, ticket prefix GAU. Binaries: `gauge-server`, `gauge` (CLI), `gauge-node`, `gauge-proc`.
- Data model: `metric_name{label="value",...} f64 @ millis`. Series identity = name + sorted label set.
- Compatibility: scrape targets speak the Prometheus text exposition format (`# TYPE`, `name{labels} value`) — counters and gauges v0; histograms parsed but stored as their component series (`_bucket`/`_sum`/`_count`).

## GAU-1: storage engine (crate `gauge-store`)

The learning core. No HTTP, no scraping — a library with write/read handles, exercised by tests.

- **Head block**: in-RAM per-series append buffers for the active 2h window. Out-of-order samples within a small tolerance (60s) accepted; older rejected with a counted error.
- **WAL**: append-only, fsynced batches, replayed on open; truncated after successful chunk flush. Crash mid-flush must not lose acked samples or double-count (pufferclone WAL lessons apply).
- **Chunk flush**: every 2h boundary (and on shutdown), head block encodes to immutable chunk files under `data/<partition-start-ts>/`: Gorilla-style encoding — delta-of-delta timestamps, XOR float values. Target ≤2 bytes/sample on regular-interval data (assert a compression-ratio bound in tests with realistic fixtures).
- **Label index**: per partition, inverted index label→series-ids + series-id→chunk offsets; serialized alongside chunks. Series lookup by exact matchers v0 (`{job="x", instance=~"..."}`: exact and regex matchers).
- **Retention**: on flush, delete partitions older than configured horizon (default 30d). Deletion is partition-level only.
- **Read API**: `select(matchers, t_min, t_max) -> iterator of (series, samples)` merging head block + chunks.
- Tests: encode/decode round-trip (property-style over random walks + regular intervals), crash-replay (kill between WAL write and flush), compression bound, out-of-order handling, retention deletion, concurrent write+read.

## GAU-5: exporters (crate `gauge-exporters`, parallel with GAU-1)

Standalone bins, no dependency on gauge-store — they only expose `/metrics` text format over HTTP.

- **gauge-node**: host stats via `sysinfo` — CPU (per-core + total), memory, swap, disk usage + IO, load averages, network bytes. Named `node_*` following node-exporter conventions where sensible.
- **gauge-proc**: config lists process name patterns; exports per-match CPU%, RSS, thread count, fd count as `proc_*{name=...}`. Answers "what are 8 codex workers costing me".
- Tests: exposition output parses with a strict parser, values sane (cpu 0-100 per core), pattern matching, endpoint concurrency.

## GAU-2: scraper (in `gauge-server`, wave 2)

- TOML config: targets (`name`, `url`, `interval`, default 15s), staleness (series absent 2 scrapes → staleness marker), scrape timeout, per-target `up{target=...}` synthetic metric.
- Exposition parser: strict on structure, tolerant on unknown types; malformed lines counted + skipped, never abort the scrape.
- Writes through gauge-store's write handle. Scrape loop must not block on flushes.
- Tests: parser corpus (valid, malformed, histogram), scrape loop against a fixture server (up/down transitions produce `up` 1/0), staleness, interval jitter bounds.

## GAU-3: query engine + HTTP API (in `gauge-server`, wave 2, disjoint modules from GAU-2)

- Expression grammar (subset, NOT full PromQL): selector `name{matchers}`, `rate(selector[5m])` (counter-reset aware), aggregations `sum|avg|min|max|count (expr) by (label,...)`, scalar arithmetic `expr * 100`. No joins, no offset, no subqueries.
- Eval: instant query (at timestamp) and range query (start/end/step) over gauge-store `select`.
- HTTP API: `GET /api/query?expr=&time=`, `GET /api/query_range?expr=&start=&end=&step=`, `GET /api/series?match=`, `GET /api/targets`, JSON responses. Localhost bind default.
- Tests: grammar corpus incl. precedence + error cases, rate over counter resets, aggregation correctness against hand-computed fixtures, range-step alignment, HTTP round-trips.

## GAU-4: CLI (crate `gauge-cli`, wave 3, after GAU-3)

- `gauge query 'expr'` (instant), `gauge range 'expr' --last 1h --step 15s` (table), `gauge graph 'expr' --last 1h` (braille/block terminal chart, multi-series with legend), `gauge series 'match'`, `gauge targets`. Global `--json`, `--url`/`GAUGE_URL`.
- Chart rendering deterministic given fixed input (goldens in tests).
- Errors: connection refused names URL; query errors surface server message.

## Delivery

Wave 1 (parallel, disjoint): GAU-1 (gauge-store crate), GAU-5 (exporters crate).
Wave 2 (parallel, disjoint modules within gauge-server, both consume GAU-1's API): GAU-2 scraper, GAU-3 query+API. Merge conflicts limited to Cargo.toml/server wiring — resolved at merge.
Wave 3: GAU-4 CLI (needs GAU-3's HTTP API).
Follow-ups in other repos (filed separately, not v0): wiki backend `/metrics` (worker counts/states, API latency), pufferclone `/metrics` (namespace bytes, budget, evictions, query latency).
Same review pipeline as pufferclone/tix: sol deep review pinned at SHA per round, orchestrator squash-merges on clean pass.

## Success criterion (v0 done)

`gauge-node` + `gauge-proc` running, `gauge-server` scraping both plus itself, and `gauge graph 'node_cpu_percent' --last 1h` renders tonight's fleet load in the terminal.

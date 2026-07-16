# Pufferclone v2 — Design

v1 (see 2026-07-15-pufferclone-v1-design.md) completed the engine: HNSW, compaction, S3Store, non-blocking cold loads with bounded admission and a lifecycle coordinator (main@87cad8d). v2 validates it against real object storage, measures it, and adds the memory-management layer that makes the cold/hot split real.

Out of scope for v2: NVMe segment cache tier (candidate PUF-12 once PUF-11 numbers exist), typed attribute schema, auth/API keys, multi-node, wiki semantic-search integration.

## PUF-9: MinIO end-to-end

S3Store passes the store conformance suite, but the full engine has never run against a real S3 API. Prove the stack end-to-end: WAL, segment publish, manifest swap, compaction input deletes, cold loads — all through MinIO.

- `docker-compose.yml` (repo root): single MinIO service, fixed ports (9000 API), fixed credentials, named volume; `make minio-up` / `make minio-down` targets (or plain docker compose commands documented in README section).
- Engine-level integration test file `tests/s3_e2e.rs`, gated behind `PUFFERCLONE_TEST_S3_URL` exactly like the store conformance gating — skipped (not failed) when env absent. Coverage: full lifecycle (upsert → flush → query), compaction against S3 (inputs deleted, orphan cleanup at reopen), cold-load fanout against S3 (one load observed via store get counts), engine reopen from empty cache reading only from MinIO.
- Each test uses a unique bucket or key prefix (test isolation; parallel test runs must not collide).
- Smoke script `scripts/s3_smoke.sh`: starts MinIO if down, creates bucket, runs the HTTP server with S3 env, drives upsert/query via curl, asserts results, cleans up. This is the "does the real binary work against real S3" check, distinct from cargo tests.
- Document any behavioral differences discovered vs LocalDirStore (list pagination, conditional-put error mapping, latency) in the PR body and as code comments only where behavior-bearing.
- Host env: docker via colima. Tests must remain green when docker/MinIO absent (skip path).

## PUF-10: benchmark suite

Zero perf data exists; v1's concurrency and index parameters (admission cap, M/ef, compaction triggers) were tuned by feel. Build the measurement harness.

- `benches/` with criterion. Benchmarks, each with seeded deterministic corpus (shared generator in a `pufferclone::testkit` module or `benches/common.rs` — no `Date::now`/entropy in corpus construction):
  - vector query: HNSW vs exact scan at 1k/10k/100k docs, dims 128/768, top-k 10 — latency; plus a non-criterion recall harness (below).
  - BM25 query latency vs corpus size (10k/100k short docs).
  - upsert→flush throughput (docs/sec through WAL + segment build).
  - compaction throughput (merge N segments of M docs, docs/sec).
  - cold-load time vs namespace size (segment count × size), LocalDirStore.
  - admission fanout: X concurrent cold namespaces at admission cap C — time-to-all-loaded and max observed in-flight (reuses the high-water instrumentation from v1 tests).
- Recall harness: `cargo run --release --bin bench_recall` — prints recall@10/@100 for HNSW vs exact scan across ef_search values on a seeded corpus; table to stdout. Not criterion (recall is a correctness-quality metric, not latency).
- `BENCHMARKS.md` at repo root: how to run, plus one committed baseline table from this machine (date-stamped, hardware-noted). Baselines are informational, not CI-enforced.
- Benches must not interfere with `cargo test` (no test-cfg leakage); `cargo bench --no-run` compiles clean in the gate.

## PUF-11: memory budget + eviction policy (serial, after PUF-10)

Eviction mechanics exist (lifecycle coordinator handles evict/load races, drains, cancellation); nothing decides WHEN to evict. Add accounting and policy.

- Memory accounting: per-loaded-namespace estimate — sum of segment doc bytes + index section sizes + memtable estimate; recomputed at load/flush/compaction swap. Exactness not required; consistency is (same namespace state → same number).
- Global budget: `PUFFERCLONE_MEMORY_BUDGET_BYTES` env (default: unlimited = current behavior; 0 invalid). Engine tracks total loaded bytes.
- Policy: LRU by last-query-or-write timestamp. On load completing (or flush growing a namespace) while over budget: evict least-recently-used loaded namespaces (never the one just touched) until under budget or nothing evictable. Never evict namespaces with in-flight operations — coordinator's existing drain rules decide evictability; policy only picks candidates.
- A single namespace over the whole budget still loads (budget is soft floor of one) — document.
- Eviction from policy goes through the same coordinator path as explicit eviction — no second eviction mechanism.
- Tests: deterministic accounting (fixed corpus → expected estimate range), LRU order correctness, budget-triggered evict then reload works, in-flight-load protected from policy evict (barrier-gated), unlimited default preserves current behavior, budget + admission stress (loads under both semaphore cap and budget churn).

## Delivery

Wave 1 (parallel, disjoint): PUF-9 (docker/tests/scripts, no engine changes), PUF-10 (benches/, testkit, bin — no engine behavior changes; instrumentation hooks read-only).
Wave 2 (serial, touches engine/lifecycle): PUF-11 after PUF-10 merges (policy tuning wants bench numbers; accounting hooks may touch the same files PUF-10 instruments).
Same review pipeline as v0/v1: sol deep review per round pinned at SHA, orchestrator merges locally on clean pass.

# Pufferclone v1 — Design

v0 (see 2026-07-15-pufferclone-v0-design.md) delivered the full architecture with exact-scan search. v1 adds the performance-bearing pieces while keeping the learning goal: ANN indexing, segment compaction, a real object-storage backend, and a non-blocking read path.

Out of scope for v1: wiki semantic-search integration (blocked on embedding-source decision), multi-node, auth, quantization.

## PUF-5: HNSW vector index

Hand-rolled HNSW in `src/index/hnsw.rs` implementing the existing `VectorIndex` trait (search + search_filtered + serialization).

- Parameters: M=16, M0=32, ef_construction=200; ef_search per-query (default 64, exposed as optional `ef_search` in the query API).
- Cosine distance, same zero-vector/dim rules as ExactScan.
- Deterministic construction: level assignment seeded from a stable hash of the doc id (reproducible builds → stable segment bytes and checksums).
- Serialization: layers (adjacency lists), entry point, vectors; self-contained to_bytes/from_bytes like other indexes.
- Integration: segment flush builds an HNSW section ("hnsw") when live doc count ≥ 256; below that the segment relies on exact scan over docs.bin (avoids graph overhead on tiny segments). Query path: per segment, use HNSW section if present, else exact scan; memtable always exact scan. search_filtered semantics: filter applied during graph traversal (skip disallowed, keep exploring), not post-truncation.
- Tests: recall@10 ≥ 0.9 vs ExactScan on ≥2k random vectors (fixed RNG seed passed in, no Date/entropy in tests), serialize→deserialize→search equivalence, filtered search correctness when filter excludes most neighbors, deterministic rebuild produces identical bytes.

## PUF-6: segment compaction

Background merge of small/stale segments in `src/segment.rs` + `src/namespace.rs`.

- Trigger (post-flush check): ≥4 live segments, or any segment where >50% of docs are superseded/tombstoned relative to newer state.
- Merge: newest-wins dedup across input segments, drop tombstoned docs physically, rebuild index sections (text, hnsw when ≥256 docs), write as a new segment, manifest swap atomically replaces inputs with output, then delete input segment objects (derive-from-manifest cleanup like WAL retirement — orphans from failed deletes cleaned at open).
- Tombstone lifetime: a delete's tombstone can only be dropped when no older segment still contains the id — compaction of the full covering set. Until then tombstones ride along in segments (persist memtable tombstones into segments as a small "tombstones" section at flush if not already).
- Concurrency: compaction is the namespace worker's job, serialized with flush (same task); queries keep serving old manifest view until swap.
- Tests: merge correctness (newest wins, deletes gone), checksum-valid output, query equivalence before/after compaction, failed-delete orphan cleanup at reopen, compaction+concurrent-query stress.

## PUF-7: S3-compatible ObjectStore

Second `ObjectStore` impl in `src/store.rs` (or `src/store_s3.rs`): `S3Store` via the `object_store` crate (S3 provider).

- Write-once enforced with conditional put (PutMode::Create → AlreadyExists on conflict); manifest replace uses plain put. This is the real-world version of the hard-link trick — document the parallel.
- No dir-fsync analog needed (object stores are their own durability domain) — document why the LocalDirStore fsync ceremony disappears.
- Config: engine reads PUFFERCLONE_S3_URL / bucket / credentials env; main.rs picks S3Store when set, LocalDirStore otherwise.
- Tests: trait conformance suite extracted so both impls run the same tests; S3 tests gated behind `PUFFERCLONE_TEST_S3_URL` env (MinIO: `docker run -p 9000:9000 minio/minio server /data`) and skipped otherwise. CI-of-one rule: gate suite runs them when env present.

## PUF-8: non-blocking reads

- Cold open must not hold the engine registry lock: per-name once-cell/loading-slot pattern — concurrent queries for the same cold namespace await one load; queries for other namespaces unaffected.
- Query path takes read locks only; no lock held across store I/O awaits where avoidable (load segment bytes outside the namespace write lock, swap in under short write section).
- LRU eviction and delete must respect in-flight loads (guard from v0 extends to the loading slot).
- Tests: concurrent cold queries (N tasks, one load observed — count store gets), no-deadlock stress mixing query/upsert/flush/evict/delete, latency sanity (loaded-namespace query not serialized behind another namespace's cold load — assert via ordering, not wall-clock).

## Delivery

Wave 1 (parallel): PUF-5 (src/index/hnsw.rs, vector trait wiring, flush/query integration), PUF-7 (store impl, disjoint).
Wave 2 (serial, both touch namespace/engine): PUF-6, then PUF-8.
Same review pipeline as v0: sol deep review per round pinned at SHA, orchestrator merges locally on clean pass.

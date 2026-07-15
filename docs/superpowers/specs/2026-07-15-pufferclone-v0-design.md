# Pufferclone v0 — Design

A single-node, learning-oriented clone of Turbopuffer's architecture in Rust: object-storage-backed vector + full-text search with namespaces, WAL-durable writes, background indexing, and a hot/cold namespace cache.

Goal: understand the architecture (storage-compute split, WAL-on-object-storage, manifest swaps, cold-start query path). Correctness over performance. Exact-scan vector search in v0; ANN behind a trait for later.

## Architecture

Single crate `pufferclone` (lib + bin). One process: axum HTTP server + storage engine in-process.

```
src/
  main.rs          — axum server bootstrap
  api.rs           — HTTP handlers, request/response types
  engine.rs        — Namespace registry, hot-cache LRU
  namespace.rs     — per-namespace state: manifest + memtable + loaded segments
  wal.rs           — WAL segment encode/decode
  segment.rs       — index segment build/read (vectors, inverted index, attributes)
  index/
    vector.rs      — VectorIndex trait + ExactScan impl
    text.rs        — inverted index + BM25
    filter.rs      — attribute filter eval
  store.rs         — ObjectStore trait + LocalDirStore impl
  types.rs         — Doc, AttrValue, Query, errors
```

## Data model

- Namespace = isolated index, created implicitly on first upsert.
- Doc: `{ id: String, vector: Option<Vec<f32>>, attributes: Map<String, AttrValue> }`.
- AttrValue: string | int | float | bool | list<string>.
- Full-text fields: string attributes marked `full_text_search: true` via per-namespace schema hint in upsert.
- Vector dim fixed per namespace, set by first upsert, enforced after.

## ObjectStore

Trait: `put(key, bytes)` (write-once, except manifest), `get(key)`, `list(prefix)`, `delete(key)`.
Impl: `LocalDirStore` — files under `data/`, atomic tmp+rename. Manifest is the one mutable key per namespace (mirrors tpuf manifest swap).

Key layout:

```
ns/<name>/wal/<seq:016>.wal
ns/<name>/segments/<segid>/{vectors.bin, text.bin, attrs.bin, meta.json}
ns/<name>/MANIFEST.json   — schema, dim, live segment list, last WAL seq
```

## Write path

1. `POST /v1/namespaces/:ns` with docs (and/or `deletes: [ids]`) → validate (dim, id, attr types).
2. Encode batch as WAL segment (bincode) → `put ns/<ns>/wal/<seq>.wal` → update manifest (`last_wal_seq`) → ack. Durable on object store before ack.
3. Apply to in-memory memtable (id → doc; upsert = replace; delete = tombstone).

## Background indexer

Per-namespace tokio task. Trigger: memtable > ~1k docs or WAL bytes > threshold.
Build segment: vector block + inverted index (lowercase-alnum tokenizer; term → postings + doc lengths for BM25) + attribute columns → put segment blobs → manifest swap (add segment, drop covered WAL seqs) → clear covered memtable range.
No segment merging in v0.

## Read path

1. `POST /v1/namespaces/:ns/query`. Cold namespace → fetch MANIFEST + segment metas + WAL tail, replay WAL into memtable, register in hot-cache LRU (cap N namespaces; evict = drop from RAM). First cold query pays fetch latency by design.
2. Plan: attribute filters → candidate bitmap; vector query → exact scan over segments + memtable (skip tombstoned/superseded ids); text query → BM25 over segments + memtable mini-index.
3. Hybrid (vector + text both present): RRF rank fusion only.
4. Top-k merge across segments + memtable → response with scores + selected attributes.

Consistency: reads see all acked writes (WAL replay + memtable).

## API

JSON over HTTP (axum):

- `POST /v1/namespaces/:ns` — upsert/delete batch
- `POST /v1/namespaces/:ns/query` — `{ vector?, text?, filters?, top_k, include_attributes? }`
- `GET /v1/namespaces` — list namespaces
- `DELETE /v1/namespaces/:ns` — drop namespace (delete prefix)

Errors: 400 validation (dim mismatch, bad filter), 404 unknown namespace on query, 500 store errors. Typed error enum → JSON `{"error": "..."}`.

## Testing

- Unit: WAL roundtrip, BM25 vs hand-computed scores, filter eval, manifest swap.
- Integration: upsert → query (memtable path); upsert → forced flush → query (segment path); cold start (new engine over same data dir → correct results); LRU eviction → re-query correct; hybrid RRF sanity.
- No load/perf tests in v0.

## Delivery plan (tickets)

- PUF-1: crate scaffold + types.rs + store.rs + wal.rs + manifest handling. Foundation; merges first.
- PUF-2: index/ modules — vector (trait + exact scan), text (inverted index + BM25), filter. Pure over types.
- PUF-3: segment.rs — segment build from memtable batch, segment read; uses index modules' serialized forms.
- PUF-4: engine.rs + namespace.rs + api.rs + main.rs + integration tests. Depends on all prior.

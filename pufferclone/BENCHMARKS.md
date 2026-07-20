# Benchmark suite

The benchmark target uses Criterion and deterministic fixtures from
`pufferclone::testkit`. Corpus construction uses fixed seeds and does not read
the clock or operating-system entropy.

Run the default suite with:

```sh
cargo bench --bench benchmarks
```

Run the recall-quality harness separately:

```sh
cargo run --release --bin bench_recall
```

The default vector suite measures 1,000, 10,000, and 100,000 documents at 128
and 768 dimensions. The 100,000-document HNSW fixtures are intentionally
expensive to construct; Criterion uses 10 samples and a one-second measurement
window for this group. Targeted non-vector runs can skip fixture construction,
and a large-only vector run is available for refreshing just the required
large rows:

```sh
PUFFERCLONE_SKIP_VECTOR=1 cargo bench --bench benchmarks -- admission_fanout
PUFFERCLONE_VECTOR_ONLY_LARGE=1 cargo bench --bench benchmarks -- vector_query
```

The default Criterion configuration uses 10 samples and a one-second target
measurement time for vector benchmarks, 10 samples and a two-second target for
lifecycle and write benchmarks, and 20 samples with a two-second target for
BM25. Upsert/flush and compaction use Criterion's small-input batch setup;
the timed operations process 1,000 documents and four 512-document segments,
respectively. `cargo bench --no-run` is the compile-only gate.

## Baseline — 2026-07-16

Measured on Apple Silicon (`arm64`, Mac17,9, 15 CPU cores, 48 GiB RAM), with
the working tree at the PUF-10 implementation before commit. Criterion values
are median point estimates from the default suite; throughput is the
corresponding Criterion estimate.

| Benchmark | Median |
| --- | ---: |
| HNSW query, 1k × 128, top-k 10 | 134.91 µs |
| Exact query, 1k × 128, top-k 10 | 144.05 µs |
| HNSW query, 1k × 768, top-k 10 | 550.97 µs |
| Exact query, 1k × 768, top-k 10 | 547.68 µs |
| HNSW query, 10k × 128, top-k 10 | 237.12 µs |
| Exact query, 10k × 128, top-k 10 | 1.095 ms |
| HNSW query, 10k × 768, top-k 10 | 1.061 ms |
| Exact query, 10k × 768, top-k 10 | 5.322 ms |
| HNSW query, 100k × 128, top-k 10 | 296.25 µs |
| Exact query, 100k × 128, top-k 10 | 11.522 ms |
| HNSW query, 100k × 768, top-k 10 | 2.146 ms |
| Exact query, 100k × 768, top-k 10 | 97.742 ms |
| BM25 query, 10k docs | 4.755 ms |
| BM25 query, 100k docs | 60.258 ms |
| Upsert → flush, 1k docs | 765 docs/s |
| Compaction, four 512-doc segments | 666 docs/s |
| Cold load, one 512-doc segment | 1.319 ms |
| Cold load, four 512-doc segments | 4.182 ms |
| Cold load, eight 512-doc segments | 9.341 ms |
| Admission fanout latency, eight namespaces, cap 4 | 7.503 ms |
| Admission high-water probe, eight namespaces, cap 4 | 7.855 ms; max admitted loads 4 |

The recall harness baseline on the same machine was:

| `ef_search` | recall@10 | recall@100 |
| ---: | ---: | ---: |
| 16 | 0.981 | 0.776 |
| 32 | 0.981 | 0.776 |
| 64 (default) | 0.981 | 0.776 |
| 128 | 0.984 | 0.822 |
| 256 | 0.994 | 0.946 |

Recall uses a seeded 10,000-document, 128-dimensional corpus and 32 seeded
queries. It is intentionally a separate correctness-quality harness rather
than a Criterion latency benchmark.

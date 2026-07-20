use std::collections::BTreeMap;
use std::sync::{Arc, Condvar, Mutex as StdMutex};
use std::time::{Duration, Instant};

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion, Throughput};
use futures_util::future::join_all;
use pufferclone::index::hnsw::Hnsw;
use pufferclone::index::text::TextIndex;
use pufferclone::index::vector::ExactScan;
use pufferclone::namespace::Query;
use pufferclone::segment::SegmentBuilder;
use pufferclone::store::{LocalDirStore, ObjectStore};
use pufferclone::testkit::{query_near, text_corpus, vector_corpus};
use pufferclone::{Engine, Error, Manifest, Result};
use tempfile::{tempdir, TempDir};
use tokio::runtime::{Builder, Runtime};

const TOP_K: usize = 10;
const SEGMENT_DOCS: usize = 512;

fn runtime() -> Runtime {
    Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .expect("benchmark runtime")
}

fn vector_query(vector: Vec<f32>) -> Query {
    Query {
        vector: Some(vector),
        text: None,
        filter: None,
        top_k: TOP_K,
        include_attributes: false,
    }
}

fn vector_query_bench(c: &mut Criterion) {
    if std::env::var_os("PUFFERCLONE_SKIP_VECTOR").is_some() {
        return;
    }
    let mut group = c.benchmark_group("vector_query");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(1));
    let cases = if std::env::var_os("PUFFERCLONE_VECTOR_ONLY_LARGE").is_some() {
        vec![(100_000, 128), (100_000, 768)]
    } else {
        vec![
            (1_000, 128),
            (1_000, 768),
            (10_000, 128),
            (10_000, 768),
            (100_000, 128),
            (100_000, 768),
        ]
    };
    for (doc_count, dimension) in cases {
        let corpus = vector_corpus(doc_count, dimension);
        let vectors = corpus.iter().filter_map(|doc| {
            doc.vector
                .as_ref()
                .map(|vector| (doc.id.clone(), vector.clone()))
        });
        let exact = ExactScan::build(vectors.clone());
        let hnsw = Hnsw::build(vectors);
        let query = query_near(&corpus, doc_count / 2);
        let label = format!("{doc_count}_docs_{dimension}_dims");
        group.bench_function(format!("hnsw/{label}"), |b| {
            b.iter(|| black_box(hnsw.search_with_ef(black_box(&query), TOP_K, 64)))
        });
        group.bench_function(format!("exact/{label}"), |b| {
            b.iter(|| black_box(exact.search(black_box(&query), TOP_K)))
        });
    }
    group.finish();
}

fn bm25_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("bm25_query");
    group
        .sample_size(20)
        .measurement_time(Duration::from_secs(2));
    for &doc_count in &[10_000, 100_000] {
        let index = TextIndex::build(text_corpus(doc_count));
        group.bench_function(format!("{doc_count}_docs"), |b| {
            b.iter(|| black_box(index.search(black_box("rust engine vector search"), TOP_K)))
        });
    }
    group.finish();
}

fn upsert_flush_bench(c: &mut Criterion) {
    let runtime = runtime();
    let docs = vector_corpus(1_000, 128);
    let mut group = c.benchmark_group("upsert_flush");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(2));
    group.throughput(Throughput::Elements(docs.len() as u64));
    group.bench_function("1k_docs", |b| {
        b.iter_batched(
            || {
                let root = tempdir().expect("benchmark tempdir");
                let engine = Engine::new(root.path()).expect("benchmark engine");
                (root, engine)
            },
            |(_root, engine)| {
                runtime.block_on(async {
                    engine
                        .upsert("bench", docs.clone(), Vec::new(), BTreeMap::new())
                        .await
                        .expect("upsert");
                    engine.force_flush("bench").await.expect("flush");
                });
            },
            BatchSize::SmallInput,
        )
    });
    group.finish();
}

fn prepare_compaction_case(runtime: &Runtime) -> (TempDir, Engine, Vec<pufferclone::Doc>) {
    let root = tempdir().expect("benchmark tempdir");
    let engine = Engine::new(root.path()).expect("benchmark engine");
    let segments = (0..3)
        .map(|segment| {
            vector_corpus(SEGMENT_DOCS, 128)
                .into_iter()
                .map(move |mut doc| {
                    doc.id = format!("segment-{segment}-{}", doc.id);
                    doc
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    runtime.block_on(async {
        for docs in &segments {
            engine
                .upsert("bench", docs.clone(), Vec::new(), BTreeMap::new())
                .await
                .expect("seed compaction segment");
            engine.force_flush("bench").await.expect("seed flush");
        }
    });
    let final_docs = vector_corpus(SEGMENT_DOCS, 128)
        .into_iter()
        .map(|mut doc| {
            doc.id = format!("final-{}", doc.id);
            doc
        })
        .collect();
    (root, engine, final_docs)
}

fn compaction_bench(c: &mut Criterion) {
    let runtime = runtime();
    let mut group = c.benchmark_group("compaction");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(2));
    group.throughput(Throughput::Elements((SEGMENT_DOCS * 4) as u64));
    group.bench_function("4_segments_x_512_docs", |b| {
        b.iter_batched(
            || prepare_compaction_case(&runtime),
            |(_root, engine, docs)| {
                runtime.block_on(async {
                    engine
                        .upsert("bench", docs, Vec::new(), BTreeMap::new())
                        .await
                        .expect("compaction upsert");
                    engine.force_flush("bench").await.expect("compact");
                });
            },
            BatchSize::SmallInput,
        )
    });
    group.finish();
}

fn seed_namespace(
    root: &std::path::Path,
    namespace: &str,
    segment_count: usize,
    docs_per_segment: usize,
) -> Result<()> {
    let store = LocalDirStore::new(root)?;
    let mut segments = Vec::with_capacity(segment_count);
    for segment in 0..segment_count {
        let docs = vector_corpus(docs_per_segment, 128)
            .into_iter()
            .map(|mut doc| {
                doc.id = format!("{namespace}-{segment}-{}", doc.id);
                doc
            })
            .collect::<Vec<_>>();
        let vectors = ExactScan::build(docs.iter().filter_map(|doc| {
            doc.vector
                .as_ref()
                .map(|vector| (doc.id.clone(), vector.clone()))
        }));
        let text = TextIndex::build(std::iter::empty::<(String, String)>());
        let hnsw = Hnsw::build(docs.iter().filter_map(|doc| {
            doc.vector
                .as_ref()
                .map(|vector| (doc.id.clone(), vector.clone()))
        }));
        let sections = vec![
            ("vectors", vectors.to_bytes()?),
            ("text", text.to_bytes()?),
            (
                "tombstones",
                bincode::serialize(&BTreeMap::<String, u64>::new())?,
            ),
            ("hnsw", hnsw.to_bytes()?),
        ];
        let meta = SegmentBuilder::new(&store, namespace, (segment as u64 + 1, segment as u64 + 1))
            .with_id(format!("segment-{segment}"))
            .build(docs, sections)
            .map_err(|error| Error::Store(error.to_string()))?;
        segments.push(meta);
    }
    Manifest {
        vector_dim: Some(128),
        full_text_fields: Vec::new(),
        segments,
        last_wal_seq: segment_count as u64,
    }
    .store(&store, namespace)
}

fn cold_load_bench(c: &mut Criterion) {
    let runtime = runtime();
    let mut group = c.benchmark_group("cold_load");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(2));
    for &(segment_count, docs_per_segment) in &[(1, 512), (4, 512), (8, 512)] {
        let root = tempdir().expect("cold-load tempdir");
        seed_namespace(root.path(), "cold", segment_count, docs_per_segment)
            .expect("seed cold namespace");
        let query = vector_query(vec![0.25; 128]);
        group.bench_function(
            format!("{segment_count}_segments_x_{docs_per_segment}_docs"),
            |b| {
                b.iter(|| {
                    let engine = Engine::new(root.path()).expect("cold engine");
                    runtime
                        .block_on(engine.query("cold", black_box(&query)))
                        .expect("cold query")
                })
            },
        );
    }
    group.finish();
}

struct AdmissionGate {
    target: usize,
    state: StdMutex<AdmissionGateState>,
    wake: Condvar,
}

#[derive(Default)]
struct AdmissionGateState {
    started: usize,
    active: usize,
    max_active: usize,
    released: bool,
}

impl AdmissionGate {
    fn new(target: usize) -> Self {
        Self {
            target,
            state: StdMutex::new(AdmissionGateState::default()),
            wake: Condvar::new(),
        }
    }

    /// Hold the gate while the namespace load's admission permit is held.
    /// The gate is entered by the first manifest read, before Namespace::open
    /// does any further store work, and released only after the target wave is
    /// observed. This makes the high-water probe deterministic and independent
    /// of the duration of individual object reads.
    fn enter(&self) {
        let mut state = self.state.lock().expect("admission gate lock");
        state.started += 1;
        state.active += 1;
        state.max_active = state.max_active.max(state.active);
        self.wake.notify_all();
        while !state.released {
            state = self.wake.wait(state).expect("admission gate wait");
        }
        state.active -= 1;
    }

    fn wait_for_target(&self) {
        let mut state = self.state.lock().expect("admission gate lock");
        while state.started < self.target {
            state = self.wake.wait(state).expect("admission target wait");
        }
    }

    fn release(&self) {
        let mut state = self.state.lock().expect("admission gate lock");
        state.released = true;
        self.wake.notify_all();
    }

    fn max_active(&self) -> usize {
        self.state.lock().expect("admission gate lock").max_active
    }
}

struct AdmissionProbeStore {
    inner: LocalDirStore,
    gate: StdMutex<Option<Arc<AdmissionGate>>>,
}

impl AdmissionProbeStore {
    fn new(root: &std::path::Path) -> Result<Self> {
        Ok(Self {
            inner: LocalDirStore::new(root)?,
            gate: StdMutex::new(None),
        })
    }

    fn install_gate(&self, gate: Arc<AdmissionGate>) {
        *self.gate.lock().expect("admission probe gate lock") = Some(gate);
    }

    fn clear_gate(&self) {
        *self.gate.lock().expect("admission probe gate lock") = None;
    }
}

impl ObjectStore for AdmissionProbeStore {
    fn put(&self, key: &str, bytes: &[u8]) -> Result<()> {
        self.inner.put(key, bytes)
    }

    fn get(&self, key: &str) -> Result<Vec<u8>> {
        let gate = self.gate.lock().expect("admission probe gate lock").clone();
        if key.ends_with("/MANIFEST.json") {
            if let Some(gate) = gate {
                gate.enter();
            }
        }
        self.inner.get(key)
    }

    fn list(&self, prefix: &str) -> Result<Vec<String>> {
        self.inner.list(prefix)
    }

    fn delete(&self, key: &str) -> Result<()> {
        self.inner.delete(key)
    }
}

fn admission_latency_bench(c: &mut Criterion) {
    let runtime = runtime();
    let namespace_count = 8;
    let root = tempdir().expect("admission tempdir");
    for index in 0..namespace_count {
        seed_namespace(root.path(), &format!("cold-{index}"), 1, SEGMENT_DOCS)
            .expect("seed admission namespace");
    }
    let store = Arc::new(LocalDirStore::new(root.path()).expect("admission store"));
    let names = (0..namespace_count)
        .map(|index| format!("cold-{index}"))
        .collect::<Vec<_>>();
    let query = vector_query(vec![0.25; 128]);
    let mut group = c.benchmark_group("admission_fanout");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(2));
    group.bench_function("8_namespaces_cap_4_latency", |b| {
        b.iter_custom(|iterations| {
            let mut elapsed = Duration::ZERO;
            for _ in 0..iterations {
                let engine = Engine::with_store(store.clone());
                let started = Instant::now();
                runtime.block_on(async {
                    let futures = names
                        .iter()
                        .map(|name| engine.query(name, &query))
                        .collect::<Vec<_>>();
                    for result in join_all(futures).await {
                        result.expect("admission query");
                    }
                });
                elapsed += started.elapsed();
            }
            elapsed
        })
    });
    group.finish();
}

fn admission_high_water_bench(c: &mut Criterion) {
    let runtime = runtime();
    let namespace_count = 8;
    let root = tempdir().expect("admission probe tempdir");
    for index in 0..namespace_count {
        seed_namespace(root.path(), &format!("cold-{index}"), 1, SEGMENT_DOCS)
            .expect("seed admission probe namespace");
    }
    let store = Arc::new(AdmissionProbeStore::new(root.path()).expect("admission probe store"));
    let names = (0..namespace_count)
        .map(|index| format!("cold-{index}"))
        .collect::<Vec<_>>();
    let query = vector_query(vec![0.25; 128]);
    let mut group = c.benchmark_group("admission_fanout");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(1));
    group.bench_function("8_namespaces_cap_4_high_water", |b| {
        b.iter_custom(|iterations| {
            let mut elapsed = Duration::ZERO;
            let mut max_active = 0;
            for _ in 0..iterations {
                let gate = Arc::new(AdmissionGate::new(Engine::cold_load_concurrency()));
                store.install_gate(gate.clone());
                let engine = Engine::with_store(store.clone());
                let names = names.clone();
                let query = query.clone();
                let started = Instant::now();
                runtime.block_on(async {
                    let join = tokio::spawn(async move {
                        let futures = names
                            .iter()
                            .map(|name| engine.query(name, &query))
                            .collect::<Vec<_>>();
                        join_all(futures).await
                    });
                    tokio::task::spawn_blocking({
                        let gate = gate.clone();
                        move || gate.wait_for_target()
                    })
                    .await
                    .expect("admission target waiter");
                    gate.release();
                    for result in join.await.expect("admission probe tasks") {
                        result.expect("admission probe query");
                    }
                });
                elapsed += started.elapsed();
                max_active = max_active.max(gate.max_active());
                store.clear_gate();
            }
            assert_eq!(
                max_active,
                Engine::cold_load_concurrency(),
                "admission high-water did not match the configured cap"
            );
            eprintln!("admission_fanout max_admitted_loads={max_active}");
            elapsed
        })
    });
    group.finish();
}

criterion_group!(
    benches,
    vector_query_bench,
    bm25_bench,
    upsert_flush_bench,
    compaction_bench,
    cold_load_bench,
    admission_latency_bench,
    admission_high_water_bench
);
criterion_main!(benches);

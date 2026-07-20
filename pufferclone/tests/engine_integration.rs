use std::collections::{BTreeMap, HashSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Barrier, Condvar, Mutex as StdMutex};
use std::task::Poll;
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use futures_util::future::poll_fn;
use pufferclone::api::router;
use pufferclone::engine::Engine;
use pufferclone::index::filter::Filter;
use pufferclone::namespace::Query;
use pufferclone::segment::SegmentReader;
use pufferclone::store::{LocalDirStore, ObjectStore};
use pufferclone::{AttrValue, Doc, Error, Manifest, Result};
use tempfile::TempDir;
use tower::ServiceExt;

fn doc(id: &str, vector: &[f32], text: &str, category: &str) -> Doc {
    Doc {
        id: id.to_owned(),
        vector: Some(vector.to_vec()),
        attributes: BTreeMap::from([
            ("text".to_owned(), AttrValue::String(text.to_owned())),
            (
                "category".to_owned(),
                AttrValue::String(category.to_owned()),
            ),
        ]),
    }
}

fn vector_query(vector: &[f32]) -> Query {
    Query {
        vector: Some(vector.to_vec()),
        text: None,
        filter: None,
        top_k: 10,
        include_attributes: false,
    }
}

fn text_query(text: &str) -> Query {
    Query {
        vector: None,
        text: Some(text.to_owned()),
        filter: None,
        top_k: 10,
        include_attributes: false,
    }
}

fn open_engine(dir: &TempDir) -> Engine {
    Engine::new(dir.path()).expect("engine")
}

async fn poll_once_until_pending<F>(future: &mut Pin<Box<F>>)
where
    F: Future,
{
    let pending = poll_fn(|cx| match future.as_mut().poll(cx) {
        Poll::Pending => Poll::Ready(true),
        Poll::Ready(_) => Poll::Ready(false),
    })
    .await;
    assert!(
        pending,
        "future completed before its transition could be cancelled"
    );
}

struct FailOnceStore {
    inner: LocalDirStore,
    fail_segment_meta: AtomicBool,
}

struct FailWalDeleteStore {
    inner: LocalDirStore,
    fail_once: AtomicBool,
}

struct FailSegmentDeleteStore {
    inner: LocalDirStore,
    fail_next: AtomicBool,
}

struct BlockingCompactionStore {
    inner: LocalDirStore,
    block_next: AtomicBool,
    started: StdMutex<Option<mpsc::Sender<()>>>,
    release: StdMutex<mpsc::Receiver<()>>,
}

struct ColdLoadStore {
    inner: LocalDirStore,
    get_count: AtomicUsize,
    block_next_get: AtomicBool,
    blocked_namespace: StdMutex<Option<String>>,
    started: StdMutex<Option<mpsc::Sender<()>>>,
    release: StdMutex<Option<mpsc::Receiver<()>>>,
    fail_next_get: AtomicBool,
    fail_next_get_as_io: AtomicBool,
    failed_namespace: StdMutex<Option<String>>,
    block_next_delete: AtomicBool,
    blocked_delete_namespace: StdMutex<Option<String>>,
    delete_started: StdMutex<Option<mpsc::Sender<()>>>,
    delete_release: StdMutex<Option<mpsc::Receiver<()>>>,
    all_get_gate: StdMutex<Option<Arc<AllGetGate>>>,
    active_gets: AtomicUsize,
    max_active_gets: AtomicUsize,
    compact_puts: AtomicUsize,
}

struct AllGetGate {
    released: StdMutex<bool>,
    wake: Condvar,
    started: mpsc::Sender<()>,
}

impl FailWalDeleteStore {
    fn new(dir: &TempDir) -> Self {
        Self {
            inner: LocalDirStore::new(dir.path()).expect("store"),
            fail_once: AtomicBool::new(true),
        }
    }
}

impl FailSegmentDeleteStore {
    fn new(dir: &TempDir) -> Self {
        Self {
            inner: LocalDirStore::new(dir.path()).expect("store"),
            fail_next: AtomicBool::new(false),
        }
    }

    fn fail_next_delete(&self) {
        self.fail_next.store(true, Ordering::Release);
    }
}

impl BlockingCompactionStore {
    fn new(dir: &TempDir, started: mpsc::Sender<()>, release: mpsc::Receiver<()>) -> Self {
        Self {
            inner: LocalDirStore::new(dir.path()).expect("store"),
            block_next: AtomicBool::new(false),
            started: StdMutex::new(Some(started)),
            release: StdMutex::new(release),
        }
    }
}

impl ColdLoadStore {
    fn new(dir: &TempDir) -> Self {
        Self {
            inner: LocalDirStore::new(dir.path()).expect("store"),
            get_count: AtomicUsize::new(0),
            block_next_get: AtomicBool::new(false),
            blocked_namespace: StdMutex::new(None),
            started: StdMutex::new(None),
            release: StdMutex::new(None),
            fail_next_get: AtomicBool::new(false),
            fail_next_get_as_io: AtomicBool::new(false),
            failed_namespace: StdMutex::new(None),
            block_next_delete: AtomicBool::new(false),
            blocked_delete_namespace: StdMutex::new(None),
            delete_started: StdMutex::new(None),
            delete_release: StdMutex::new(None),
            all_get_gate: StdMutex::new(None),
            active_gets: AtomicUsize::new(0),
            max_active_gets: AtomicUsize::new(0),
            compact_puts: AtomicUsize::new(0),
        }
    }

    fn reset_get_count(&self) {
        self.get_count.store(0, Ordering::Release);
    }

    fn get_count(&self) -> usize {
        self.get_count.load(Ordering::Acquire)
    }

    fn block_next_get_for(&self, namespace: &str) -> (mpsc::Receiver<()>, mpsc::Sender<()>) {
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        *self
            .blocked_namespace
            .lock()
            .expect("blocked namespace lock") = Some(namespace.to_owned());
        *self.started.lock().expect("started lock") = Some(started_tx);
        *self.release.lock().expect("release lock") = Some(release_rx);
        self.block_next_get.store(true, Ordering::Release);
        (started_rx, release_tx)
    }

    fn fail_next_get_for(&self, namespace: &str) {
        *self.failed_namespace.lock().expect("failed namespace lock") = Some(namespace.to_owned());
        self.fail_next_get.store(true, Ordering::Release);
    }

    fn fail_next_get_as_io_for(&self, namespace: &str) {
        *self.failed_namespace.lock().expect("failed namespace lock") = Some(namespace.to_owned());
        self.fail_next_get_as_io.store(true, Ordering::Release);
    }

    fn block_next_delete_for(&self, namespace: &str) -> (mpsc::Receiver<()>, mpsc::Sender<()>) {
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        *self
            .blocked_delete_namespace
            .lock()
            .expect("blocked delete namespace lock") = Some(namespace.to_owned());
        *self.delete_started.lock().expect("delete started lock") = Some(started_tx);
        *self.delete_release.lock().expect("delete release lock") = Some(release_rx);
        self.block_next_delete.store(true, Ordering::Release);
        (started_rx, release_tx)
    }

    fn block_all_gets(&self) -> mpsc::Receiver<()> {
        let (started_tx, started_rx) = mpsc::channel();
        *self.all_get_gate.lock().expect("all-get gate lock") = Some(Arc::new(AllGetGate {
            released: StdMutex::new(false),
            wake: Condvar::new(),
            started: started_tx,
        }));
        started_rx
    }

    fn release_all_gets(&self) {
        let gate = self
            .all_get_gate
            .lock()
            .expect("all-get gate lock")
            .clone()
            .expect("all-get gate");
        *gate.released.lock().expect("all-get release lock") = true;
        gate.wake.notify_all();
    }

    fn max_active_gets(&self) -> usize {
        self.max_active_gets.load(Ordering::Acquire)
    }

    fn active_gets(&self) -> usize {
        self.active_gets.load(Ordering::Acquire)
    }

    fn reset_compact_puts(&self) {
        self.compact_puts.store(0, Ordering::Release);
    }

    fn compact_puts(&self) -> usize {
        self.compact_puts.load(Ordering::Acquire)
    }
}

impl ObjectStore for FailWalDeleteStore {
    fn put(&self, key: &str, bytes: &[u8]) -> Result<()> {
        self.inner.put(key, bytes)
    }

    fn get(&self, key: &str) -> Result<Vec<u8>> {
        self.inner.get(key)
    }

    fn list(&self, prefix: &str) -> Result<Vec<String>> {
        self.inner.list(prefix)
    }

    fn delete(&self, key: &str) -> Result<()> {
        if key.ends_with(".wal") && self.fail_once.swap(false, Ordering::AcqRel) {
            return Err(Error::Store("injected WAL retirement failure".to_owned()));
        }
        self.inner.delete(key)
    }
}

impl ObjectStore for FailSegmentDeleteStore {
    fn put(&self, key: &str, bytes: &[u8]) -> Result<()> {
        self.inner.put(key, bytes)
    }

    fn get(&self, key: &str) -> Result<Vec<u8>> {
        self.inner.get(key)
    }

    fn list(&self, prefix: &str) -> Result<Vec<String>> {
        self.inner.list(prefix)
    }

    fn delete(&self, key: &str) -> Result<()> {
        if key.contains("/segments/") && self.fail_next.swap(false, Ordering::AcqRel) {
            return Err(Error::Store("injected segment deletion failure".to_owned()));
        }
        self.inner.delete(key)
    }
}

impl ObjectStore for BlockingCompactionStore {
    fn put(&self, key: &str, bytes: &[u8]) -> Result<()> {
        if key.contains("/segments/compact-")
            && key.ends_with("/docs.bin")
            && self.block_next.swap(false, Ordering::AcqRel)
        {
            self.started
                .lock()
                .expect("started lock")
                .take()
                .expect("started sender")
                .send(())
                .expect("started receiver");
            self.release
                .lock()
                .expect("release lock")
                .recv()
                .expect("release sender");
        }
        self.inner.put(key, bytes)
    }

    fn get(&self, key: &str) -> Result<Vec<u8>> {
        self.inner.get(key)
    }

    fn list(&self, prefix: &str) -> Result<Vec<String>> {
        self.inner.list(prefix)
    }

    fn delete(&self, key: &str) -> Result<()> {
        self.inner.delete(key)
    }
}

impl ObjectStore for ColdLoadStore {
    fn put(&self, key: &str, bytes: &[u8]) -> Result<()> {
        if key.contains("/segments/compact-") {
            self.compact_puts.fetch_add(1, Ordering::AcqRel);
        }
        self.inner.put(key, bytes)
    }

    fn get(&self, key: &str) -> Result<Vec<u8>> {
        self.get_count.fetch_add(1, Ordering::AcqRel);
        let active = self.active_gets.fetch_add(1, Ordering::AcqRel) + 1;
        self.max_active_gets.fetch_max(active, Ordering::AcqRel);
        let all_get_gate = self.all_get_gate.lock().expect("all-get gate lock").clone();
        if let Some(gate) = &all_get_gate {
            let _ = gate.started.send(());
            let mut released = gate.released.lock().expect("all-get release lock");
            while !*released {
                released = gate.wake.wait(released).expect("all-get wait");
            }
        }
        let blocked_namespace = self
            .blocked_namespace
            .lock()
            .expect("blocked namespace lock")
            .clone();
        if self.block_next_get.load(Ordering::Acquire)
            && blocked_namespace
                .as_deref()
                .is_some_and(|namespace| key.starts_with(&format!("ns/{namespace}/")))
            && self.block_next_get.swap(false, Ordering::AcqRel)
        {
            self.started
                .lock()
                .expect("started lock")
                .take()
                .expect("started sender")
                .send(())
                .expect("started receiver");
            self.release
                .lock()
                .expect("release lock")
                .take()
                .expect("release sender")
                .recv()
                .expect("release sender");
        }

        let failed_namespace = self
            .failed_namespace
            .lock()
            .expect("failed namespace lock")
            .clone();
        let result = if self.fail_next_get_as_io.load(Ordering::Acquire)
            && failed_namespace
                .as_deref()
                .is_some_and(|namespace| key.starts_with(&format!("ns/{namespace}/")))
            && self.fail_next_get_as_io.swap(false, Ordering::AcqRel)
        {
            Err(Error::Io(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "injected cold-load I/O failure",
            )))
        } else if self.fail_next_get.load(Ordering::Acquire)
            && failed_namespace
                .as_deref()
                .is_some_and(|namespace| key.starts_with(&format!("ns/{namespace}/")))
            && self.fail_next_get.swap(false, Ordering::AcqRel)
        {
            Err(Error::Store("injected cold-load failure".to_owned()))
        } else {
            self.inner.get(key)
        };
        self.active_gets.fetch_sub(1, Ordering::AcqRel);
        result
    }

    fn list(&self, prefix: &str) -> Result<Vec<String>> {
        self.inner.list(prefix)
    }

    fn delete(&self, key: &str) -> Result<()> {
        let blocked_namespace = self
            .blocked_delete_namespace
            .lock()
            .expect("blocked delete namespace lock")
            .clone();
        if self.block_next_delete.load(Ordering::Acquire)
            && blocked_namespace
                .as_deref()
                .is_some_and(|namespace| key.starts_with(&format!("ns/{namespace}/")))
            && self.block_next_delete.swap(false, Ordering::AcqRel)
        {
            self.delete_started
                .lock()
                .expect("delete started lock")
                .take()
                .expect("delete started sender")
                .send(())
                .expect("delete started receiver");
            self.delete_release
                .lock()
                .expect("delete release lock")
                .take()
                .expect("delete release sender")
                .recv()
                .expect("delete release sender");
        }
        self.inner.delete(key)
    }
}

struct BlockingStore {
    inner: LocalDirStore,
    block_once: AtomicBool,
    started: Arc<Barrier>,
    release: Arc<Barrier>,
}

impl BlockingStore {
    fn new(dir: &TempDir, started: Arc<Barrier>, release: Arc<Barrier>) -> Self {
        Self {
            inner: LocalDirStore::new(dir.path()).expect("store"),
            block_once: AtomicBool::new(true),
            started,
            release,
        }
    }
}

impl ObjectStore for BlockingStore {
    fn put(&self, key: &str, bytes: &[u8]) -> Result<()> {
        if key.ends_with(".wal") && self.block_once.swap(false, Ordering::AcqRel) {
            self.started.wait();
            self.release.wait();
        }
        self.inner.put(key, bytes)
    }

    fn get(&self, key: &str) -> Result<Vec<u8>> {
        self.inner.get(key)
    }

    fn list(&self, prefix: &str) -> Result<Vec<String>> {
        self.inner.list(prefix)
    }

    fn delete(&self, key: &str) -> Result<()> {
        self.inner.delete(key)
    }
}

impl FailOnceStore {
    fn new(dir: &TempDir) -> Self {
        Self {
            inner: LocalDirStore::new(dir.path()).expect("store"),
            fail_segment_meta: AtomicBool::new(true),
        }
    }
}

impl ObjectStore for FailOnceStore {
    fn put(&self, key: &str, bytes: &[u8]) -> Result<()> {
        if key.contains("/segments/")
            && key.ends_with("/meta.json")
            && self.fail_segment_meta.swap(false, Ordering::AcqRel)
        {
            return Err(Error::Store(
                "injected segment publication failure".to_owned(),
            ));
        }
        self.inner.put(key, bytes)
    }

    fn get(&self, key: &str) -> Result<Vec<u8>> {
        self.inner.get(key)
    }

    fn list(&self, prefix: &str) -> Result<Vec<String>> {
        self.inner.list(prefix)
    }

    fn delete(&self, key: &str) -> Result<()> {
        self.inner.delete(key)
    }
}

#[tokio::test]
async fn upsert_query_and_flush_segment_paths_are_consistent() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = open_engine(&dir);
    engine
        .upsert(
            "demo",
            vec![doc("a", &[1.0, 0.0], "rust", "code")],
            Vec::new(),
            BTreeMap::from([("text".to_owned(), true)]),
        )
        .await
        .expect("upsert");
    assert_eq!(
        engine
            .query("demo", &vector_query(&[1.0, 0.0]))
            .await
            .expect("query")[0]
            .id,
        "a"
    );
    engine.force_flush("demo").await.expect("flush");
    assert_eq!(
        engine
            .query("demo", &text_query("rust"))
            .await
            .expect("query")[0]
            .id,
        "a"
    );
}

#[tokio::test]
async fn hnsw_sections_are_thresholded_and_match_exact_results() {
    let large_dir = tempfile::tempdir().expect("tempdir");
    let large_engine = open_engine(&large_dir);
    let large_docs = (0..256)
        .map(|index| {
            let angle = index as f32 * 0.01;
            doc(
                &format!("doc-{index:03}"),
                &[angle.cos(), angle.sin()],
                "value",
                "code",
            )
        })
        .collect::<Vec<_>>();
    large_engine
        .upsert("large", large_docs, Vec::new(), BTreeMap::new())
        .await
        .expect("upsert");
    let expected = large_engine
        .query("large", &vector_query(&[1.0, 0.0]))
        .await
        .expect("memtable query");
    large_engine.force_flush("large").await.expect("flush");
    let actual = large_engine
        .query("large", &vector_query(&[1.0, 0.0]))
        .await
        .expect("hnsw query");
    assert_eq!(actual, expected);
    drop(large_engine);
    let large_engine = open_engine(&large_dir);
    assert_eq!(
        large_engine
            .query("large", &vector_query(&[1.0, 0.0]))
            .await
            .expect("cold hnsw query"),
        expected
    );

    let store = LocalDirStore::new(large_dir.path()).expect("store");
    let manifest = Manifest::load(&store, "large").expect("manifest");
    assert_eq!(manifest.segments.len(), 1);
    assert!(manifest.segments[0]
        .sections
        .iter()
        .any(|name| name == "hnsw"));
    let reader = SegmentReader::open(&store, "large", &manifest.segments[0].id).expect("segment");
    assert!(reader.section_bytes("hnsw").is_ok());

    let small_dir = tempfile::tempdir().expect("tempdir");
    let small_engine = open_engine(&small_dir);
    let small_docs = (0..255)
        .map(|index| {
            let angle = index as f32 * 0.01;
            doc(
                &format!("doc-{index:03}"),
                &[angle.cos(), angle.sin()],
                "value",
                "code",
            )
        })
        .collect::<Vec<_>>();
    small_engine
        .upsert("small", small_docs, Vec::new(), BTreeMap::new())
        .await
        .expect("upsert");
    let expected = small_engine
        .query("small", &vector_query(&[1.0, 0.0]))
        .await
        .expect("memtable query");
    small_engine.force_flush("small").await.expect("flush");
    assert_eq!(
        small_engine
            .query("small", &vector_query(&[1.0, 0.0]))
            .await
            .expect("exact segment query"),
        expected
    );
    let store = LocalDirStore::new(small_dir.path()).expect("store");
    let manifest = Manifest::load(&store, "small").expect("manifest");
    assert!(!manifest.segments[0]
        .sections
        .iter()
        .any(|name| name == "hnsw"));
}

#[tokio::test]
async fn namespace_hnsw_path_retains_recall_after_flush() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = open_engine(&dir);
    let mut state = 0x1234_5678_9abc_def0_u64;
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        ((state >> 32) as u32) as f32 / u32::MAX as f32 * 2.0 - 1.0
    };
    let documents = (0..256)
        .map(|index| {
            let vector = (0..16).map(|_| next()).collect::<Vec<_>>();
            doc(&format!("doc-{index:03}"), &vector, "value", "code")
        })
        .collect::<Vec<_>>();
    let query = (0..16)
        .map(|dimension| (dimension as f32 * 0.37).cos())
        .collect::<Vec<_>>();
    engine
        .upsert("recall", documents, Vec::new(), BTreeMap::new())
        .await
        .expect("upsert");
    let expected = engine
        .query("recall", &vector_query(&query))
        .await
        .expect("exact memtable query");
    engine.force_flush("recall").await.expect("flush");
    let actual = engine
        .query("recall", &vector_query(&query))
        .await
        .expect("hnsw query");
    let expected_ids = expected
        .iter()
        .map(|result| result.id.clone())
        .collect::<HashSet<_>>();
    let actual_ids = actual
        .iter()
        .map(|result| result.id.clone())
        .collect::<HashSet<_>>();
    let hits = expected_ids.intersection(&actual_ids).count();
    assert!(hits as f32 / expected_ids.len() as f32 >= 0.9);
}

#[tokio::test]
async fn compaction_merges_newest_wins_deletes_and_indexes() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = open_engine(&dir);
    engine
        .upsert(
            "compact",
            vec![
                doc("a", &[1.0, 0.0], "old shared", "keep"),
                doc("b", &[0.0, 1.0], "deleted shared", "drop"),
            ],
            Vec::new(),
            BTreeMap::from([("text".to_owned(), true)]),
        )
        .await
        .expect("first upsert");
    engine.force_flush("compact").await.expect("first flush");
    engine
        .upsert(
            "compact",
            vec![doc("a", &[0.0, 1.0], "new shared", "keep")],
            vec!["b".to_owned()],
            BTreeMap::new(),
        )
        .await
        .expect("second upsert");

    let before = engine
        .query(
            "compact",
            &Query {
                text: Some("new".to_owned()),
                filter: Some(Filter::Eq {
                    field: "category".to_owned(),
                    value: AttrValue::String("keep".to_owned()),
                }),
                top_k: 10,
                include_attributes: true,
                vector: None,
            },
        )
        .await
        .expect("query before compaction");
    engine
        .force_flush("compact")
        .await
        .expect("compacting flush");
    let after = engine
        .query(
            "compact",
            &Query {
                text: Some("new".to_owned()),
                filter: Some(Filter::Eq {
                    field: "category".to_owned(),
                    value: AttrValue::String("keep".to_owned()),
                }),
                top_k: 10,
                include_attributes: true,
                vector: None,
            },
        )
        .await
        .expect("query after compaction");
    assert_eq!(after, before);
    assert_eq!(after[0].id, "a");
    assert_eq!(
        after[0].attributes.as_ref().expect("attributes")["text"],
        AttrValue::String("new shared".to_owned())
    );
    let vector_results = engine
        .query("compact", &vector_query(&[1.0, 0.0]))
        .await
        .expect("deleted vector query");
    assert_eq!(vector_results.len(), 1);
    assert_eq!(vector_results[0].id, "a");

    let store = LocalDirStore::new(dir.path()).expect("store");
    let manifest = Manifest::load(&store, "compact").expect("manifest");
    assert_eq!(manifest.segments.len(), 1);
    let reader = SegmentReader::open(&store, "compact", &manifest.segments[0].id)
        .expect("compacted segment");
    assert_eq!(
        reader
            .documents()
            .map(|doc| doc.id.as_str())
            .collect::<Vec<_>>(),
        vec!["a"]
    );
    for section in ["vectors", "text", "tombstones"] {
        reader
            .section_bytes(section)
            .expect("valid section checksum");
    }
    assert!(reader.section_bytes("hnsw").is_err());
    drop(engine);
    let cold = open_engine(&dir);
    let cold_new = cold
        .query("compact", &text_query("new"))
        .await
        .expect("cold newest query");
    assert_eq!(cold_new.len(), 1);
    assert_eq!(cold_new[0].id, "a");
    assert!(cold
        .query("compact", &text_query("deleted"))
        .await
        .expect("cold deleted query")
        .is_empty());
}

#[tokio::test]
async fn four_segment_trigger_compacts_and_reopens_equivalently() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = open_engine(&dir);
    for batch in 0..3 {
        let documents = (0..64)
            .map(|offset| {
                let index = batch * 64 + offset;
                doc(
                    &format!("doc-{index:03}"),
                    &[1.0, 0.0],
                    "common term",
                    if index % 2 == 0 { "keep" } else { "drop" },
                )
            })
            .collect();
        engine
            .upsert(
                "four",
                documents,
                Vec::new(),
                BTreeMap::from([("text".to_owned(), true)]),
            )
            .await
            .expect("upsert");
        engine.force_flush("four").await.expect("flush");
    }

    engine
        .upsert(
            "four",
            (0..64)
                .map(|offset| {
                    let index = 192 + offset;
                    doc(
                        &format!("doc-{index:03}"),
                        &[1.0, 0.0],
                        "common term",
                        if index % 2 == 0 { "keep" } else { "drop" },
                    )
                })
                .collect(),
            Vec::new(),
            BTreeMap::new(),
        )
        .await
        .expect("fourth segment upsert");
    let before = engine
        .query(
            "four",
            &Query {
                vector: Some(vec![1.0, 0.0]),
                text: Some("common".to_owned()),
                filter: Some(Filter::Eq {
                    field: "category".to_owned(),
                    value: AttrValue::String("keep".to_owned()),
                }),
                top_k: 10,
                include_attributes: false,
            },
        )
        .await
        .expect("query before compaction");
    engine.force_flush("four").await.expect("compaction");
    let after = engine
        .query(
            "four",
            &Query {
                vector: Some(vec![1.0, 0.0]),
                text: Some("common".to_owned()),
                filter: Some(Filter::Eq {
                    field: "category".to_owned(),
                    value: AttrValue::String("keep".to_owned()),
                }),
                top_k: 10,
                include_attributes: false,
            },
        )
        .await
        .expect("query after compaction");
    assert_eq!(after, before);

    let store = LocalDirStore::new(dir.path()).expect("store");
    assert_eq!(
        Manifest::load(&store, "four")
            .expect("manifest")
            .segments
            .len(),
        1
    );
    drop(engine);
    let cold = open_engine(&dir);
    assert_eq!(
        cold.query("four", &text_query("common"))
            .await
            .expect("cold query")
            .len(),
        10
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn concurrent_queries_continue_during_compaction_build() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (started_tx, started_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let store = Arc::new(BlockingCompactionStore::new(&dir, started_tx, release_rx));
    let engine = Arc::new(Engine::with_store(Arc::clone(&store)));
    for index in 0..3 {
        engine
            .upsert(
                "concurrent",
                vec![doc(
                    &format!("doc-{index}"),
                    &[1.0, 0.0],
                    "concurrent merge",
                    "keep",
                )],
                Vec::new(),
                BTreeMap::from([("text".to_owned(), true)]),
            )
            .await
            .expect("upsert");
        engine.force_flush("concurrent").await.expect("flush");
    }
    engine
        .upsert(
            "concurrent",
            vec![doc("doc-3", &[1.0, 0.0], "concurrent merge", "keep")],
            Vec::new(),
            BTreeMap::new(),
        )
        .await
        .expect("fourth upsert");
    store.block_next.store(true, Ordering::Release);

    let flush_engine = Arc::clone(&engine);
    let flush = tokio::spawn(async move { flush_engine.force_flush("concurrent").await });
    tokio::task::spawn_blocking(move || started_rx.recv().expect("compaction started"))
        .await
        .expect("started task");

    let mut queries = Vec::new();
    for _ in 0..16 {
        let query_engine = Arc::clone(&engine);
        queries.push(tokio::spawn(async move {
            tokio::time::timeout(
                Duration::from_secs(1),
                query_engine.query("concurrent", &text_query("concurrent")),
            )
            .await
            .expect("query completed during compaction")
            .expect("query");
        }));
    }
    for query in queries {
        query.await.expect("query task");
    }
    release_tx.send(()).expect("release compaction");
    flush.await.expect("flush task").expect("compaction");
}

#[tokio::test]
async fn compaction_orphan_cleanup_repairs_failed_input_deletes() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = Arc::new(FailSegmentDeleteStore::new(&dir));
    let engine = Engine::with_store(Arc::clone(&store));
    for index in 0..3 {
        engine
            .upsert(
                "orphans",
                vec![doc(
                    &format!("doc-{index}"),
                    &[1.0, index as f32 + 1.0],
                    "orphan test",
                    "keep",
                )],
                Vec::new(),
                BTreeMap::from([("text".to_owned(), true)]),
            )
            .await
            .expect("upsert");
        engine.force_flush("orphans").await.expect("flush");
    }
    store.fail_next_delete();
    engine
        .upsert(
            "orphans",
            vec![doc("doc-3", &[1.0, 4.0], "orphan test", "keep")],
            Vec::new(),
            BTreeMap::new(),
        )
        .await
        .expect("fourth upsert");
    assert!(engine.force_flush("orphans").await.is_err());
    drop(engine);

    let reopened = Engine::with_store(Arc::clone(&store));
    let results = reopened
        .query("orphans", &text_query("orphan"))
        .await
        .expect("reopen query");
    assert_eq!(results.len(), 4);
    let manifest = Manifest::load(store.as_ref(), "orphans").expect("manifest");
    let keys = store.list("ns/orphans/segments/").expect("segments");
    assert!(keys.iter().all(|key| {
        manifest
            .segments
            .iter()
            .any(|segment| key.contains(&format!("/{}/", segment.id)))
    }));
}

#[tokio::test]
async fn mixed_vector_dimensions_and_invalid_schema_are_rejected_atomically() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = open_engine(&dir);
    let mixed = engine
        .upsert(
            "vectors",
            vec![
                doc("two", &[1.0, 0.0], "two", "code"),
                doc("three", &[1.0, 0.0, 0.0], "three", "code"),
            ],
            Vec::new(),
            BTreeMap::new(),
        )
        .await;
    assert!(matches!(mixed, Err(Error::Validation(_))));
    engine
        .upsert(
            "vectors",
            vec![doc("three", &[1.0, 0.0, 0.0], "three", "code")],
            Vec::new(),
            BTreeMap::new(),
        )
        .await
        .expect("valid retry");
    assert_eq!(
        engine
            .query(
                "vectors",
                &Query {
                    vector: Some(vec![1.0, 0.0, 0.0]),
                    text: None,
                    filter: None,
                    top_k: 1,
                    include_attributes: false,
                },
            )
            .await
            .expect("query")[0]
            .id,
        "three"
    );

    let invalid_schema = engine
        .upsert(
            "schema",
            vec![Doc {
                id: "bad".to_owned(),
                vector: None,
                attributes: BTreeMap::from([("title".to_owned(), AttrValue::Int(4))]),
            }],
            Vec::new(),
            BTreeMap::from([("title".to_owned(), true)]),
        )
        .await;
    assert!(matches!(invalid_schema, Err(Error::Validation(_))));
    engine
        .upsert(
            "schema",
            vec![Doc {
                id: "good".to_owned(),
                vector: None,
                attributes: BTreeMap::from([(
                    "title".to_owned(),
                    AttrValue::String("rust".to_owned()),
                )]),
            }],
            Vec::new(),
            BTreeMap::new(),
        )
        .await
        .expect("schemaless retry");
    assert!(engine
        .query("schema", &text_query("rust"))
        .await
        .expect("query")
        .is_empty());
}

#[tokio::test]
async fn rejected_first_write_does_not_publish_a_ghost_namespace() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = open_engine(&dir);
    let result = engine
        .upsert(
            "ghost",
            vec![
                doc("two", &[1.0, 0.0], "two", "code"),
                doc("three", &[1.0, 0.0, 0.0], "three", "code"),
            ],
            Vec::new(),
            BTreeMap::new(),
        )
        .await;
    assert!(matches!(result, Err(Error::Validation(_))));
    assert!(engine.list_namespaces().expect("namespaces").is_empty());
    assert!(!dir.path().join("ns/ghost").exists());
}

#[tokio::test]
async fn filtered_text_scores_use_the_unfiltered_live_corpus_stats() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = open_engine(&dir);
    engine
        .upsert(
            "stats",
            vec![
                doc("a", &[1.0, 0.0], "rust", "keep"),
                doc("b", &[0.0, 1.0], "rust", "drop"),
                doc("c", &[1.0, 1.0], "rust", "drop"),
            ],
            Vec::new(),
            BTreeMap::from([("text".to_owned(), true)]),
        )
        .await
        .expect("upsert");
    let unfiltered = engine
        .query("stats", &text_query("rust"))
        .await
        .expect("query");
    let filtered = engine
        .query(
            "stats",
            &Query {
                filter: Some(Filter::Eq {
                    field: "category".to_owned(),
                    value: AttrValue::String("keep".to_owned()),
                }),
                ..text_query("rust")
            },
        )
        .await
        .expect("query");
    assert_eq!(unfiltered[0].id, "a");
    assert_eq!(filtered[0].id, "a");
    assert_eq!(unfiltered[0].score, filtered[0].score);
}

#[tokio::test]
async fn cold_start_replays_an_unflushed_wal_tail() {
    let dir = tempfile::tempdir().expect("tempdir");
    {
        let engine = open_engine(&dir);
        engine
            .upsert(
                "tail",
                vec![doc("a", &[1.0, 0.0], "tail", "code")],
                Vec::new(),
                BTreeMap::new(),
            )
            .await
            .expect("upsert");
    }
    let engine = open_engine(&dir);
    assert_eq!(
        engine
            .query("tail", &vector_query(&[1.0, 0.0]))
            .await
            .expect("cold query")[0]
            .id,
        "a"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_cold_queries_share_one_store_load() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = Arc::new(ColdLoadStore::new(&dir));
    {
        let engine = Engine::with_store(Arc::clone(&store));
        engine
            .upsert(
                "cold",
                vec![doc("a", &[1.0, 0.0], "cold", "code")],
                Vec::new(),
                BTreeMap::new(),
            )
            .await
            .expect("upsert");
        engine.force_flush("cold").await.expect("flush");
    }

    store.reset_get_count();
    let baseline = Engine::with_store(Arc::clone(&store));
    baseline
        .query("cold", &vector_query(&[1.0, 0.0]))
        .await
        .expect("baseline cold query");
    let one_load_gets = store.get_count();
    drop(baseline);

    store.reset_get_count();
    let (started_rx, release_tx) = store.block_next_get_for("cold");
    let engine = Arc::new(Engine::with_store(Arc::clone(&store)));
    let first_engine = Arc::clone(&engine);
    let first =
        tokio::spawn(async move { first_engine.query("cold", &vector_query(&[1.0, 0.0])).await });
    tokio::task::spawn_blocking(move || started_rx.recv().expect("cold load started"))
        .await
        .expect("started task");

    let mut queries = Vec::new();
    for _ in 0..15 {
        let query_engine = Arc::clone(&engine);
        queries.push(tokio::spawn(async move {
            query_engine.query("cold", &vector_query(&[1.0, 0.0])).await
        }));
    }
    release_tx.send(()).expect("release cold load");
    assert!(first.await.expect("first query task").is_ok());
    for query in queries {
        assert!(query.await.expect("query task").is_ok());
    }
    assert_eq!(store.get_count(), one_load_gets);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn cold_load_admission_bounds_blocking_store_work() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = Arc::new(ColdLoadStore::new(&dir));
    {
        let engine = Engine::with_store(Arc::clone(&store));
        for index in 0..(Engine::cold_load_concurrency() + 2) {
            let namespace = format!("admission-{index}");
            engine
                .upsert(
                    &namespace,
                    vec![doc("a", &[1.0, 0.0], "admission", "code")],
                    Vec::new(),
                    BTreeMap::new(),
                )
                .await
                .expect("upsert");
            engine.force_flush(&namespace).await.expect("flush");
        }
    }

    let expected_bound = Engine::cold_load_concurrency();
    let started_rx = store.block_all_gets();
    let engine = Engine::with_store(Arc::clone(&store));
    let mut queries = Vec::new();
    for index in 0..expected_bound {
        let namespace = format!("admission-{index}");
        let engine_ref = &engine;
        let mut query = Box::pin(async move {
            engine_ref
                .query(&namespace, &vector_query(&[1.0, 0.0]))
                .await
        });
        poll_once_until_pending(&mut query).await;
        queries.push(query);
    }

    tokio::task::spawn_blocking(move || {
        for _ in 0..expected_bound {
            started_rx.recv().expect("cold load entered store");
        }
    })
    .await
    .expect("started receiver task");

    assert_eq!(
        store.active_gets(),
        expected_bound,
        "the admission slots were not all occupied before overflow loads"
    );

    let mut overflow = Vec::new();
    for index in expected_bound..(expected_bound + 2) {
        let namespace = format!("admission-{index}");
        let engine_ref = &engine;
        let mut query = Box::pin(async move {
            engine_ref
                .query(&namespace, &vector_query(&[1.0, 0.0]))
                .await
        });
        poll_once_until_pending(&mut query).await;
        overflow.push(query);
    }
    assert_eq!(
        store.active_gets(),
        expected_bound,
        "overflow loads entered blocking store work before a permit was released"
    );
    assert_eq!(
        store.max_active_gets(),
        expected_bound,
        "blocking cold-load high-water mark exceeded admission capacity before release"
    );

    store.release_all_gets();

    for query in queries {
        assert!(query.await.is_ok());
    }
    for query in overflow {
        assert!(query.await.is_ok());
    }
    assert!(
        store.max_active_gets() <= expected_bound,
        "blocking cold loads exceeded admission bound: {} > {expected_bound}",
        store.max_active_gets()
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn evict_cancels_a_cold_load_queued_for_admission() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = Arc::new(ColdLoadStore::new(&dir));
    {
        let engine = Engine::with_store(Arc::clone(&store));
        for index in 0..=Engine::cold_load_concurrency() {
            let namespace = format!("queued-{index}");
            engine
                .upsert(
                    &namespace,
                    vec![doc("a", &[1.0, 0.0], "queued", "code")],
                    Vec::new(),
                    BTreeMap::new(),
                )
                .await
                .expect("upsert");
            engine.force_flush(&namespace).await.expect("flush");
        }
    }

    let expected_bound = Engine::cold_load_concurrency();
    let started_rx = store.block_all_gets();
    let engine = Arc::new(Engine::with_store(Arc::clone(&store)));
    let mut active_queries = Vec::new();
    for index in 0..expected_bound {
        let namespace = format!("queued-{index}");
        let query_engine = Arc::clone(&engine);
        let mut query = Box::pin(async move {
            query_engine
                .query(&namespace, &vector_query(&[1.0, 0.0]))
                .await
        });
        poll_once_until_pending(&mut query).await;
        active_queries.push(query);
    }
    tokio::task::spawn_blocking(move || {
        for _ in 0..expected_bound {
            started_rx.recv().expect("cold load entered store");
        }
    })
    .await
    .expect("started receiver task");

    let queued_namespace = format!("queued-{expected_bound}");
    let queued_engine = Arc::clone(&engine);
    let queued_name = queued_namespace.clone();
    let mut queued = Box::pin(async move {
        queued_engine
            .query(&queued_name, &vector_query(&[1.0, 0.0]))
            .await
    });
    poll_once_until_pending(&mut queued).await;
    assert_eq!(store.active_gets(), expected_bound);

    let evict_engine = Arc::clone(&engine);
    let evict = Box::pin(async move { evict_engine.evict(&queued_namespace).await });
    let (evict_result, queued_result) = tokio::time::timeout(Duration::from_secs(1), async {
        tokio::join!(evict, queued)
    })
    .await
    .expect("eviction remained behind a queued cold load");
    evict_result.expect("eviction failed");
    assert!(matches!(queued_result, Err(Error::NotFound(_))));

    store.release_all_gets();
    for query in active_queries {
        assert!(query.await.is_ok());
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn other_namespace_queries_do_not_wait_for_a_cold_load() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = Arc::new(ColdLoadStore::new(&dir));
    {
        let engine = Engine::with_store(Arc::clone(&store));
        for namespace in ["cold", "hot"] {
            engine
                .upsert(
                    namespace,
                    vec![doc("a", &[1.0, 0.0], namespace, "code")],
                    Vec::new(),
                    BTreeMap::new(),
                )
                .await
                .expect("upsert");
            engine.force_flush(namespace).await.expect("flush");
        }
    }

    let (started_rx, release_tx) = store.block_next_get_for("cold");
    let engine = Arc::new(Engine::with_store(Arc::clone(&store)));
    let cold_engine = Arc::clone(&engine);
    let cold =
        tokio::spawn(async move { cold_engine.query("cold", &vector_query(&[1.0, 0.0])).await });
    tokio::task::spawn_blocking(move || started_rx.recv().expect("cold load started"))
        .await
        .expect("started task");

    let hot_engine = Arc::clone(&engine);
    let hot =
        tokio::spawn(async move { hot_engine.query("hot", &vector_query(&[1.0, 0.0])).await });
    let hot_result = tokio::time::timeout(Duration::from_secs(1), hot)
        .await
        .expect("hot query serialized behind cold load")
        .expect("hot query task")
        .expect("hot query");
    assert_eq!(hot_result[0].id, "a");

    release_tx.send(()).expect("release cold load");
    assert!(cold.await.expect("cold query task").is_ok());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn delete_waits_for_a_cold_load_before_removing_storage() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = Arc::new(ColdLoadStore::new(&dir));
    {
        let engine = Engine::with_store(Arc::clone(&store));
        engine
            .upsert(
                "cold",
                vec![doc("a", &[1.0, 0.0], "cold", "code")],
                Vec::new(),
                BTreeMap::new(),
            )
            .await
            .expect("upsert");
        engine.force_flush("cold").await.expect("flush");
    }

    let (started_rx, release_tx) = store.block_next_get_for("cold");
    let engine = Arc::new(Engine::with_store(Arc::clone(&store)));
    let query_engine = Arc::clone(&engine);
    let query =
        tokio::spawn(async move { query_engine.query("cold", &vector_query(&[1.0, 0.0])).await });
    tokio::task::spawn_blocking(move || started_rx.recv().expect("cold load started"))
        .await
        .expect("started task");

    let delete_engine = Arc::clone(&engine);
    let delete = tokio::spawn(async move { delete_engine.delete_namespace("cold").await });
    tokio::task::yield_now().await;
    release_tx.send(()).expect("release cold load");
    assert!(matches!(
        query.await.expect("query task"),
        Err(Error::NotFound(_))
    ));
    assert!(delete.await.expect("delete task").is_ok());
    assert!(store.list("ns/cold/").expect("list objects").is_empty());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn evict_cancels_a_cold_load_without_losing_storage() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = Arc::new(ColdLoadStore::new(&dir));
    {
        let engine = Engine::with_store(Arc::clone(&store));
        engine
            .upsert(
                "cold",
                vec![doc("a", &[1.0, 0.0], "cold", "code")],
                Vec::new(),
                BTreeMap::new(),
            )
            .await
            .expect("upsert");
        engine.force_flush("cold").await.expect("flush");
    }

    let (started_rx, release_tx) = store.block_next_get_for("cold");
    let engine = Arc::new(Engine::with_store(Arc::clone(&store)));
    let query_engine = Arc::clone(&engine);
    let query =
        tokio::spawn(async move { query_engine.query("cold", &vector_query(&[1.0, 0.0])).await });
    tokio::task::spawn_blocking(move || started_rx.recv().expect("cold load started"))
        .await
        .expect("started task");

    let evict_engine = Arc::clone(&engine);
    let evict = tokio::spawn(async move { evict_engine.evict("cold").await });
    tokio::task::yield_now().await;
    release_tx.send(()).expect("release cold load");
    assert!(matches!(
        query.await.expect("query task"),
        Err(Error::NotFound(_))
    ));
    assert!(evict.await.expect("evict task").is_ok());
    assert_eq!(
        engine
            .query("cold", &vector_query(&[1.0, 0.0]))
            .await
            .expect("retry after evict")[0]
            .id,
        "a"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn failed_cold_load_is_retryable() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = Arc::new(ColdLoadStore::new(&dir));
    {
        let engine = Engine::with_store(Arc::clone(&store));
        engine
            .upsert(
                "retry",
                vec![doc("a", &[1.0, 0.0], "retry", "code")],
                Vec::new(),
                BTreeMap::new(),
            )
            .await
            .expect("upsert");
        engine.force_flush("retry").await.expect("flush");
    }

    store.fail_next_get_for("retry");
    let engine = Engine::with_store(Arc::clone(&store));
    assert!(matches!(
        engine.query("retry", &vector_query(&[1.0, 0.0])).await,
        Err(Error::Store(_))
    ));
    assert_eq!(
        engine
            .query("retry", &vector_query(&[1.0, 0.0]))
            .await
            .expect("retry cold load")[0]
            .id,
        "a"
    );

    engine.evict("retry").await.expect("evict before I/O retry");
    store.fail_next_get_as_io_for("retry");
    assert!(matches!(
        engine.query("retry", &vector_query(&[1.0, 0.0])).await,
        Err(Error::Io(_))
    ));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn failed_cold_load_fans_out_one_error_to_all_waiters() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = Arc::new(ColdLoadStore::new(&dir));
    {
        let engine = Engine::with_store(Arc::clone(&store));
        engine
            .upsert(
                "fanout",
                vec![doc("a", &[1.0, 0.0], "fanout", "code")],
                Vec::new(),
                BTreeMap::new(),
            )
            .await
            .expect("upsert");
        engine.force_flush("fanout").await.expect("flush");
    }

    let (started_rx, release_tx) = store.block_next_get_for("fanout");
    store.fail_next_get_for("fanout");
    let engine = Arc::new(Engine::with_store(Arc::clone(&store)));
    let first_engine = Arc::clone(&engine);
    let first = tokio::spawn(async move {
        first_engine
            .query("fanout", &vector_query(&[1.0, 0.0]))
            .await
    });
    tokio::task::spawn_blocking(move || started_rx.recv().expect("cold load started"))
        .await
        .expect("started task");

    // Poll every waiter into the shared slot before releasing the one store
    // load. This makes the fanout ordering channel-driven, not scheduler- or
    // sleep-dependent.
    let mut waiters = Vec::new();
    for _ in 0..3 {
        let waiter_engine = Arc::clone(&engine);
        let mut waiter = Box::pin(async move {
            waiter_engine
                .query("fanout", &vector_query(&[1.0, 0.0]))
                .await
        });
        poll_once_until_pending(&mut waiter).await;
        waiters.push(waiter);
    }
    release_tx.send(()).expect("release cold load");

    assert!(matches!(
        first.await.expect("first query task"),
        Err(Error::Store(_))
    ));
    for waiter in waiters {
        assert!(matches!(waiter.await, Err(Error::Store(_))));
    }
    assert_eq!(
        engine
            .query("fanout", &vector_query(&[1.0, 0.0]))
            .await
            .expect("retry after fanout failure")[0]
            .id,
        "a"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cancelled_evict_finishes_and_reload_joins_the_transition() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = Arc::new(ColdLoadStore::new(&dir));
    {
        let engine = Engine::with_store(Arc::clone(&store));
        engine
            .upsert(
                "evict-race",
                vec![doc("a", &[1.0, 0.0], "evict", "code")],
                Vec::new(),
                BTreeMap::new(),
            )
            .await
            .expect("upsert");
        engine.force_flush("evict-race").await.expect("flush");
    }

    let (started_rx, release_tx) = store.block_next_get_for("evict-race");
    let engine = Arc::new(Engine::with_store(Arc::clone(&store)));
    let query_engine = Arc::clone(&engine);
    let query = tokio::spawn(async move {
        query_engine
            .query("evict-race", &vector_query(&[1.0, 0.0]))
            .await
    });
    tokio::task::spawn_blocking(move || started_rx.recv().expect("cold load started"))
        .await
        .expect("started task");

    let mut evict = Box::pin(engine.evict("evict-race"));
    poll_once_until_pending(&mut evict).await;
    // Dropping the caller future must not cancel the detached finalizer.
    drop(evict);

    let reload_query = vector_query(&[1.0, 0.0]);
    let mut reload = Box::pin(engine.query("evict-race", &reload_query));
    poll_once_until_pending(&mut reload).await;
    release_tx.send(()).expect("release cold load");

    assert!(matches!(
        query.await.expect("query task"),
        Err(Error::NotFound(_))
    ));
    assert_eq!(reload.await.expect("reload after eviction")[0].id, "a");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cancelled_delete_finishes_after_the_caller_future_is_dropped() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = Arc::new(ColdLoadStore::new(&dir));
    let engine = Arc::new(Engine::with_store(Arc::clone(&store)));
    engine
        .upsert(
            "delete-cancel",
            vec![doc("a", &[1.0, 0.0], "delete", "code")],
            Vec::new(),
            BTreeMap::new(),
        )
        .await
        .expect("upsert");
    engine.force_flush("delete-cancel").await.expect("flush");

    let (started_rx, release_tx) = store.block_next_delete_for("delete-cancel");
    let mut delete = Box::pin(engine.delete_namespace("delete-cancel"));
    poll_once_until_pending(&mut delete).await;
    tokio::task::spawn_blocking(move || started_rx.recv().expect("delete started"))
        .await
        .expect("started task");
    drop(delete);
    release_tx.send(()).expect("release delete");

    let result = tokio::time::timeout(
        Duration::from_secs(2),
        engine.query("delete-cancel", &vector_query(&[1.0, 0.0])),
    )
    .await
    .expect("query remained behind cancelled delete")
    .expect_err("deleted namespace unexpectedly reloaded");
    assert!(matches!(result, Error::NotFound(_)));
    assert!(store
        .list("ns/delete-cancel/")
        .expect("list objects")
        .is_empty());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 6)]
async fn channel_orchestrated_six_operation_lifecycle_interleaving() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = Arc::new(ColdLoadStore::new(&dir));
    let engine = Arc::new(Engine::with_store(Arc::clone(&store)));

    engine
        .upsert(
            "load-evict",
            vec![doc("a", &[1.0, 0.0], "load", "code")],
            Vec::new(),
            BTreeMap::new(),
        )
        .await
        .expect("seed load-evict");
    engine
        .force_flush("load-evict")
        .await
        .expect("flush load-evict");
    engine
        .upsert(
            "delete",
            vec![doc("a", &[1.0, 0.0], "delete", "code")],
            Vec::new(),
            BTreeMap::new(),
        )
        .await
        .expect("seed delete");
    engine.force_flush("delete").await.expect("flush delete");

    // Build three segments; the single force_flush operation below creates
    // the fourth and must enter compaction.
    for index in 0..3 {
        engine
            .upsert(
                "compact",
                vec![doc(
                    &format!("seed-{index}"),
                    &[1.0, 0.0],
                    "compact",
                    "code",
                )],
                Vec::new(),
                BTreeMap::new(),
            )
            .await
            .expect("seed compact upsert");
        engine
            .force_flush("compact")
            .await
            .expect("seed compact flush");
    }
    store.reset_compact_puts();

    // Exact load-vs-evict-vs-reload ordering. The cold load cannot publish,
    // eviction cannot drain, and reload cannot bypass the registered
    // transition until this channel is released.
    engine
        .evict("load-evict")
        .await
        .expect("evict before cold race");
    let (load_started_rx, load_release_tx) = store.block_next_get_for("load-evict");
    let cold_engine = Arc::clone(&engine);
    let cold_query = tokio::spawn(async move {
        cold_engine
            .query("load-evict", &vector_query(&[1.0, 0.0]))
            .await
    });
    tokio::task::spawn_blocking(move || load_started_rx.recv().expect("cold load started"))
        .await
        .expect("load marker task");

    let mut evict = Box::pin(engine.evict("load-evict"));
    poll_once_until_pending(&mut evict).await;
    let reload_query = vector_query(&[1.0, 0.0]);
    let mut reload = Box::pin(engine.query("load-evict", &reload_query));
    poll_once_until_pending(&mut reload).await;
    load_release_tx.send(()).expect("release cold load");

    assert!(matches!(
        cold_query.await.expect("cold query task"),
        Err(Error::NotFound(_))
    ));
    evict.await.expect("evict future");
    assert_eq!(reload.await.expect("reload future")[0].id, "a");

    // Upsert + flush/compaction are real operations, and the store marker
    // proves the force_flush call crossed the compaction publication path.
    engine
        .upsert(
            "compact",
            vec![doc("fourth", &[1.0, 0.0], "compact", "code")],
            Vec::new(),
            BTreeMap::new(),
        )
        .await
        .expect("fourth upsert");
    engine
        .force_flush("compact")
        .await
        .expect("flush and compaction");
    assert!(
        store.compact_puts() > 0,
        "compaction publication was not entered"
    );
    assert!(engine
        .last_flush_error("compact")
        .await
        .expect("last flush error")
        .is_none());

    // Delete is held in storage after its lifecycle transition is registered.
    // Query and upsert must both join that transition; the latter recreates
    // the namespace only after deletion has fully completed.
    let (delete_started_rx, delete_release_tx) = store.block_next_delete_for("delete");
    let mut delete = Box::pin(engine.delete_namespace("delete"));
    poll_once_until_pending(&mut delete).await;
    tokio::task::spawn_blocking(move || delete_started_rx.recv().expect("delete started"))
        .await
        .expect("delete marker task");

    let delete_query_spec = vector_query(&[1.0, 0.0]);
    let mut delete_query = Box::pin(engine.query("delete", &delete_query_spec));
    poll_once_until_pending(&mut delete_query).await;
    let mut recreate = Box::pin(engine.upsert(
        "delete",
        vec![doc("recreated", &[1.0, 0.0], "recreated", "code")],
        Vec::new(),
        BTreeMap::new(),
    ));
    poll_once_until_pending(&mut recreate).await;
    delete_release_tx.send(()).expect("release delete");

    delete.await.expect("delete future");
    assert!(matches!(delete_query.await, Err(Error::NotFound(_))));
    recreate.await.expect("recreate upsert future");
    assert_eq!(
        engine
            .query("delete", &vector_query(&[1.0, 0.0]))
            .await
            .expect("query recreated namespace")[0]
            .id,
        "recreated"
    );
}

#[tokio::test]
async fn automatic_threshold_flushes_and_failed_flush_retries() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = open_engine(&dir);
    let docs = (0..1_000)
        .map(|index| doc(&format!("doc-{index}"), &[1.0, 0.0], "value", "code"))
        .collect();
    engine
        .upsert("threshold", docs, Vec::new(), BTreeMap::new())
        .await
        .expect("upsert");
    let mut flushed = false;
    for _ in 0..100 {
        if std::fs::read_dir(dir.path().join("ns/threshold/segments")).is_ok_and(|entries| {
            entries
                .flatten()
                .any(|entry| entry.path().join("meta.json").is_file())
        }) {
            flushed = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(
        flushed,
        "automatic threshold flush did not publish a segment"
    );

    let retry_dir = tempfile::tempdir().expect("tempdir");
    let store = Arc::new(FailOnceStore::new(&retry_dir));
    let retry_engine = Engine::with_store(store);
    retry_engine
        .upsert(
            "retry",
            vec![doc("a", &[1.0, 0.0], "retry", "code")],
            Vec::new(),
            BTreeMap::new(),
        )
        .await
        .expect("upsert");
    assert!(retry_engine.force_flush("retry").await.is_err());
    retry_engine
        .force_flush("retry")
        .await
        .expect("retry flush");
    assert_eq!(
        retry_engine
            .query("retry", &vector_query(&[1.0, 0.0]))
            .await
            .expect("query")[0]
            .id,
        "a"
    );
}

#[tokio::test]
async fn restart_derives_and_repairs_failed_wal_retirement() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = Arc::new(FailWalDeleteStore::new(&dir));
    {
        let engine = Engine::with_store(Arc::clone(&store));
        engine
            .upsert(
                "retire",
                vec![doc("a", &[1.0, 0.0], "retire", "code")],
                Vec::new(),
                BTreeMap::new(),
            )
            .await
            .expect("upsert");
        assert!(engine.force_flush("retire").await.is_err());
        assert!(std::fs::read_dir(dir.path().join("ns/retire/wal"))
            .expect("wal directory")
            .next()
            .is_some());
    }
    let engine = Engine::with_store(store);
    assert_eq!(
        engine
            .query("retire", &vector_query(&[1.0, 0.0]))
            .await
            .expect("cold query")[0]
            .id,
        "a"
    );
    assert!(std::fs::read_dir(dir.path().join("ns/retire/wal"))
        .expect("wal directory")
        .next()
        .is_none());
}

#[tokio::test]
async fn overwrite_and_delete_never_resurface_after_flush_boundary() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = open_engine(&dir);
    engine
        .upsert(
            "demo",
            vec![doc("a", &[1.0, 0.0], "old", "code")],
            Vec::new(),
            BTreeMap::new(),
        )
        .await
        .expect("upsert");
    engine.force_flush("demo").await.expect("flush");
    engine
        .upsert(
            "demo",
            vec![doc("a", &[0.0, 1.0], "new", "code")],
            Vec::new(),
            BTreeMap::new(),
        )
        .await
        .expect("overwrite");
    assert_eq!(
        engine
            .query("demo", &vector_query(&[0.0, 1.0]))
            .await
            .expect("query")[0]
            .id,
        "a"
    );
    engine.force_flush("demo").await.expect("flush");
    let old_vector = engine
        .query("demo", &vector_query(&[1.0, 0.0]))
        .await
        .expect("query");
    assert_eq!(old_vector[0].id, "a");
    assert_eq!(old_vector[0].score, 0.0);
    engine
        .upsert("demo", Vec::new(), vec!["a".to_owned()], BTreeMap::new())
        .await
        .expect("delete");
    assert!(engine
        .query("demo", &vector_query(&[0.0, 1.0]))
        .await
        .expect("query")
        .is_empty());
    engine.force_flush("demo").await.expect("flush");
    assert!(engine
        .query("demo", &vector_query(&[1.0, 0.0]))
        .await
        .expect("query")
        .is_empty());
}

#[tokio::test]
async fn persisted_tombstones_survive_a_later_segment_and_cold_start() {
    let dir = tempfile::tempdir().expect("tempdir");
    {
        let engine = open_engine(&dir);
        engine
            .upsert(
                "demo",
                vec![doc("old", &[1.0, 0.0], "old", "code")],
                Vec::new(),
                BTreeMap::new(),
            )
            .await
            .expect("upsert");
        engine.force_flush("demo").await.expect("flush");
        engine
            .upsert("demo", Vec::new(), vec!["old".to_owned()], BTreeMap::new())
            .await
            .expect("delete");
        engine
            .force_flush("demo")
            .await
            .expect("tombstone checkpoint");
        engine
            .upsert(
                "demo",
                vec![doc("new", &[0.0, 1.0], "new", "code")],
                Vec::new(),
                BTreeMap::new(),
            )
            .await
            .expect("upsert");
        engine.force_flush("demo").await.expect("flush");
        assert!(std::fs::read_dir(dir.path().join("ns/demo/wal"))
            .expect("wal directory")
            .next()
            .is_none());
    }
    let engine = open_engine(&dir);
    assert_eq!(
        engine
            .query("demo", &vector_query(&[1.0, 0.0]))
            .await
            .expect("cold query")[0]
            .id,
        "new"
    );
    assert_eq!(
        engine
            .query("demo", &vector_query(&[0.0, 1.0]))
            .await
            .expect("cold query")[0]
            .id,
        "new"
    );
}

#[tokio::test]
async fn cold_start_and_lru_eviction_reload_from_store() {
    let dir = tempfile::tempdir().expect("tempdir");
    {
        let engine = open_engine(&dir);
        engine
            .upsert(
                "persisted",
                vec![doc("a", &[1.0, 0.0], "persisted", "code")],
                Vec::new(),
                BTreeMap::new(),
            )
            .await
            .expect("upsert");
        engine.force_flush("persisted").await.expect("flush");
    }
    let engine = open_engine(&dir);
    assert_eq!(
        engine
            .query("persisted", &vector_query(&[1.0, 0.0]))
            .await
            .expect("cold query")[0]
            .id,
        "a"
    );
    for index in 0..=Engine::hot_cache_capacity() {
        let name = format!("ns-{index}");
        engine
            .upsert(
                &name,
                vec![doc("a", &[1.0, 0.0], "value", "code")],
                Vec::new(),
                BTreeMap::new(),
            )
            .await
            .expect("upsert");
    }
    assert!(engine.loaded_namespaces().await.expect("loaded").len() <= 8);
    assert_eq!(
        engine
            .query("persisted", &vector_query(&[1.0, 0.0]))
            .await
            .expect("eviction reload")[0]
            .id,
        "a"
    );
}

#[tokio::test]
async fn hybrid_rrf_and_filter_are_applied_together() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = open_engine(&dir);
    engine
        .upsert(
            "demo",
            vec![
                doc("a", &[1.0, 0.0], "rust", "code"),
                doc("b", &[0.9, 0.1], "rust", "other"),
            ],
            Vec::new(),
            BTreeMap::from([("text".to_owned(), true)]),
        )
        .await
        .expect("upsert");
    let results = engine
        .query(
            "demo",
            &Query {
                vector: Some(vec![1.0, 0.0]),
                text: Some("rust".to_owned()),
                filter: Some(Filter::Eq {
                    field: "category".to_owned(),
                    value: AttrValue::String("other".to_owned()),
                }),
                top_k: 10,
                include_attributes: true,
            },
        )
        .await
        .expect("hybrid query");
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, "b");
    assert_eq!(
        results[0].attributes.as_ref().expect("attributes")["category"],
        AttrValue::String("other".to_owned())
    );
    let all_results = engine
        .query(
            "demo",
            &Query {
                vector: Some(vec![1.0, 0.0]),
                text: Some("rust".to_owned()),
                filter: None,
                top_k: 10,
                include_attributes: false,
            },
        )
        .await
        .expect("hybrid query");
    assert_eq!(all_results.len(), 2);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn concurrent_delete_waits_for_an_inflight_write_before_removing_storage() {
    let dir = tempfile::tempdir().expect("tempdir");
    let started = Arc::new(Barrier::new(2));
    let release = Arc::new(Barrier::new(2));
    let store = Arc::new(BlockingStore::new(
        &dir,
        Arc::clone(&started),
        Arc::clone(&release),
    ));
    let engine = Arc::new(Engine::with_store(store));
    let writer_engine = Arc::clone(&engine);
    let writer = tokio::spawn(async move {
        writer_engine
            .upsert(
                "race",
                vec![doc("a", &[1.0, 0.0], "race", "code")],
                Vec::new(),
                BTreeMap::new(),
            )
            .await
    });
    started.wait();
    let deleter_engine = Arc::clone(&engine);
    let deleter = tokio::spawn(async move { deleter_engine.delete_namespace("race").await });
    release.wait();
    assert!(writer.await.expect("writer task").is_ok());
    assert!(deleter.await.expect("delete task").is_ok());
    assert!(matches!(
        engine.query("race", &vector_query(&[1.0, 0.0])).await,
        Err(Error::NotFound(_))
    ));
}

#[tokio::test]
async fn http_upsert_query_and_error_mapping() {
    let dir = tempfile::tempdir().expect("tempdir");
    let app = router(Arc::new(open_engine(&dir)));
    let body = r#"{"upserts":[{"id":"a","vector":[1,0],"attributes":{"text":"rust"}}],"schema":{"text":{"full_text_search":true}}}"#;
    let response = app
        .clone()
        .oneshot(
            Request::post("/v1/namespaces/demo")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(response.status(), StatusCode::OK);

    let response = app
        .clone()
        .oneshot(
            Request::post("/v1/namespaces/demo/query")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"text":"rust","top_k":1}"#))
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(response.status(), StatusCode::OK);

    let response = app
        .clone()
        .oneshot(
            Request::post("/v1/namespaces/demo/query")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"vector":[1],"top_k":1}"#))
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);

    let response = app
        .clone()
        .oneshot(
            Request::post("/v1/namespaces/demo/query")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"vector":[1,0],"top_k":513}"#))
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);

    let response = app
        .oneshot(
            Request::post("/v1/namespaces/missing/query")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"vector":[1,0],"top_k":1}"#))
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn unknown_delete_is_not_found() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = open_engine(&dir);
    assert!(matches!(
        engine.delete_namespace("missing").await,
        Err(Error::NotFound(_))
    ));
}

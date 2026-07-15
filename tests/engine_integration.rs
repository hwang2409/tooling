use std::collections::{BTreeMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Barrier};
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
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

struct FailOnceStore {
    inner: LocalDirStore,
    fail_segment_meta: AtomicBool,
}

struct FailWalDeleteStore {
    inner: LocalDirStore,
    fail_once: AtomicBool,
}

impl FailWalDeleteStore {
    fn new(dir: &TempDir) -> Self {
        Self {
            inner: LocalDirStore::new(dir.path()).expect("store"),
            fail_once: AtomicBool::new(true),
        }
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

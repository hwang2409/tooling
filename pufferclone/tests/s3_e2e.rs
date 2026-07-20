use std::collections::{BTreeMap, HashSet};
use std::env;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::task::Poll;
use std::time::{SystemTime, UNIX_EPOCH};

use futures_util::future::poll_fn;
use pufferclone::engine::Engine;
use pufferclone::namespace::Query;
use pufferclone::segment::SegmentBuilder;
use pufferclone::store::ObjectStore;
use pufferclone::store_s3::S3Store;
use pufferclone::wal::Manifest;
use pufferclone::{AttrValue, Doc, Result};

static NAMESPACE_COUNTER: AtomicU64 = AtomicU64::new(0);

struct CountingS3Store {
    inner: Arc<S3Store>,
    get_count: AtomicUsize,
    docs_get_count: AtomicUsize,
    block_next_meta: AtomicBool,
    started: Mutex<Option<mpsc::Sender<()>>>,
    release: Mutex<Option<mpsc::Receiver<()>>>,
}

struct S3Fixture {
    store: Arc<CountingS3Store>,
    namespace: String,
}

impl CountingS3Store {
    fn new(inner: Arc<S3Store>) -> Self {
        Self {
            inner,
            get_count: AtomicUsize::new(0),
            docs_get_count: AtomicUsize::new(0),
            block_next_meta: AtomicBool::new(false),
            started: Mutex::new(None),
            release: Mutex::new(None),
        }
    }

    fn reset_get_counts(&self) {
        self.get_count.store(0, Ordering::Release);
        self.docs_get_count.store(0, Ordering::Release);
    }

    fn get_count(&self) -> usize {
        self.get_count.load(Ordering::Acquire)
    }

    fn docs_get_count(&self) -> usize {
        self.docs_get_count.load(Ordering::Acquire)
    }

    fn block_next_segment_load(&self) -> (mpsc::Receiver<()>, mpsc::Sender<()>) {
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        *self.started.lock().expect("started lock") = Some(started_tx);
        *self.release.lock().expect("release lock") = Some(release_rx);
        self.block_next_meta.store(true, Ordering::Release);
        (started_rx, release_tx)
    }
}

impl ObjectStore for CountingS3Store {
    fn put(&self, key: &str, bytes: &[u8]) -> Result<()> {
        self.inner.put(key, bytes)
    }

    fn get(&self, key: &str) -> Result<Vec<u8>> {
        self.get_count.fetch_add(1, Ordering::AcqRel);
        let is_meta = key.ends_with("/meta.json");
        if key.ends_with("/docs.bin") {
            self.docs_get_count.fetch_add(1, Ordering::AcqRel);
        }
        if is_meta && self.block_next_meta.swap(false, Ordering::AcqRel) {
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
                .expect("release receiver")
                .recv()
                .expect("release sender");
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

impl Drop for S3Fixture {
    fn drop(&mut self) {
        let prefix = format!("ns/{}/", self.namespace);
        if let Ok(keys) = self.store.list(&prefix) {
            for key in keys {
                let _ = self.store.delete(&key);
            }
        }
    }
}

fn fixture(label: &str) -> Option<S3Fixture> {
    let url = match env::var("PUFFERCLONE_TEST_S3_URL") {
        Ok(url) => url,
        Err(_) => {
            eprintln!("skipping S3 engine test: PUFFERCLONE_TEST_S3_URL is not set");
            return None;
        }
    };
    let bucket = env::var("PUFFERCLONE_TEST_S3_BUCKET")
        .or_else(|_| env::var("PUFFERCLONE_S3_BUCKET"))
        .unwrap_or_else(|_| "pufferclone-test".to_owned());
    let store = Arc::new(CountingS3Store::new(Arc::new(
        S3Store::new(url, bucket).expect("S3 store"),
    )));
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock")
        .as_nanos();
    let namespace = format!(
        "s3-e2e-{label}-{}-{timestamp}-{}",
        std::process::id(),
        NAMESPACE_COUNTER.fetch_add(1, Ordering::Relaxed)
    );
    Some(S3Fixture { store, namespace })
}

fn doc(id: &str, vector: &[f32], text: &str) -> Doc {
    Doc {
        id: id.to_owned(),
        vector: Some(vector.to_vec()),
        attributes: BTreeMap::from([("text".to_owned(), AttrValue::String(text.to_owned()))]),
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

async fn poll_once_until_pending<F>(future: &mut Pin<Box<F>>)
where
    F: Future,
{
    let pending = poll_fn(|cx| match future.as_mut().poll(cx) {
        Poll::Pending => Poll::Ready(true),
        Poll::Ready(_) => Poll::Ready(false),
    })
    .await;
    assert!(pending, "future completed before fanout registration");
}

#[tokio::test]
async fn full_lifecycle_upsert_flush_and_query_uses_minio() {
    let Some(fixture) = fixture("lifecycle") else {
        return;
    };
    let engine = Engine::with_store(Arc::clone(&fixture.store));
    engine
        .upsert(
            &fixture.namespace,
            vec![doc("lifecycle-doc", &[1.0, 0.0], "minio lifecycle")],
            Vec::new(),
            BTreeMap::from([("text".to_owned(), true)]),
        )
        .await
        .expect("upsert");
    engine.force_flush(&fixture.namespace).await.expect("flush");

    let results = engine
        .query(&fixture.namespace, &text_query("lifecycle"))
        .await
        .expect("query");
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, "lifecycle-doc");
}

#[tokio::test]
async fn compaction_deletes_inputs_and_reopen_cleans_orphans_on_minio() {
    let Some(fixture) = fixture("compaction") else {
        return;
    };
    let engine = Engine::with_store(Arc::clone(&fixture.store));
    for index in 0..3 {
        engine
            .upsert(
                &fixture.namespace,
                vec![doc(
                    &format!("doc-{index}"),
                    &[1.0, index as f32],
                    "compaction",
                )],
                Vec::new(),
                BTreeMap::from([("text".to_owned(), true)]),
            )
            .await
            .expect("seed upsert");
        engine
            .force_flush(&fixture.namespace)
            .await
            .expect("seed flush");
    }

    let before = Manifest::load(fixture.store.as_ref(), &fixture.namespace).expect("manifest");
    let old_keys = before
        .segments
        .iter()
        .flat_map(|segment| {
            fixture
                .store
                .list(&format!(
                    "ns/{}/segments/{}/",
                    fixture.namespace, segment.id
                ))
                .expect("old segment keys")
        })
        .collect::<Vec<_>>();
    assert!(!old_keys.is_empty(), "captured input segment keys");

    engine
        .upsert(
            &fixture.namespace,
            vec![doc("doc-3", &[1.0, 3.0], "compaction")],
            vec!["doc-0".to_owned()],
            BTreeMap::new(),
        )
        .await
        .expect("compaction upsert");
    engine
        .force_flush(&fixture.namespace)
        .await
        .expect("compaction flush");

    let after = Manifest::load(fixture.store.as_ref(), &fixture.namespace).expect("manifest");
    assert_eq!(after.segments.len(), 1);
    let remaining_keys = fixture
        .store
        .list(&format!("ns/{}/segments/", fixture.namespace))
        .expect("remaining segments");
    let expected_keys = after
        .segments
        .iter()
        .flat_map(|segment| {
            fixture
                .store
                .list(&format!(
                    "ns/{}/segments/{}/",
                    fixture.namespace, segment.id
                ))
                .expect("post-compaction segment keys")
        })
        .collect::<HashSet<_>>();
    assert!(remaining_keys.iter().all(|key| expected_keys.contains(key)));
    assert!(old_keys
        .iter()
        .all(|key| !remaining_keys.iter().any(|remaining| remaining == key)));
    let results = engine
        .query(&fixture.namespace, &text_query("compaction"))
        .await
        .expect("compaction query");
    assert_eq!(results.len(), 3);
    assert!(!results.iter().any(|result| result.id == "doc-0"));

    let orphan_id = "orphan-for-reopen";
    SegmentBuilder::new(fixture.store.as_ref(), &fixture.namespace, (100, 100))
        .with_id(orphan_id)
        .build_docs(vec![doc("orphan", &[0.0, 1.0], "orphan")])
        .expect("orphan segment");
    let orphan_keys = fixture
        .store
        .list(&format!("ns/{}/segments/{orphan_id}/", fixture.namespace))
        .expect("orphan keys");
    assert!(!orphan_keys.is_empty());

    drop(engine);
    let reopened = Engine::with_store(Arc::clone(&fixture.store));
    let reopened_results = reopened
        .query(&fixture.namespace, &text_query("compaction"))
        .await
        .expect("reopen query");
    assert_eq!(reopened_results.len(), 3);
    let after_reopen = fixture
        .store
        .list(&format!("ns/{}/segments/{orphan_id}/", fixture.namespace))
        .expect("orphan cleanup listing");
    assert!(orphan_keys
        .iter()
        .all(|key| !after_reopen.iter().any(|remaining| remaining == key)));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cold_load_fanout_performs_one_segment_load_on_minio() {
    let Some(fixture) = fixture("fanout") else {
        return;
    };
    {
        let engine = Engine::with_store(Arc::clone(&fixture.store));
        engine
            .upsert(
                &fixture.namespace,
                vec![doc("fanout-doc", &[1.0, 0.0], "fanout")],
                Vec::new(),
                BTreeMap::new(),
            )
            .await
            .expect("seed upsert");
        engine
            .force_flush(&fixture.namespace)
            .await
            .expect("seed flush");
    }

    fixture.store.reset_get_counts();
    let (started_rx, release_tx) = fixture.store.block_next_segment_load();
    let engine = Arc::new(Engine::with_store(Arc::clone(&fixture.store)));
    let first_engine = Arc::clone(&engine);
    let namespace = fixture.namespace.clone();
    let first = tokio::spawn(async move {
        first_engine
            .query(&namespace, &vector_query(&[1.0, 0.0]))
            .await
    });
    tokio::task::spawn_blocking(move || started_rx.recv().expect("segment load started"))
        .await
        .expect("segment load notification");

    let mut waiters = Vec::new();
    for _ in 0..3 {
        let waiter_engine = Arc::clone(&engine);
        let namespace = fixture.namespace.clone();
        let mut waiter = Box::pin(async move {
            waiter_engine
                .query(&namespace, &vector_query(&[1.0, 0.0]))
                .await
        });
        poll_once_until_pending(&mut waiter).await;
        waiters.push(waiter);
    }
    release_tx.send(()).expect("release segment load");

    assert_eq!(
        first.await.expect("first query task").expect("first query")[0].id,
        "fanout-doc"
    );
    for waiter in waiters {
        assert_eq!(waiter.await.expect("query")[0].id, "fanout-doc");
    }
    assert_eq!(fixture.store.docs_get_count(), 1);
    assert!(fixture.store.get_count() > 1);
}

#[tokio::test]
async fn engine_reopen_reads_persisted_data_from_minio() {
    let Some(fixture) = fixture("reopen") else {
        return;
    };
    {
        let engine = Engine::with_store(Arc::clone(&fixture.store));
        engine
            .upsert(
                &fixture.namespace,
                vec![doc("reopen-doc", &[0.0, 1.0], "reopen")],
                Vec::new(),
                BTreeMap::new(),
            )
            .await
            .expect("upsert");
        engine.force_flush(&fixture.namespace).await.expect("flush");
    }

    fixture.store.reset_get_counts();
    let reopened = Engine::with_store(Arc::clone(&fixture.store));
    assert!(reopened
        .list_namespaces()
        .expect("namespace listing")
        .contains(&fixture.namespace));
    let results = reopened
        .query(&fixture.namespace, &vector_query(&[0.0, 1.0]))
        .await
        .expect("reopen query");
    assert_eq!(results[0].id, "reopen-doc");
    assert!(fixture.store.get_count() > 0);
}

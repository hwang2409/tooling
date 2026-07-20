use std::env;
use std::fs;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::sync::OnceLock;
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

use pufferclone::store::{LocalDirStore, ObjectStore};
use pufferclone::store_s3::S3Store;
use pufferclone::Error;
use tempfile::TempDir;

static PREFIX_COUNTER: AtomicU64 = AtomicU64::new(0);
static RUN_NONCE: OnceLock<u64> = OnceLock::new();

struct Fixture {
    store: Arc<dyn ObjectStore + Send + Sync>,
    root: Option<TempDir>,
    prefix: String,
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let Ok(keys) = self.store.list(&self.prefix) else {
            return;
        };
        for key in keys {
            let _ = self.store.delete(&key);
        }
    }
}

fn next_prefix() -> String {
    format!(
        "conformance/{}/{}/",
        run_nonce(),
        PREFIX_COUNTER.fetch_add(1, Ordering::Relaxed)
    )
}

fn run_nonce() -> u64 {
    *RUN_NONCE.get_or_init(|| {
        let marker = Box::new(());
        let address = &*marker as *const () as usize;
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock")
            .as_nanos();
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        address.hash(&mut hasher);
        timestamp.hash(&mut hasher);
        std::process::id().hash(&mut hasher);
        hasher.finish()
    })
}

fn local_fixture_with_prefix(prefix: impl Into<String>) -> Fixture {
    let root = tempfile::tempdir().expect("tempdir");
    let store = LocalDirStore::new(root.path()).expect("local store");
    Fixture {
        store: Arc::new(store),
        root: Some(root),
        prefix: prefix.into(),
    }
}

fn local_fixture() -> Fixture {
    local_fixture_with_prefix(next_prefix())
}

fn s3_fixture() -> Option<Fixture> {
    let url = match env::var("PUFFERCLONE_TEST_S3_URL") {
        Ok(url) => url,
        Err(_) => {
            eprintln!("skipping S3 store conformance: PUFFERCLONE_TEST_S3_URL is not set");
            return None;
        }
    };
    let bucket = env::var("PUFFERCLONE_TEST_S3_BUCKET")
        .or_else(|_| env::var("PUFFERCLONE_S3_BUCKET"))
        .unwrap_or_else(|_| "pufferclone-test".to_owned());
    let store = S3Store::new(url, bucket).expect("S3 store");
    Some(Fixture {
        store: Arc::new(store),
        root: None,
        prefix: next_prefix(),
    })
}

fn each_backend(test: impl Fn(&Fixture)) {
    let local = local_fixture();
    test(&local);
    if let Some(s3) = s3_fixture() {
        test(&s3);
    }
}

fn key(fixture: &Fixture, suffix: &str) -> String {
    format!("{}{suffix}", fixture.prefix)
}

#[test]
fn store_conformance_put_get_list_delete_and_write_once() {
    each_backend(|fixture| {
        let a = key(fixture, "a");
        let b = key(fixture, "b");
        fixture.store.put(&a, b"alpha").expect("put a");
        fixture.store.put(&b, b"beta").expect("put b");
        assert_eq!(fixture.store.get(&a).expect("get a"), b"alpha");
        assert_eq!(
            fixture.store.list(&fixture.prefix).expect("list"),
            vec![a.clone(), b]
        );
        assert!(matches!(
            fixture.store.put(&a, b"new"),
            Err(Error::AlreadyExists(key)) if key == a
        ));

        fixture.store.delete(&a).expect("delete a");
        assert!(matches!(
            fixture.store.get(&a),
            Err(Error::NotFound(key)) if key == a
        ));
        assert!(matches!(
            fixture.store.delete(&a),
            Err(Error::NotFound(key)) if key == a
        ));
    });
}

#[test]
fn store_conformance_manifest_overwrite_and_exact_filename() {
    each_backend(|fixture| {
        let manifest = key(fixture, "MANIFEST.json");
        fixture.store.put(&manifest, b"one").expect("put manifest");
        fixture
            .store
            .put(&manifest, b"two")
            .expect("overwrite manifest");
        assert_eq!(fixture.store.get(&manifest).expect("get manifest"), b"two");

        let almost_manifest = key(fixture, "fooMANIFEST.json");
        fixture
            .store
            .put(&almost_manifest, b"one")
            .expect("put almost manifest");
        assert!(matches!(
            fixture.store.put(&almost_manifest, b"two"),
            Err(Error::AlreadyExists(key)) if key == almost_manifest
        ));
    });
}

#[test]
fn store_conformance_concurrent_puts_publish_one_value() {
    each_backend(|fixture| {
        let object_key = key(fixture, "concurrent");
        let workers = (0..8)
            .map(|index| {
                let store = Arc::clone(&fixture.store);
                let object_key = object_key.clone();
                thread::spawn(move || store.put(&object_key, &[index]))
            })
            .collect::<Vec<_>>();
        let results = workers
            .into_iter()
            .map(|worker| worker.join().expect("worker"))
            .collect::<Vec<_>>();

        assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
        assert_eq!(
            results
                .iter()
                .filter(|result| matches!(result, Err(Error::AlreadyExists(_))))
                .count(),
            7
        );
        assert_eq!(fixture.store.list(&fixture.prefix).expect("list").len(), 1);
    });
}

#[test]
fn store_conformance_list_filters_by_prefix() {
    each_backend(|fixture| {
        let exact = key(fixture, "foo");
        let partial = format!("{exact}bar");
        let adjacent = format!("{}x/other", fixture.prefix);
        fixture.store.put(&exact, b"exact").expect("put exact");
        fixture
            .store
            .put(&partial, b"partial")
            .expect("put partial");
        fixture
            .store
            .put(&adjacent, b"adjacent")
            .expect("put adjacent");
        assert_eq!(
            fixture.store.list(&exact).expect("list"),
            vec![exact, partial]
        );
    });
}

#[test]
fn store_conformance_rejects_invalid_keys() {
    each_backend(|fixture| {
        for invalid in [
            "../x",
            "./x",
            "ns//x",
            "ns\\x",
            "/absolute",
            "ns/./x",
            "ns/x/.",
            ".tmp/hidden",
            "./.tmp/x",
        ] {
            assert!(
                matches!(fixture.store.put(invalid, b"x"), Err(Error::InvalidKey(_))),
                "{invalid}"
            );
            assert!(
                matches!(fixture.store.get(invalid), Err(Error::InvalidKey(_))),
                "{invalid}"
            );
            assert!(
                matches!(fixture.store.delete(invalid), Err(Error::InvalidKey(_))),
                "{invalid}"
            );
            assert!(
                matches!(fixture.store.list(invalid), Err(Error::InvalidKey(_))),
                "{invalid}"
            );
        }
    });
}

#[test]
fn local_store_list_hides_internal_temp_files() {
    let fixture = local_fixture();
    let live = key(&fixture, "live");
    fixture.store.put(&live, b"live").expect("put live");
    fs::write(
        fixture
            .root
            .as_ref()
            .expect("local root")
            .path()
            .join(".tmp/stray-temp"),
        b"partial",
    )
    .expect("stray temp");

    assert_eq!(fixture.store.list("").expect("list"), vec![live]);
}

#[test]
fn fixture_cleanup_respects_delimited_namespaces() {
    let root = tempfile::tempdir().expect("tempdir");
    let store: Arc<dyn ObjectStore + Send + Sync> =
        Arc::new(LocalDirStore::new(root.path()).expect("local store"));
    let first = Fixture {
        store: Arc::clone(&store),
        root: None,
        prefix: "x/1/".to_owned(),
    };
    let second = Fixture {
        store,
        root: Some(root),
        prefix: "x/10/".to_owned(),
    };
    let first_key = key(&first, "first");
    let second_key = key(&second, "second");
    first.store.put(&first_key, b"first").expect("put first");
    second
        .store
        .put(&second_key, b"second")
        .expect("put second");

    drop(first);

    assert!(matches!(
        second.store.get(&first_key),
        Err(Error::NotFound(key)) if key == first_key
    ));
    assert_eq!(
        second.store.get(&second_key).expect("get second"),
        b"second"
    );
}

//! Namespace registry and bounded hot cache.

use std::collections::{HashMap, VecDeque};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex as StdMutex, Weak};

use tokio::sync::{watch, Mutex, RwLock};
use tokio::task::JoinHandle;

#[cfg(test)]
use std::sync::OnceLock;
#[cfg(test)]
use tokio::sync::Notify;

use crate::namespace::{Namespace, Query, QueryResult, Schema, SharedStore, WriteSummary};
use crate::store::{LocalDirStore, ObjectStore};
use crate::{Error, Result};

const HOT_CACHE_CAPACITY: usize = 8;

type Loaded = Arc<RwLock<Namespace>>;

struct Worker {
    cancel: watch::Sender<bool>,
    join: JoinHandle<()>,
}

struct NamespaceEntry {
    namespace: Loaded,
    operation: Mutex<()>,
    closing: AtomicBool,
    worker: StdMutex<Option<Worker>>,
}

#[cfg(test)]
struct WorkerRaceHook {
    before_operation: Notify,
    release_before_operation: Notify,
    shutdown_operation_acquired: Notify,
}

#[cfg(test)]
static WORKER_RACE_HOOK: OnceLock<Arc<WorkerRaceHook>> = OnceLock::new();

struct Registry {
    entries: HashMap<String, Arc<NamespaceEntry>>,
    lru: VecDeque<String>,
}

/// The process-wide namespace registry and hot namespace cache.
pub struct Engine {
    store: SharedStore,
    registry: Mutex<Registry>,
}

impl Engine {
    /// Open an engine backed by a local object-store directory.
    pub fn new(root: impl AsRef<Path>) -> Result<Self> {
        let store = Arc::new(LocalDirStore::new(root)?);
        Ok(Self::with_store(store))
    }

    /// Construct an engine over a caller-provided object store.
    pub fn with_store<S>(store: Arc<S>) -> Self
    where
        S: ObjectStore + Send + Sync + 'static,
    {
        Self {
            store,
            registry: Mutex::new(Registry {
                entries: HashMap::new(),
                lru: VecDeque::new(),
            }),
        }
    }

    pub fn hot_cache_capacity() -> usize {
        HOT_CACHE_CAPACITY
    }

    /// Upsert documents and/or delete IDs, creating the namespace on demand.
    pub async fn upsert(
        &self,
        namespace: &str,
        upserts: Vec<crate::Doc>,
        deletes: Vec<String>,
        schema: Schema,
    ) -> Result<WriteSummary> {
        let entry = match self.get_or_open(namespace).await {
            Ok(entry) => entry,
            Err(Error::NotFound(_)) => {
                return self
                    .create_and_upsert(namespace, upserts, deletes, schema)
                    .await;
            }
            Err(error) => return Err(error),
        };
        self.upsert_loaded(entry, namespace, upserts, deletes, schema)
            .await
    }

    async fn upsert_loaded(
        &self,
        entry: Arc<NamespaceEntry>,
        namespace: &str,
        upserts: Vec<crate::Doc>,
        deletes: Vec<String>,
        schema: Schema,
    ) -> Result<WriteSummary> {
        let _operation = entry.operation.lock().await;
        ensure_open(&entry, namespace)?;
        let result = entry
            .namespace
            .write()
            .await
            .upsert(upserts, deletes, schema);
        result
    }

    pub async fn query(&self, namespace: &str, query: &Query) -> Result<Vec<QueryResult>> {
        self.query_with_ef_search(namespace, query, crate::namespace::DEFAULT_EF_SEARCH)
            .await
    }

    /// Query a namespace with an explicit HNSW traversal breadth.
    pub async fn query_with_ef_search(
        &self,
        namespace: &str,
        query: &Query,
        ef_search: usize,
    ) -> Result<Vec<QueryResult>> {
        let entry = self.get_or_open(namespace).await?;
        let loaded = entry.namespace.read().await;
        ensure_open(&entry, namespace)?;
        loaded.query_with_ef_search(query, ef_search)
    }

    pub async fn force_flush(&self, namespace: &str) -> Result<()> {
        let entry = self.get_or_open(namespace).await?;
        let _operation = entry.operation.lock().await;
        ensure_open(&entry, namespace)?;
        flush_and_compact(&entry).await
    }

    /// Inspect the most recent background flush failure for a loaded
    /// namespace. The namespace remains retryable after an error.
    pub async fn last_flush_error(&self, namespace: &str) -> Result<Option<String>> {
        let entry = self.get_or_open(namespace).await?;
        let _operation = entry.operation.lock().await;
        ensure_open(&entry, namespace)?;
        let error = entry
            .namespace
            .read()
            .await
            .last_flush_error()
            .map(str::to_owned);
        Ok(error)
    }

    /// List namespaces from manifests in the store, including cold namespaces.
    pub fn list_namespaces(&self) -> Result<Vec<String>> {
        let mut names = self
            .store
            .list("ns/")?
            .into_iter()
            .filter_map(|key| {
                key.strip_prefix("ns/")
                    .and_then(|key| key.strip_suffix("/MANIFEST.json"))
                    .filter(|name| !name.is_empty() && !name.contains('/'))
                    .map(str::to_owned)
            })
            .collect::<Vec<_>>();
        names.sort();
        names.dedup();
        Ok(names)
    }

    /// Delete every object under a namespace prefix and evict it from RAM.
    pub async fn delete_namespace(&self, namespace: &str) -> Result<()> {
        let namespace = validate_name(namespace)?;
        let mut registry = self.registry.lock().await;
        let entry = registry.entries.get(&namespace).cloned();
        if let Some(entry) = entry {
            entry.closing.store(true, Ordering::Release);
            let _operation = entry.operation.lock().await;
            entry.stop_worker().await;
            let keys = self.store.list(&format!("ns/{namespace}/"))?;
            if keys.is_empty() {
                entry.closing.store(false, Ordering::Release);
                return Err(Error::NotFound(format!("namespace not found: {namespace}")));
            }
            registry.entries.remove(&namespace);
            registry.lru.retain(|candidate| candidate != &namespace);
            for key in keys {
                self.store.delete(&key)?;
            }
            return Ok(());
        }

        // The registry lock prevents a concurrent cold open while this list
        // and delete sequence is in progress.
        let keys = self.store.list(&format!("ns/{namespace}/"))?;
        if keys.is_empty() {
            return Err(Error::NotFound(format!("namespace not found: {namespace}")));
        }
        for key in keys {
            self.store.delete(&key)?;
        }
        Ok(())
    }

    /// Drop one loaded namespace from the hot cache without touching storage.
    pub async fn evict(&self, namespace: &str) -> Result<()> {
        let namespace = validate_name(namespace)?;
        let mut registry = self.registry.lock().await;
        if let Some(entry) = registry.entries.remove(&namespace) {
            entry.closing.store(true, Ordering::Release);
            let _operation = entry.operation.lock().await;
            #[cfg(test)]
            if let Some(hook) = WORKER_RACE_HOOK.get() {
                hook.shutdown_operation_acquired.notify_one();
            }
            entry.stop_worker().await;
            registry.lru.retain(|candidate| candidate != &namespace);
        }
        Ok(())
    }

    pub async fn loaded_namespaces(&self) -> Result<Vec<String>> {
        let registry = self.registry.lock().await;
        let mut names = registry.entries.keys().cloned().collect::<Vec<_>>();
        names.sort();
        Ok(names)
    }

    async fn get_or_open(&self, name: &str) -> Result<Arc<NamespaceEntry>> {
        let name = validate_name(name)?;
        let mut registry = self.registry.lock().await;
        if let Some(entry) = registry.entries.get(&name).cloned() {
            touch_locked(&mut registry, &name);
            return Ok(entry);
        }

        let namespace = match Namespace::open(self.store.clone(), &name) {
            Ok(namespace) => namespace,
            Err(Error::NotFound(_)) => {
                return Err(Error::NotFound(format!("namespace not found: {name}")))
            }
            Err(error) => return Err(error),
        };
        let loaded = Arc::new(RwLock::new(namespace));
        let entry = Arc::new(NamespaceEntry {
            namespace: loaded,
            operation: Mutex::new(()),
            closing: AtomicBool::new(false),
            worker: StdMutex::new(None),
        });

        // Start before publication so cold replay thresholds cannot miss the
        // worker, and contention cannot permanently skip worker creation.
        publish_locked(&mut registry, &name, entry.clone()).await;
        Ok(entry)
    }

    async fn create_and_upsert(
        &self,
        namespace: &str,
        upserts: Vec<crate::Doc>,
        deletes: Vec<String>,
        schema: Schema,
    ) -> Result<WriteSummary> {
        let name = validate_name(namespace)?;
        let mut registry = self.registry.lock().await;
        if let Some(entry) = registry.entries.get(&name).cloned() {
            drop(registry);
            return self
                .upsert_loaded(entry, &name, upserts, deletes, schema)
                .await;
        }

        let mut namespace = match Namespace::open(self.store.clone(), &name) {
            Ok(namespace) => namespace,
            Err(Error::NotFound(_)) => Namespace::unpersisted(self.store.clone(), &name)?,
            Err(error) => return Err(error),
        };
        // No manifest or registry entry exists yet. Validation and the first
        // durable write happen before publication, so a rejected request
        // cannot leave a ghost namespace behind.
        let summary = namespace.upsert(upserts, deletes, schema)?;
        let loaded = Arc::new(RwLock::new(namespace));
        let entry = Arc::new(NamespaceEntry {
            namespace: loaded,
            operation: Mutex::new(()),
            closing: AtomicBool::new(false),
            worker: StdMutex::new(None),
        });
        publish_locked(&mut registry, &name, entry).await;
        Ok(summary)
    }
}

async fn flush_and_compact(entry: &Arc<NamespaceEntry>) -> Result<()> {
    {
        let mut namespace = entry.namespace.write().await;
        namespace.flush_only()?
    }

    let plan = {
        let namespace = entry.namespace.read().await;
        namespace.prepare_compaction()?
    };
    let Some(plan) = plan else {
        return Ok(());
    };

    let input_objects = {
        let mut namespace = entry.namespace.write().await;
        namespace.publish_compaction(plan)?
    };
    let cleanup_result = {
        let namespace = entry.namespace.read().await;
        namespace.finish_compaction(&input_objects)
    };
    if let Err(error) = cleanup_result {
        let mut namespace = entry.namespace.write().await;
        namespace.record_flush_error(&error);
        return Err(error);
    }
    Ok(())
}

impl NamespaceEntry {
    fn start_worker(self: &Arc<Self>) {
        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            return;
        };
        let Ok(namespace) = self.namespace.try_read() else {
            return;
        };
        let notify = namespace.flush_notify.clone();
        let initial_flush = namespace.should_flush();
        drop(namespace);
        let (cancel, mut cancelled) = watch::channel(false);
        let weak_entry: Weak<Self> = Arc::downgrade(self);
        let join = handle.spawn(async move {
            let mut initial = initial_flush;
            loop {
                if !initial {
                    tokio::select! {
                        changed = cancelled.changed() => {
                            if changed.is_err() || *cancelled.borrow() {
                                break;
                            }
                        }
                        _ = notify.notified() => {}
                    }
                } else {
                    initial = false;
                }
                let Some(entry) = weak_entry.upgrade() else {
                    break;
                };
                if entry.closing.load(Ordering::Acquire) {
                    break;
                }
                let should_flush = entry.namespace.read().await.should_flush();
                if should_flush {
                    #[cfg(test)]
                    if let Some(hook) = WORKER_RACE_HOOK.get() {
                        hook.before_operation.notify_one();
                        hook.release_before_operation.notified().await;
                    }
                    // Shutdown acquires this mutex before joining the worker;
                    // cancellation must win if the worker is queued here.
                    let _operation = tokio::select! {
                        operation = entry.operation.lock() => operation,
                        changed = cancelled.changed() => {
                            if changed.is_err() || *cancelled.borrow() {
                                break;
                            }
                            continue;
                        }
                    };
                    if entry.closing.load(Ordering::Acquire) {
                        break;
                    }
                    if let Err(error) = flush_and_compact(&entry).await {
                        let mut namespace = entry.namespace.write().await;
                        namespace.record_flush_error(&error);
                        eprintln!(
                            "namespace {} background flush failed: {error}",
                            namespace.name()
                        );
                    }
                }
            }
        });
        *self.worker.lock().expect("worker lock") = Some(Worker { cancel, join });
    }

    async fn stop_worker(&self) {
        let worker = self.worker.lock().expect("worker lock").take();
        let Some(worker) = worker else {
            return;
        };
        let _ = worker.cancel.send(true);
        let _ = worker.join.await;
    }
}

fn touch_locked(registry: &mut Registry, name: &str) {
    registry.lru.retain(|candidate| candidate != name);
    registry.lru.push_back(name.to_owned());
}

async fn publish_locked(registry: &mut Registry, name: &str, entry: Arc<NamespaceEntry>) {
    entry.start_worker();
    registry.entries.insert(name.to_owned(), entry);
    touch_locked(registry, name);
    if registry.lru.len() > HOT_CACHE_CAPACITY {
        if let Some(evicted_name) = registry.lru.pop_front() {
            if let Some(evicted) = registry.entries.remove(&evicted_name) {
                evicted.closing.store(true, Ordering::Release);
                let _operation = evicted.operation.lock().await;
                evicted.stop_worker().await;
            }
        }
    }
}

fn ensure_open(entry: &NamespaceEntry, name: &str) -> Result<()> {
    if entry.closing.load(Ordering::Acquire) {
        Err(Error::NotFound(format!("namespace not found: {name}")))
    } else {
        Ok(())
    }
}

fn validate_name(name: &str) -> Result<String> {
    if name.is_empty() || name == "." || name == ".." || name.contains('/') || name.contains('\\') {
        return Err(Error::InvalidKey(format!("invalid namespace: {name}")));
    }
    Ok(name.to_owned())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::time::Duration;

    use tempfile::tempdir;

    use super::{Engine, WorkerRaceHook, WORKER_RACE_HOOK};
    use crate::Doc;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn evict_cancels_worker_waiting_for_operation_lock() {
        let hook = std::sync::Arc::new(WorkerRaceHook {
            before_operation: tokio::sync::Notify::new(),
            release_before_operation: tokio::sync::Notify::new(),
            shutdown_operation_acquired: tokio::sync::Notify::new(),
        });
        assert!(WORKER_RACE_HOOK.set(std::sync::Arc::clone(&hook)).is_ok());

        let root = tempdir().expect("tempdir");
        let engine = std::sync::Arc::new(Engine::new(root.path()).expect("engine"));
        let documents = (0..1_000)
            .map(|index| Doc {
                id: format!("doc-{index}"),
                vector: None,
                attributes: BTreeMap::new(),
            })
            .collect();
        engine
            .upsert("race", documents, Vec::new(), BTreeMap::new())
            .await
            .expect("upsert");

        // The worker has passed its closing check and is paused immediately
        // before waiting for the operation mutex.
        hook.before_operation.notified().await;
        let evict_engine = std::sync::Arc::clone(&engine);
        let evict = tokio::spawn(async move { evict_engine.evict("race").await });
        hook.shutdown_operation_acquired.notified().await;
        hook.release_before_operation.notify_one();

        let result = tokio::time::timeout(Duration::from_secs(1), evict)
            .await
            .expect("eviction did not deadlock")
            .expect("eviction task panicked");
        assert!(result.is_ok(), "eviction failed: {result:?}");
    }
}

//! Namespace lifecycle coordination and bounded hot-cache admission.

use std::collections::{HashMap, VecDeque};
use std::env;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex as StdMutex, Weak};

use tokio::sync::{watch, Mutex, Notify, OwnedSemaphorePermit, RwLock, Semaphore};
use tokio::task::JoinHandle;

#[cfg(test)]
use std::sync::OnceLock;

use crate::namespace::{Namespace, Query, QueryResult, Schema, SharedStore, WriteSummary};
use crate::{Error, Result};

pub(crate) const HOT_CACHE_CAPACITY: usize = 8;
pub(crate) const COLD_LOAD_CONCURRENCY: usize = 4;
pub(crate) const MEMORY_BUDGET_ENV: &str = "PUFFERCLONE_MEMORY_BUDGET_BYTES";

type Loaded = Arc<RwLock<Namespace>>;

struct Worker {
    cancel: watch::Sender<bool>,
    join: JoinHandle<()>,
}

struct NamespaceEntry {
    namespace: Loaded,
    operation: Mutex<()>,
    closing: AtomicBool,
    memory_bytes: AtomicUsize,
    worker: StdMutex<Option<Worker>>,
}

#[derive(Debug, Clone)]
enum LoadFailure {
    Io(String),
    Json(String),
    Bincode(String),
    InvalidKey(String),
    AlreadyExists(String),
    NotFound(String),
    Store(String),
    Validation(String),
}

impl LoadFailure {
    fn into_error(self) -> Error {
        match self {
            Self::Io(message) => Error::Io(std::io::Error::other(message)),
            Self::Json(message) => {
                Error::Json(serde_json::Error::io(std::io::Error::other(message)))
            }
            Self::Bincode(message) => Error::Bincode(Box::new(bincode::ErrorKind::Custom(message))),
            Self::InvalidKey(message) => Error::InvalidKey(message),
            Self::AlreadyExists(message) => Error::AlreadyExists(message),
            Self::NotFound(message) => Error::NotFound(message),
            Self::Store(message) => Error::Store(message),
            Self::Validation(message) => Error::Validation(message),
        }
    }
}

#[derive(Clone)]
struct LoadOutcome {
    entry: Option<Arc<NamespaceEntry>>,
    summary: Option<WriteSummary>,
    error: Option<LoadFailure>,
}

struct LoadingSlot {
    outcome: Mutex<Option<LoadOutcome>>,
    notify: Notify,
    cancelled: AtomicBool,
    cancel_notify: Notify,
    #[cfg(test)]
    admission_waiting: Notify,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum LifecycleKind {
    Delete,
    Evict,
}

struct LifecycleTransition {
    kind: LifecycleKind,
    outcome: Mutex<Option<std::result::Result<(), LoadFailure>>>,
    notify: Notify,
}

impl LifecycleTransition {
    fn new(kind: LifecycleKind) -> Self {
        Self {
            kind,
            outcome: Mutex::new(None),
            notify: Notify::new(),
        }
    }

    async fn complete(&self, outcome: std::result::Result<(), LoadFailure>) {
        *self.outcome.lock().await = Some(outcome);
        self.notify.notify_waiters();
    }

    async fn wait(&self) -> Result<()> {
        loop {
            let notified = self.notify.notified();
            if let Some(outcome) = self.outcome.lock().await.clone() {
                return outcome.map_err(LoadFailure::into_error);
            }
            notified.await;
        }
    }
}

impl LoadingSlot {
    fn new() -> Self {
        Self {
            outcome: Mutex::new(None),
            notify: Notify::new(),
            cancelled: AtomicBool::new(false),
            cancel_notify: Notify::new(),
            #[cfg(test)]
            admission_waiting: Notify::new(),
        }
    }

    fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
        self.cancel_notify.notify_waiters();
    }

    async fn wait_cancelled(&self) {
        loop {
            let notified = self.cancel_notify.notified();
            if self.cancelled.load(Ordering::Acquire) {
                return;
            }
            notified.await;
        }
    }

    async fn complete(&self, outcome: LoadOutcome) {
        *self.outcome.lock().await = Some(outcome);
        self.notify.notify_waiters();
    }

    async fn wait(&self) -> Result<(Arc<NamespaceEntry>, Option<WriteSummary>)> {
        loop {
            let notified = self.notify.notified();
            if let Some(outcome) = self.outcome.lock().await.clone() {
                let Some(entry) = outcome.entry else {
                    return Err(outcome
                        .error
                        .expect("load outcome without entry or error")
                        .into_error());
                };
                return Ok((entry, outcome.summary));
            }
            notified.await;
        }
    }
}

enum OpenAction {
    Existing(Arc<NamespaceEntry>),
    Wait(Arc<LoadingSlot>),
    Start(Arc<LoadingSlot>),
    Transition(Arc<LifecycleTransition>),
}

enum CreateAction {
    Existing(Arc<NamespaceEntry>),
    Wait(Arc<LoadingSlot>),
    Start(Arc<LoadingSlot>),
    Transition(Arc<LifecycleTransition>),
}

struct LifecycleAction {
    transition: Arc<LifecycleTransition>,
    entry: Option<Arc<NamespaceEntry>>,
    loading: Option<Arc<LoadingSlot>>,
}

struct LoadStartGuard {
    registry: Arc<Mutex<Registry>>,
    name: String,
    loading: Arc<LoadingSlot>,
    armed: bool,
}

#[derive(Clone)]
struct ColdLoadContext {
    registry: Arc<Mutex<Registry>>,
    store: SharedStore,
    admission: Arc<Semaphore>,
}

impl LoadStartGuard {
    fn new(registry: Arc<Mutex<Registry>>, name: String, loading: Arc<LoadingSlot>) -> Self {
        Self {
            registry,
            name,
            loading,
            armed: true,
        }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for LoadStartGuard {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let registry = self.registry.clone();
        let name = self.name.clone();
        let loading = self.loading.clone();
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(finish_load(
                registry,
                name,
                loading,
                Err(Error::NotFound("cold load caller cancelled".to_owned())),
            ));
        }
    }
}

#[cfg(test)]
struct WorkerRaceHook {
    before_operation: Notify,
    release_before_operation: Notify,
    shutdown_operation_acquired: Notify,
}

#[cfg(test)]
static WORKER_RACE_HOOK: OnceLock<Arc<WorkerRaceHook>> = OnceLock::new();

#[cfg(test)]
struct AdmissionWaitHook {
    loading: Arc<LoadingSlot>,
    entered: Notify,
}

#[cfg(test)]
static ADMISSION_WAIT_HOOK: OnceLock<Arc<AdmissionWaitHook>> = OnceLock::new();

#[cfg(test)]
struct LoadFinalizeHook {
    loading: Arc<LoadingSlot>,
    gate_next: AtomicBool,
    entered: Notify,
    release: Notify,
}

#[cfg(test)]
static LOAD_FINALIZE_HOOK: OnceLock<Arc<LoadFinalizeHook>> = OnceLock::new();

struct Registry {
    entries: HashMap<String, Arc<NamespaceEntry>>,
    loading: HashMap<String, Arc<LoadingSlot>>,
    transitions: HashMap<String, Arc<LifecycleTransition>>,
    lru: VecDeque<String>,
    total_memory_bytes: usize,
    memory_budget: Option<usize>,
}

/// Coordinates namespace loads, lifecycle transitions, and the hot cache.
pub(crate) struct Coordinator {
    store: SharedStore,
    registry: Arc<Mutex<Registry>>,
    cold_load_admission: Arc<Semaphore>,
    memory_budget: Option<usize>,
}

impl Coordinator {
    #[cfg(test)]
    pub(crate) fn new(store: SharedStore) -> Self {
        Self::with_memory_budget(store, None)
    }

    pub(crate) fn from_env(store: SharedStore) -> Result<Self> {
        Ok(Self::with_memory_budget(store, memory_budget_from_env()?))
    }

    fn with_memory_budget(store: SharedStore, memory_budget: Option<usize>) -> Self {
        Self {
            store,
            registry: Arc::new(Mutex::new(Registry {
                entries: HashMap::new(),
                loading: HashMap::new(),
                transitions: HashMap::new(),
                lru: VecDeque::new(),
                total_memory_bytes: 0,
                memory_budget,
            })),
            cold_load_admission: Arc::new(Semaphore::new(COLD_LOAD_CONCURRENCY)),
            memory_budget,
        }
    }

    pub(crate) fn hot_cache_capacity() -> usize {
        HOT_CACHE_CAPACITY
    }

    pub(crate) fn cold_load_concurrency() -> usize {
        COLD_LOAD_CONCURRENCY
    }

    /// Upsert documents and/or delete IDs, creating the namespace on demand.
    pub(crate) async fn upsert(
        &self,
        namespace: &str,
        upserts: Vec<crate::Doc>,
        deletes: Vec<String>,
        schema: Schema,
    ) -> Result<WriteSummary> {
        loop {
            let entry = match self.get_or_open(namespace).await {
                Ok(entry) => entry,
                Err(Error::NotFound(_)) => {
                    return self
                        .create_and_upsert(namespace, upserts, deletes, schema)
                        .await;
                }
                Err(error) => return Err(error),
            };
            match self
                .upsert_loaded(
                    entry,
                    namespace,
                    upserts.clone(),
                    deletes.clone(),
                    schema.clone(),
                )
                .await
            {
                Err(Error::NotFound(_)) => continue,
                result => return result,
            }
        }
    }

    async fn upsert_loaded(
        &self,
        entry: Arc<NamespaceEntry>,
        namespace: &str,
        upserts: Vec<crate::Doc>,
        deletes: Vec<String>,
        schema: Schema,
    ) -> Result<WriteSummary> {
        let result = {
            let _operation = entry.operation.lock().await;
            ensure_open(&entry, namespace)?;
            let result = entry
                .namespace
                .write()
                .await
                .upsert(upserts, deletes, schema);
            if result.is_ok() && self.memory_budget.is_some() {
                refresh_memory_and_schedule_evictions(
                    self.registry.clone(),
                    namespace.to_owned(),
                    entry.clone(),
                )
                .await;
            }
            result
        };
        result
    }

    pub(crate) async fn query(&self, namespace: &str, query: &Query) -> Result<Vec<QueryResult>> {
        self.query_with_ef_search(namespace, query, crate::namespace::DEFAULT_EF_SEARCH)
            .await
    }

    /// Query a namespace with an explicit HNSW traversal breadth.
    pub(crate) async fn query_with_ef_search(
        &self,
        namespace: &str,
        query: &Query,
        ef_search: usize,
    ) -> Result<Vec<QueryResult>> {
        loop {
            let entry = self.get_or_open(namespace).await?;
            let loaded = entry.namespace.read().await;
            if entry.closing.load(Ordering::Acquire) {
                continue;
            }
            return loaded.query_with_ef_search(query, ef_search);
        }
    }

    pub(crate) async fn force_flush(&self, namespace: &str) -> Result<()> {
        let entry = self.get_or_open(namespace).await?;
        let result = {
            let _operation = entry.operation.lock().await;
            ensure_open(&entry, namespace)?;
            let result = flush_and_compact(&entry).await;
            if self.memory_budget.is_some() {
                refresh_memory_and_schedule_evictions(
                    self.registry.clone(),
                    namespace.to_owned(),
                    entry.clone(),
                )
                .await;
            }
            result
        };
        result
    }

    /// Inspect the most recent background flush failure for a loaded
    /// namespace. The namespace remains retryable after an error.
    pub(crate) async fn last_flush_error(&self, namespace: &str) -> Result<Option<String>> {
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
    pub(crate) fn list_namespaces(&self) -> Result<Vec<String>> {
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
    pub(crate) async fn delete_namespace(&self, namespace: &str) -> Result<()> {
        let namespace = validate_name(namespace)?;
        loop {
            let (action, joined) = {
                let mut registry = self.registry.lock().await;
                if let Some(transition) = registry.transitions.get(&namespace).cloned() {
                    (None, Some(transition))
                } else {
                    let transition = Arc::new(LifecycleTransition::new(LifecycleKind::Delete));
                    registry
                        .transitions
                        .insert(namespace.clone(), transition.clone());
                    let entry = registry.entries.get(&namespace).cloned();
                    if let Some(entry) = &entry {
                        entry.closing.store(true, Ordering::Release);
                        registry.lru.retain(|candidate| candidate != &namespace);
                    }
                    let loading = registry.loading.get(&namespace).cloned();
                    (
                        Some(LifecycleAction {
                            transition,
                            entry,
                            loading,
                        }),
                        None,
                    )
                }
            };

            if let Some(action) = action {
                let transition = action.transition.clone();
                start_delete_transition(
                    self.registry.clone(),
                    self.store.clone(),
                    namespace.clone(),
                    action,
                );
                // The actual transition is detached. Cancelling this wait
                // cannot strand a registered entry or lifecycle marker.
                return transition.wait().await;
            }

            let transition = joined.expect("delete action had neither start nor join");
            let result = transition.wait().await;
            if transition.kind == LifecycleKind::Delete {
                return result;
            }
            result?;
        }
    }

    /// Drop one loaded namespace from the hot cache without touching storage.
    pub(crate) async fn evict(&self, namespace: &str) -> Result<()> {
        let namespace = validate_name(namespace)?;
        let (action, joined) = {
            let mut registry = self.registry.lock().await;
            if let Some(transition) = registry.transitions.get(&namespace).cloned() {
                (None, Some(transition))
            } else {
                let entry = registry.entries.get(&namespace).cloned();
                let loading = registry.loading.get(&namespace).cloned();
                if entry.is_none() && loading.is_none() {
                    return Ok(());
                }
                let transition = Arc::new(LifecycleTransition::new(LifecycleKind::Evict));
                registry
                    .transitions
                    .insert(namespace.clone(), transition.clone());
                if let Some(entry) = &entry {
                    entry.closing.store(true, Ordering::Release);
                    registry.lru.retain(|candidate| candidate != &namespace);
                }
                if let Some(loading) = &loading {
                    loading.cancel();
                }
                (
                    Some(LifecycleAction {
                        transition,
                        entry,
                        loading,
                    }),
                    None,
                )
            }
        };

        if let Some(action) = action {
            let transition = action.transition.clone();
            start_evict_transition(self.registry.clone(), namespace, action);
            return transition.wait().await;
        }
        joined
            .expect("eviction action had neither start nor join")
            .wait()
            .await
    }

    pub(crate) async fn loaded_namespaces(&self) -> Result<Vec<String>> {
        let registry = self.registry.lock().await;
        let mut names = registry.entries.keys().cloned().collect::<Vec<_>>();
        names.sort();
        Ok(names)
    }

    pub(crate) async fn loaded_memory_bytes(&self) -> usize {
        self.registry.lock().await.total_memory_bytes
    }

    async fn get_or_open(&self, name: &str) -> Result<Arc<NamespaceEntry>> {
        let name = validate_name(name)?;
        loop {
            let action = {
                let mut registry = self.registry.lock().await;
                if let Some(transition) = registry.transitions.get(&name).cloned() {
                    OpenAction::Transition(transition)
                } else if let Some(entry) = registry.entries.get(&name).cloned() {
                    touch_locked(&mut registry, &name);
                    OpenAction::Existing(entry)
                } else if let Some(loading) = registry.loading.get(&name).cloned() {
                    OpenAction::Wait(loading)
                } else {
                    let loading = Arc::new(LoadingSlot::new());
                    registry.loading.insert(name.clone(), loading.clone());
                    OpenAction::Start(loading)
                }
            };
            match action {
                OpenAction::Existing(entry) => return Ok(entry),
                OpenAction::Wait(loading) => return loading.wait().await.map(|(entry, _)| entry),
                OpenAction::Start(loading) => {
                    start_open_load(
                        ColdLoadContext {
                            registry: self.registry.clone(),
                            store: self.store.clone(),
                            admission: self.cold_load_admission.clone(),
                        },
                        name.clone(),
                        loading.clone(),
                    )
                    .await;
                    return loading.wait().await.map(|(entry, _)| entry);
                }
                OpenAction::Transition(transition) => {
                    let _ = transition.wait().await;
                }
            }
        }
    }

    async fn create_and_upsert(
        &self,
        namespace: &str,
        upserts: Vec<crate::Doc>,
        deletes: Vec<String>,
        schema: Schema,
    ) -> Result<WriteSummary> {
        let name = validate_name(namespace)?;
        loop {
            let action = {
                let mut registry = self.registry.lock().await;
                if let Some(transition) = registry.transitions.get(&name).cloned() {
                    CreateAction::Transition(transition)
                } else if let Some(entry) = registry.entries.get(&name).cloned() {
                    touch_locked(&mut registry, &name);
                    CreateAction::Existing(entry)
                } else if let Some(loading) = registry.loading.get(&name).cloned() {
                    CreateAction::Wait(loading)
                } else {
                    let loading = Arc::new(LoadingSlot::new());
                    registry.loading.insert(name.clone(), loading.clone());
                    CreateAction::Start(loading)
                }
            };

            match action {
                CreateAction::Existing(entry) => {
                    return self
                        .upsert_loaded(entry, &name, upserts, deletes, schema)
                        .await;
                }
                CreateAction::Wait(loading) => match loading.wait().await {
                    Ok((entry, _)) => match self
                        .upsert_loaded(
                            entry,
                            &name,
                            upserts.clone(),
                            deletes.clone(),
                            schema.clone(),
                        )
                        .await
                    {
                        Ok(summary) => return Ok(summary),
                        Err(Error::NotFound(_)) => continue,
                        Err(error) => return Err(error),
                    },
                    Err(Error::NotFound(_)) => continue,
                    Err(error) => return Err(error),
                },
                CreateAction::Transition(transition) => {
                    let _ = transition.wait().await;
                    continue;
                }
                CreateAction::Start(loading) => {
                    start_create_load(
                        ColdLoadContext {
                            registry: self.registry.clone(),
                            store: self.store.clone(),
                            admission: self.cold_load_admission.clone(),
                        },
                        name.clone(),
                        loading.clone(),
                        upserts,
                        deletes,
                        schema,
                    )
                    .await;
                    let (_, summary) = loading.wait().await?;
                    return summary.ok_or_else(|| {
                        Error::Store(
                            "namespace creation completed without a write summary".to_owned(),
                        )
                    });
                }
            }
        }
    }
}

async fn start_open_load(context: ColdLoadContext, name: String, loading: Arc<LoadingSlot>) {
    let ColdLoadContext {
        registry,
        store,
        admission,
    } = context;
    let mut guard = LoadStartGuard::new(registry.clone(), name.clone(), loading.clone());
    let permit = match acquire_admission(admission, &loading).await {
        Ok(permit) => permit,
        Err(error) => {
            spawn_finish_load(registry, name, loading, Err(error));
            guard.disarm();
            return;
        }
    };
    let load_name = name.clone();
    tokio::spawn(async move {
        let result = match tokio::task::spawn_blocking(move || {
            let _permit = permit;
            Namespace::open(store, load_name)
        })
        .await
        {
            Ok(result) => result.map(|namespace| (namespace, None)),
            Err(error) => Err(Error::Store(format!("namespace load task failed: {error}"))),
        };
        finish_load(registry, name, loading, result).await;
    });
    guard.disarm();
}

async fn start_create_load(
    context: ColdLoadContext,
    name: String,
    loading: Arc<LoadingSlot>,
    upserts: Vec<crate::Doc>,
    deletes: Vec<String>,
    schema: Schema,
) {
    let ColdLoadContext {
        registry,
        store,
        admission,
    } = context;
    let mut guard = LoadStartGuard::new(registry.clone(), name.clone(), loading.clone());
    let permit = match acquire_admission(admission, &loading).await {
        Ok(permit) => permit,
        Err(error) => {
            spawn_finish_load(registry, name, loading, Err(error));
            guard.disarm();
            return;
        }
    };
    let load_name = name.clone();
    tokio::spawn(async move {
        let result = match tokio::task::spawn_blocking(move || {
            let _permit = permit;
            let mut namespace = match Namespace::open(store.clone(), &load_name) {
                Ok(namespace) => namespace,
                Err(Error::NotFound(_)) => Namespace::unpersisted(store, load_name)?,
                Err(error) => return Err(error),
            };
            let summary = namespace.upsert(upserts, deletes, schema)?;
            Ok((namespace, Some(summary)))
        })
        .await
        {
            Ok(result) => result,
            Err(error) => Err(Error::Store(format!(
                "namespace creation task failed: {error}"
            ))),
        };
        finish_load(registry, name, loading, result).await;
    });
    guard.disarm();
}

fn spawn_finish_load(
    registry: Arc<Mutex<Registry>>,
    name: String,
    loading: Arc<LoadingSlot>,
    result: Result<(Namespace, Option<WriteSummary>)>,
) {
    tokio::spawn(finish_load(registry, name, loading, result));
}

async fn acquire_admission(
    admission: Arc<Semaphore>,
    loading: &Arc<LoadingSlot>,
) -> Result<OwnedSemaphorePermit> {
    #[cfg(test)]
    loading.admission_waiting.notify_waiters();
    #[cfg(test)]
    if let Some(hook) = ADMISSION_WAIT_HOOK
        .get()
        .filter(|hook| Arc::ptr_eq(&hook.loading, loading))
    {
        hook.entered.notify_one();
    }
    tokio::select! {
        biased;
        _ = loading.wait_cancelled() => Err(Error::NotFound("cold load cancelled".to_owned())),
        permit = admission.acquire_owned() => permit
            .map_err(|error| Error::Store(format!("cold-load admission closed: {error}"))),
    }
}

fn start_delete_transition(
    registry: Arc<Mutex<Registry>>,
    store: SharedStore,
    name: String,
    action: LifecycleAction,
) {
    tokio::spawn(async move {
        let LifecycleAction {
            transition,
            entry: initial_entry,
            loading,
        } = action;
        if let Some(loading) = &loading {
            let _ = loading.wait().await;
        }
        let entry = {
            let registry = registry.lock().await;
            registry.entries.get(&name).cloned().or(initial_entry)
        };
        if let Some(entry) = &entry {
            drain_entry(entry).await;
        }

        let result = (|| {
            let keys = store.list(&format!("ns/{name}/"))?;
            if keys.is_empty() {
                return Err(Error::NotFound(format!("namespace not found: {name}")));
            }
            for key in keys {
                store.delete(&key)?;
            }
            Ok(())
        })();
        complete_transition(registry, name, transition, entry, loading, result).await;
    });
}

fn start_evict_transition(registry: Arc<Mutex<Registry>>, name: String, action: LifecycleAction) {
    tokio::spawn(async move {
        let LifecycleAction {
            transition,
            entry: initial_entry,
            loading,
        } = action;
        if let Some(loading) = &loading {
            let _ = loading.wait().await;
        }
        let entry = {
            let registry = registry.lock().await;
            registry.entries.get(&name).cloned().or(initial_entry)
        };
        if let Some(entry) = &entry {
            drain_entry(entry).await;
        }
        complete_transition(registry, name, transition, entry, loading, Ok(())).await;
    });
}

async fn drain_entry(entry: &NamespaceEntry) {
    let _operation = entry.operation.lock().await;
    #[cfg(test)]
    if let Some(hook) = WORKER_RACE_HOOK.get() {
        hook.shutdown_operation_acquired.notify_one();
    }
    entry.stop_worker().await;
}

async fn complete_transition(
    registry: Arc<Mutex<Registry>>,
    name: String,
    transition: Arc<LifecycleTransition>,
    entry: Option<Arc<NamespaceEntry>>,
    loading: Option<Arc<LoadingSlot>>,
    result: Result<()>,
) {
    let outcome = result.map_err(|error| load_failure(&name, error));
    {
        let mut registry = registry.lock().await;
        if let Some(current) = registry.entries.get(&name).cloned() {
            if entry
                .as_ref()
                .is_none_or(|expected| Arc::ptr_eq(expected, &current))
            {
                registry.entries.remove(&name);
                registry.lru.retain(|candidate| candidate != &name);
                // Keep draining bytes in the global ledger until the
                // operation lock and worker have both been drained.
                let bytes = current.memory_bytes.swap(0, Ordering::AcqRel);
                registry.total_memory_bytes = registry.total_memory_bytes.saturating_sub(bytes);
            }
        }
        if loading.as_ref().is_some_and(|loading| {
            registry
                .loading
                .get(&name)
                .is_some_and(|candidate| Arc::ptr_eq(candidate, loading))
        }) {
            registry.loading.remove(&name);
        }
    }

    // Publish before removing the transition. A new caller that arrives in
    // this interval joins the completed lifecycle rather than starting a
    // replacement while cleanup is still being retired from the registry.
    transition.complete(outcome).await;
    let mut registry = registry.lock().await;
    if registry
        .transitions
        .get(&name)
        .is_some_and(|candidate| Arc::ptr_eq(candidate, &transition))
    {
        registry.transitions.remove(&name);
    }
}

async fn finish_load(
    registry: Arc<Mutex<Registry>>,
    name: String,
    loading: Arc<LoadingSlot>,
    result: Result<(Namespace, Option<WriteSummary>)>,
) {
    let registry_handle = registry.clone();
    let (outcome, evicted) = {
        let mut registry = registry.lock().await;
        let active = registry
            .loading
            .get(&name)
            .is_some_and(|candidate| Arc::ptr_eq(candidate, &loading));
        let cancelled = loading.cancelled.load(Ordering::Acquire);

        if active && !cancelled {
            match result {
                Ok((namespace, summary)) => {
                    let accounting_enabled = registry.memory_budget.is_some();
                    let memory_bytes = if accounting_enabled {
                        namespace.memory_bytes()
                    } else {
                        0
                    };
                    let closing = registry
                        .transitions
                        .get(&name)
                        .is_some_and(|transition| transition.kind == LifecycleKind::Delete);
                    let entry = Arc::new(NamespaceEntry {
                        namespace: Arc::new(RwLock::new(namespace)),
                        operation: Mutex::new(()),
                        closing: AtomicBool::new(closing),
                        memory_bytes: AtomicUsize::new(memory_bytes),
                        worker: StdMutex::new(None),
                    });
                    entry.start_worker(
                        Arc::downgrade(&registry_handle),
                        name.clone(),
                        accounting_enabled,
                    );
                    registry.entries.insert(name.clone(), entry.clone());
                    registry.total_memory_bytes =
                        registry.total_memory_bytes.saturating_add(memory_bytes);
                    touch_locked(&mut registry, &name);

                    let evicted = policy_evictions_locked(&mut registry, &name);
                    (
                        LoadOutcome {
                            entry: Some(entry),
                            summary,
                            error: None,
                        },
                        evicted,
                    )
                }
                Err(error) => (
                    LoadOutcome {
                        entry: None,
                        summary: None,
                        error: Some(load_failure(&name, error)),
                    },
                    Vec::new(),
                ),
            }
        } else {
            (
                LoadOutcome {
                    entry: None,
                    summary: None,
                    error: Some(LoadFailure::NotFound(format!(
                        "namespace not found: {name}"
                    ))),
                },
                Vec::new(),
            )
        }
    };

    #[cfg(test)]
    if loading.cancelled.load(Ordering::Acquire) {
        if let Some(hook) = LOAD_FINALIZE_HOOK
            .get()
            .filter(|hook| Arc::ptr_eq(&hook.loading, &loading))
        {
            if hook.gate_next.swap(false, Ordering::AcqRel) {
                hook.entered.notify_one();
                hook.release.notified().await;
            }
        }
    }
    loading.complete(outcome).await;
    {
        let mut registry = registry.lock().await;
        if registry
            .loading
            .get(&name)
            .is_some_and(|candidate| Arc::ptr_eq(candidate, &loading))
        {
            registry.loading.remove(&name);
        }
    }
    for (evicted_name, action) in evicted {
        start_evict_transition(registry.clone(), evicted_name, action);
    }
}

fn begin_evict_locked(registry: &mut Registry, name: &str) -> Option<(String, LifecycleAction)> {
    if registry.transitions.contains_key(name) {
        return None;
    }
    let entry = registry.entries.get(name).cloned()?;
    if entry.closing.load(Ordering::Acquire) {
        return None;
    }
    let transition = Arc::new(LifecycleTransition::new(LifecycleKind::Evict));
    registry
        .transitions
        .insert(name.to_owned(), transition.clone());
    entry.closing.store(true, Ordering::Release);
    registry.lru.retain(|candidate| candidate != name);
    // Do not release this entry's bytes until complete_transition, after the
    // coordinator has drained its in-flight operation and worker.
    let loading = registry.loading.get(name).cloned();
    if let Some(loading) = &loading {
        loading.cancel();
    }
    Some((
        name.to_owned(),
        LifecycleAction {
            transition,
            entry: Some(entry),
            loading,
        },
    ))
}

fn policy_evictions_locked(
    registry: &mut Registry,
    touched: &str,
) -> Vec<(String, LifecycleAction)> {
    let mut evicted = Vec::new();
    loop {
        let over_capacity = registry.lru.len() > HOT_CACHE_CAPACITY;
        let projected_memory_bytes = registry
            .total_memory_bytes
            .saturating_sub(draining_memory_bytes(registry));
        let over_budget = registry
            .memory_budget
            .is_some_and(|budget| projected_memory_bytes > budget);
        if !over_capacity && !over_budget {
            break;
        }

        let candidate = registry
            .lru
            .iter()
            .find(|name| {
                name.as_str() != touched
                    && registry
                        .entries
                        .get(*name)
                        .is_some_and(|entry| !entry.closing.load(Ordering::Acquire))
            })
            .cloned();
        let Some(candidate) = candidate else {
            break;
        };
        let Some(action) = begin_evict_locked(registry, &candidate) else {
            registry.lru.retain(|name| name != &candidate);
            continue;
        };
        evicted.push(action);
    }
    evicted
}

/// The global ledger retains bytes until a transition has drained the
/// operation lock and worker. Policy decisions should nevertheless project
/// those already-registered drains out of the snapshot, or one policy pass
/// would schedule every older namespace instead of only the bytes needed to
/// get back under budget.
fn draining_memory_bytes(registry: &Registry) -> usize {
    registry
        .transitions
        .keys()
        .filter_map(|name| {
            registry
                .entries
                .get(name)
                .map(|entry| entry.memory_bytes.load(Ordering::Acquire))
        })
        .fold(0usize, usize::saturating_add)
}

async fn refresh_memory_and_schedule_evictions(
    registry: Arc<Mutex<Registry>>,
    name: String,
    entry: Arc<NamespaceEntry>,
) {
    // Every caller holds entry.operation while sampling and committing this
    // estimate. That keeps the ledger update in the same order as the
    // namespace mutation that produced it.
    let memory_bytes = entry.namespace.read().await.memory_bytes();
    let evicted = {
        let mut registry = registry.lock().await;
        let current = registry
            .entries
            .get(&name)
            .is_some_and(|candidate| Arc::ptr_eq(candidate, &entry));
        if !current || entry.closing.load(Ordering::Acquire) {
            Vec::new()
        } else {
            let previous = entry.memory_bytes.swap(memory_bytes, Ordering::AcqRel);
            if memory_bytes >= previous {
                registry.total_memory_bytes = registry
                    .total_memory_bytes
                    .saturating_add(memory_bytes - previous);
            } else {
                registry.total_memory_bytes = registry
                    .total_memory_bytes
                    .saturating_sub(previous - memory_bytes);
            }
            policy_evictions_locked(&mut registry, &name)
        }
    };
    for (evicted_name, action) in evicted {
        start_evict_transition(registry.clone(), evicted_name, action);
    }
}

fn load_failure(name: &str, error: Error) -> LoadFailure {
    match error {
        Error::Io(error) => LoadFailure::Io(error.to_string()),
        Error::Json(error) => LoadFailure::Json(error.to_string()),
        Error::Bincode(error) => LoadFailure::Bincode(error.to_string()),
        Error::InvalidKey(message) => LoadFailure::InvalidKey(message),
        Error::AlreadyExists(message) => LoadFailure::AlreadyExists(message),
        Error::NotFound(_) => LoadFailure::NotFound(format!("namespace not found: {name}")),
        Error::Store(message) => LoadFailure::Store(message),
        Error::Validation(message) => LoadFailure::Validation(message),
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
    fn start_worker(
        self: &Arc<Self>,
        registry: Weak<Mutex<Registry>>,
        name: String,
        accounting_enabled: bool,
    ) {
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
                    let flush_result = flush_and_compact(&entry).await;
                    if accounting_enabled {
                        let Some(registry) = registry.upgrade() else {
                            break;
                        };
                        refresh_memory_and_schedule_evictions(
                            registry,
                            name.clone(),
                            entry.clone(),
                        )
                        .await;
                    }
                    drop(_operation);
                    if let Err(error) = flush_result {
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

fn memory_budget_from_env() -> Result<Option<usize>> {
    let Ok(value) = env::var(MEMORY_BUDGET_ENV) else {
        return Ok(None);
    };
    let budget = value.parse::<usize>().map_err(|_| {
        Error::Validation(format!(
            "{MEMORY_BUDGET_ENV} must be a positive integer number of bytes"
        ))
    })?;
    if budget == 0 {
        return Err(Error::Validation(format!(
            "{MEMORY_BUDGET_ENV} must be greater than zero"
        )));
    }
    Ok(Some(budget))
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

    use super::{
        start_evict_transition, start_open_load, AdmissionWaitHook, ColdLoadContext, Coordinator,
        LoadFinalizeHook, LoadingSlot, WorkerRaceHook, ADMISSION_WAIT_HOOK, COLD_LOAD_CONCURRENCY,
        HOT_CACHE_CAPACITY, LOAD_FINALIZE_HOOK, WORKER_RACE_HOOK,
    };
    use crate::engine::Engine;
    use crate::namespace::{Query, SharedStore};
    use crate::store::{LocalDirStore, ObjectStore};
    use crate::Doc;

    struct LoadGate {
        block_next: std::sync::atomic::AtomicBool,
        entered: std::sync::Barrier,
        release: std::sync::Barrier,
    }

    struct GatedStore {
        inner: LocalDirStore,
        gate: std::sync::Arc<LoadGate>,
    }

    struct AdmissionStore {
        inner: LocalDirStore,
        enabled: std::sync::atomic::AtomicBool,
        block_remaining: std::sync::atomic::AtomicUsize,
        active: std::sync::atomic::AtomicUsize,
        max_active: std::sync::atomic::AtomicUsize,
        entered: std::sync::Arc<std::sync::Barrier>,
        released: std::sync::Arc<std::sync::Barrier>,
    }

    impl AdmissionStore {
        fn set_enabled(&self) {
            self.enabled
                .store(true, std::sync::atomic::Ordering::Release);
        }

        fn update_max(&self, active: usize) {
            let mut current = self.max_active.load(std::sync::atomic::Ordering::Acquire);
            while active > current {
                match self.max_active.compare_exchange(
                    current,
                    active,
                    std::sync::atomic::Ordering::AcqRel,
                    std::sync::atomic::Ordering::Acquire,
                ) {
                    Ok(_) => break,
                    Err(observed) => current = observed,
                }
            }
        }

        fn take_block(&self) -> bool {
            let mut remaining = self
                .block_remaining
                .load(std::sync::atomic::Ordering::Acquire);
            loop {
                if remaining == 0 {
                    return false;
                }
                match self.block_remaining.compare_exchange(
                    remaining,
                    remaining - 1,
                    std::sync::atomic::Ordering::AcqRel,
                    std::sync::atomic::Ordering::Acquire,
                ) {
                    Ok(_) => return true,
                    Err(observed) => remaining = observed,
                }
            }
        }
    }

    impl ObjectStore for GatedStore {
        fn put(&self, key: &str, bytes: &[u8]) -> crate::Result<()> {
            self.inner.put(key, bytes)
        }

        fn get(&self, key: &str) -> crate::Result<Vec<u8>> {
            if key.starts_with("ns/a/")
                && self
                    .gate
                    .block_next
                    .swap(false, std::sync::atomic::Ordering::AcqRel)
            {
                self.gate.entered.wait();
                self.gate.release.wait();
            }
            self.inner.get(key)
        }

        fn list(&self, prefix: &str) -> crate::Result<Vec<String>> {
            self.inner.list(prefix)
        }

        fn delete(&self, key: &str) -> crate::Result<()> {
            self.inner.delete(key)
        }
    }

    impl ObjectStore for AdmissionStore {
        fn put(&self, key: &str, bytes: &[u8]) -> crate::Result<()> {
            self.inner.put(key, bytes)
        }

        fn get(&self, key: &str) -> crate::Result<Vec<u8>> {
            let active = self
                .active
                .fetch_add(1, std::sync::atomic::Ordering::AcqRel)
                + 1;
            self.update_max(active);
            let first_manifest = self.enabled.load(std::sync::atomic::Ordering::Acquire)
                && key.ends_with("/MANIFEST.json")
                && self.take_block();
            if first_manifest {
                self.entered.wait();
                self.released.wait();
            }
            let result = self.inner.get(key);
            self.active
                .fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
            result
        }

        fn list(&self, prefix: &str) -> crate::Result<Vec<String>> {
            self.inner.list(prefix)
        }

        fn delete(&self, key: &str) -> crate::Result<()> {
            self.inner.delete(key)
        }
    }

    async fn wait_for_transition(coordinator: &Coordinator, name: &str) {
        for _ in 0..1_000 {
            if coordinator
                .registry
                .lock()
                .await
                .transitions
                .contains_key(name)
            {
                return;
            }
            tokio::task::yield_now().await;
        }
        panic!("policy did not register a transition for {name}");
    }

    async fn wait_for_transition_cleared(coordinator: &Coordinator, name: &str) {
        for _ in 0..1_000 {
            if !coordinator
                .registry
                .lock()
                .await
                .transitions
                .contains_key(name)
            {
                return;
            }
            tokio::task::yield_now().await;
        }
        panic!("transition for {name} did not complete");
    }

    async fn wait_for_no_transitions(coordinator: &Coordinator) {
        let mut clear_rounds = 0;
        for _ in 0..1_000 {
            if coordinator.registry.lock().await.transitions.is_empty() {
                clear_rounds += 1;
                if clear_rounds == 3 {
                    return;
                }
            } else {
                clear_rounds = 0;
            }
            tokio::task::yield_now().await;
        }
        panic!("lifecycle transitions did not complete");
    }

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

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn cancelled_queued_load_during_finalization_clears_its_slot() {
        let root = tempdir().expect("tempdir");
        let store = std::sync::Arc::new(LocalDirStore::new(root.path()).expect("store"));
        {
            let engine = Engine::with_store(std::sync::Arc::clone(&store));
            engine
                .upsert(
                    "cancel-finalize",
                    vec![Doc {
                        id: "a".to_owned(),
                        vector: Some(vec![1.0, 0.0]),
                        attributes: BTreeMap::new(),
                    }],
                    Vec::new(),
                    BTreeMap::new(),
                )
                .await
                .expect("seed upsert");
            engine
                .force_flush("cancel-finalize")
                .await
                .expect("seed flush");
        }

        let shared_store: SharedStore = store;
        let coordinator = std::sync::Arc::new(Coordinator::new(shared_store));
        let mut permits = Vec::new();
        for _ in 0..COLD_LOAD_CONCURRENCY {
            permits.push(
                coordinator
                    .cold_load_admission
                    .clone()
                    .acquire_owned()
                    .await
                    .expect("admission permit"),
            );
        }

        let name = "cancel-finalize".to_owned();
        let loading = std::sync::Arc::new(LoadingSlot::new());
        let admission_hook = std::sync::Arc::new(AdmissionWaitHook {
            loading: std::sync::Arc::clone(&loading),
            entered: tokio::sync::Notify::new(),
        });
        assert!(ADMISSION_WAIT_HOOK
            .set(std::sync::Arc::clone(&admission_hook))
            .is_ok());
        let finalize_hook = std::sync::Arc::new(LoadFinalizeHook {
            loading: std::sync::Arc::clone(&loading),
            gate_next: std::sync::atomic::AtomicBool::new(true),
            entered: tokio::sync::Notify::new(),
            release: tokio::sync::Notify::new(),
        });
        assert!(LOAD_FINALIZE_HOOK
            .set(std::sync::Arc::clone(&finalize_hook))
            .is_ok());

        coordinator
            .registry
            .lock()
            .await
            .loading
            .insert(name.clone(), std::sync::Arc::clone(&loading));

        let context = ColdLoadContext {
            registry: coordinator.registry.clone(),
            store: coordinator.store.clone(),
            admission: coordinator.cold_load_admission.clone(),
        };
        let initiating = tokio::spawn({
            let name = name.clone();
            let loading = std::sync::Arc::clone(&loading);
            async move {
                start_open_load(context, name, std::sync::Arc::clone(&loading)).await;
                loading.wait().await
            }
        });
        admission_hook.entered.notified().await;

        let evict = tokio::spawn({
            let coordinator = std::sync::Arc::clone(&coordinator);
            let name = name.clone();
            async move { coordinator.evict(&name).await }
        });
        finalize_hook.entered.notified().await;

        initiating.abort();
        match initiating.await {
            Err(error) => assert!(error.is_cancelled()),
            Ok(_) => panic!("initiating load completed"),
        }

        finalize_hook.release.notify_one();
        tokio::time::timeout(Duration::from_secs(1), evict)
            .await
            .expect("eviction remained behind cancelled finalization")
            .expect("eviction task panicked")
            .expect("eviction failed");

        coordinator.evict(&name).await.expect("subsequent eviction");
        drop(permits);

        let result = coordinator
            .query(
                &name,
                &Query {
                    vector: Some(vec![1.0, 0.0]),
                    text: None,
                    filter: None,
                    top_k: 10,
                    include_attributes: false,
                },
            )
            .await
            .expect("reload after cancelled finalization");
        assert_eq!(result[0].id, "a");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn budget_eviction_is_lru_and_reloads_through_the_coordinator() {
        let root = tempdir().expect("tempdir");
        let store = std::sync::Arc::new(LocalDirStore::new(root.path()).expect("store"));
        let shared_store: SharedStore = store.clone();
        let seed = Coordinator::new(shared_store.clone());
        for name in ["a", "b"] {
            seed.upsert(
                name,
                vec![Doc {
                    id: "doc".to_owned(),
                    vector: Some(vec![1.0, 0.0]),
                    attributes: BTreeMap::new(),
                }],
                Vec::new(),
                BTreeMap::new(),
            )
            .await
            .expect("seed upsert");
            seed.force_flush(name).await.expect("seed flush");
            seed.evict(name).await.expect("seed eviction");
        }

        let probe = Coordinator::with_memory_budget(shared_store.clone(), Some(usize::MAX));
        probe
            .query(
                "a",
                &Query {
                    vector: Some(vec![1.0, 0.0]),
                    text: None,
                    filter: None,
                    top_k: 1,
                    include_attributes: false,
                },
            )
            .await
            .expect("probe a");
        let a_bytes = probe.loaded_memory_bytes().await;
        probe.evict("a").await.expect("probe eviction");
        probe
            .query(
                "b",
                &Query {
                    vector: Some(vec![1.0, 0.0]),
                    text: None,
                    filter: None,
                    top_k: 1,
                    include_attributes: false,
                },
            )
            .await
            .expect("probe b");
        let b_bytes = probe.loaded_memory_bytes().await;
        probe.evict("b").await.expect("probe eviction");

        assert_eq!(
            a_bytes, b_bytes,
            "identical loaded state must account identically"
        );
        let budget = a_bytes.max(b_bytes) + a_bytes.min(b_bytes) - 1;
        let coordinator = Coordinator::with_memory_budget(shared_store, Some(budget));
        let query = Query {
            vector: Some(vec![1.0, 0.0]),
            text: None,
            filter: None,
            top_k: 1,
            include_attributes: false,
        };
        coordinator.query("a", &query).await.expect("load a");
        let a_entry = coordinator
            .registry
            .lock()
            .await
            .entries
            .get("a")
            .cloned()
            .expect("loaded a entry");
        let a_operation = a_entry.operation.lock().await;
        coordinator.query("b", &query).await.expect("load b");
        wait_for_transition(&coordinator, "a").await;
        assert!(coordinator.loaded_memory_bytes().await > budget);

        let b_entry = coordinator
            .registry
            .lock()
            .await
            .entries
            .get("b")
            .cloned()
            .expect("loaded b entry");
        let b_operation = b_entry.operation.lock().await;
        drop(a_operation);
        coordinator.query("a", &query).await.expect("reload a");
        wait_for_transition(&coordinator, "b").await;
        drop(b_operation);
        wait_for_transition_cleared(&coordinator, "b").await;
        assert_eq!(
            coordinator.loaded_namespaces().await.expect("loaded"),
            vec!["a"]
        );
        assert!(coordinator.loaded_memory_bytes().await <= budget);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn budget_policy_projects_draining_bytes_and_evicts_once() {
        let root = tempdir().expect("tempdir");
        let store = std::sync::Arc::new(LocalDirStore::new(root.path()).expect("store"));
        let shared_store: SharedStore = store;
        let coordinator = Coordinator::with_memory_budget(shared_store, Some(usize::MAX));
        let query = Query {
            vector: Some(vec![1.0, 0.0]),
            text: None,
            filter: None,
            top_k: 1,
            include_attributes: false,
        };

        for name in ["a", "b", "c"] {
            coordinator
                .upsert(
                    name,
                    vec![Doc {
                        id: "doc".to_owned(),
                        vector: Some(vec![1.0, 0.0]),
                        attributes: BTreeMap::new(),
                    }],
                    Vec::new(),
                    BTreeMap::new(),
                )
                .await
                .expect("seed upsert");
            coordinator.force_flush(name).await.expect("seed flush");
            coordinator.evict(name).await.expect("seed eviction");
        }

        coordinator.query("a", &query).await.expect("load a");
        let namespace_bytes = coordinator.loaded_memory_bytes().await;
        coordinator.query("b", &query).await.expect("load b");
        coordinator.query("c", &query).await.expect("load c");
        coordinator.query("b", &query).await.expect("touch b");

        let a_entry = coordinator
            .registry
            .lock()
            .await
            .entries
            .get("a")
            .cloned()
            .expect("loaded a entry");
        let a_operation = a_entry.operation.lock().await;
        let budget = namespace_bytes.saturating_mul(2);
        let actions = {
            let mut registry = coordinator.registry.lock().await;
            registry.memory_budget = Some(budget);
            super::policy_evictions_locked(&mut registry, "b")
        };

        assert_eq!(actions.len(), 1, "policy scheduled more than one eviction");
        assert_eq!(actions[0].0, "a", "policy must choose the LRU namespace");
        for (name, action) in actions {
            start_evict_transition(coordinator.registry.clone(), name, action);
        }
        assert_eq!(
            coordinator.registry.lock().await.transitions.len(),
            1,
            "one over-budget decision should register one drain"
        );
        assert!(coordinator.loaded_memory_bytes().await > budget);

        drop(a_operation);
        wait_for_transition_cleared(&coordinator, "a").await;
        assert_eq!(
            coordinator.loaded_namespaces().await.expect("loaded"),
            vec!["b", "c"]
        );
        assert!(coordinator.loaded_memory_bytes().await <= budget);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn policy_keeps_an_inflight_load_out_of_eviction_candidates() {
        let root = tempdir().expect("tempdir");
        let gate = std::sync::Arc::new(LoadGate {
            block_next: std::sync::atomic::AtomicBool::new(false),
            entered: std::sync::Barrier::new(2),
            release: std::sync::Barrier::new(2),
        });
        let store = std::sync::Arc::new(GatedStore {
            inner: LocalDirStore::new(root.path()).expect("store"),
            gate: gate.clone(),
        });
        let shared_store: SharedStore = store.clone();
        let seed = Coordinator::new(shared_store.clone());
        for name in ["a", "b", "c"] {
            seed.upsert(
                name,
                vec![Doc {
                    id: "doc".to_owned(),
                    vector: Some(vec![1.0, 0.0]),
                    attributes: BTreeMap::new(),
                }],
                Vec::new(),
                BTreeMap::new(),
            )
            .await
            .expect("seed upsert");
            seed.force_flush(name).await.expect("seed flush");
            seed.evict(name).await.expect("seed eviction");
        }

        let probe = Coordinator::with_memory_budget(shared_store.clone(), Some(usize::MAX));
        let query = Query {
            vector: Some(vec![1.0, 0.0]),
            text: None,
            filter: None,
            top_k: 1,
            include_attributes: false,
        };
        probe.query("b", &query).await.expect("probe load");
        let namespace_bytes = probe.loaded_memory_bytes().await;
        probe.evict("b").await.expect("probe eviction");

        let coordinator = std::sync::Arc::new(Coordinator::with_memory_budget(
            shared_store,
            Some(namespace_bytes + 1),
        ));
        coordinator.query("b", &query).await.expect("load b");
        let b_entry = coordinator
            .registry
            .lock()
            .await
            .entries
            .get("b")
            .cloned()
            .expect("loaded b entry");
        let b_operation = b_entry.operation.lock().await;

        gate.block_next
            .store(true, std::sync::atomic::Ordering::Release);
        let loading = std::sync::Arc::new(LoadingSlot::new());
        coordinator
            .registry
            .lock()
            .await
            .loading
            .insert("a".to_owned(), loading.clone());
        let context = ColdLoadContext {
            registry: coordinator.registry.clone(),
            store: coordinator.store.clone(),
            admission: coordinator.cold_load_admission.clone(),
        };
        let admission_waiting = loading.admission_waiting.notified();
        let a_load = tokio::spawn({
            let loading = loading.clone();
            async move {
                start_open_load(context, "a".to_owned(), loading.clone()).await;
                loading.wait().await
            }
        });
        admission_waiting.await;
        let entered_gate = {
            let gate = gate.clone();
            tokio::task::spawn_blocking(move || gate.entered.wait())
        };
        entered_gate.await.expect("load gate task");

        coordinator.query("c", &query).await.expect("load c");
        wait_for_transition(&coordinator, "b").await;
        {
            let registry = coordinator.registry.lock().await;
            assert!(registry.loading.contains_key("a"));
            assert!(!registry.transitions.contains_key("a"));
        }

        drop(b_operation);
        wait_for_transition_cleared(&coordinator, "b").await;
        let release_gate = {
            let gate = gate.clone();
            tokio::task::spawn_blocking(move || gate.release.wait())
        };
        release_gate.await.expect("release gate task");
        assert!(a_load.await.expect("load task").is_ok());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn budget_churn_preserves_admission_bound() {
        let root = tempdir().expect("tempdir");
        let admission = COLD_LOAD_CONCURRENCY;
        let store = std::sync::Arc::new(AdmissionStore {
            inner: LocalDirStore::new(root.path()).expect("store"),
            enabled: std::sync::atomic::AtomicBool::new(false),
            block_remaining: std::sync::atomic::AtomicUsize::new(admission),
            active: std::sync::atomic::AtomicUsize::new(0),
            max_active: std::sync::atomic::AtomicUsize::new(0),
            entered: std::sync::Arc::new(std::sync::Barrier::new(admission + 1)),
            released: std::sync::Arc::new(std::sync::Barrier::new(admission + 1)),
        });
        let shared_store: SharedStore = store.clone();
        let seed = Coordinator::new(shared_store.clone());
        let names = (0..admission + 2)
            .map(|index| format!("admission-{index}"))
            .collect::<Vec<_>>();
        for name in &names {
            seed.upsert(
                name,
                vec![Doc {
                    id: "doc".to_owned(),
                    vector: Some(vec![1.0, 0.0]),
                    attributes: BTreeMap::new(),
                }],
                Vec::new(),
                BTreeMap::new(),
            )
            .await
            .expect("seed upsert");
            seed.force_flush(name).await.expect("seed flush");
            seed.evict(name).await.expect("seed eviction");
        }

        let probe = Coordinator::with_memory_budget(shared_store.clone(), Some(usize::MAX));
        let query = Query {
            vector: Some(vec![1.0, 0.0]),
            text: None,
            filter: None,
            top_k: 1,
            include_attributes: false,
        };
        probe.query(&names[0], &query).await.expect("probe load");
        let namespace_bytes = probe.loaded_memory_bytes().await;
        probe.evict(&names[0]).await.expect("probe eviction");
        let budget = namespace_bytes.saturating_mul(2).saturating_add(1);
        let coordinator =
            std::sync::Arc::new(Coordinator::with_memory_budget(shared_store, Some(budget)));
        store.set_enabled();

        let start = std::sync::Arc::new(tokio::sync::Barrier::new(names.len() + 1));
        let mut loads = Vec::new();
        for name in names {
            let coordinator = coordinator.clone();
            let start = start.clone();
            let query = query.clone();
            loads.push(tokio::spawn(async move {
                start.wait().await;
                coordinator.query(&name, &query).await
            }));
        }
        start.wait().await;
        let entered = {
            let store = store.clone();
            tokio::task::spawn_blocking(move || store.entered.wait())
        };
        entered.await.expect("admission gate task");
        let released = {
            let store = store.clone();
            tokio::task::spawn_blocking(move || store.released.wait())
        };
        released.await.expect("release gate task");

        for load in loads {
            let result = load.await.expect("load task");
            assert!(result.is_ok(), "budget churn query failed: {result:?}");
        }
        wait_for_no_transitions(&coordinator).await;
        assert!(
            store.max_active.load(std::sync::atomic::Ordering::Acquire) <= admission,
            "admission high-water mark exceeded: {} > {admission}",
            store.max_active.load(std::sync::atomic::Ordering::Acquire)
        );
        assert!(coordinator.loaded_memory_bytes().await <= budget);
    }

    #[tokio::test]
    async fn oversized_namespace_is_the_budget_soft_floor() {
        let root = tempdir().expect("tempdir");
        let store = std::sync::Arc::new(LocalDirStore::new(root.path()).expect("store"));
        let shared_store: SharedStore = store;
        let coordinator = Coordinator::with_memory_budget(shared_store, Some(1));
        coordinator
            .upsert(
                "oversized",
                vec![Doc {
                    id: "doc".to_owned(),
                    vector: Some(vec![1.0, 0.0]),
                    attributes: BTreeMap::new(),
                }],
                Vec::new(),
                BTreeMap::new(),
            )
            .await
            .expect("upsert oversized namespace");

        assert_eq!(
            coordinator.loaded_namespaces().await.expect("loaded"),
            vec!["oversized"]
        );
        assert!(coordinator.loaded_memory_bytes().await > 1);
    }

    #[tokio::test]
    async fn tombstones_are_included_in_memory_accounting() {
        let root = tempdir().expect("tempdir");
        let store = std::sync::Arc::new(LocalDirStore::new(root.path()).expect("store"));
        let shared_store: SharedStore = store;
        let coordinator = Coordinator::with_memory_budget(shared_store, Some(usize::MAX));
        coordinator
            .upsert(
                "tombstones",
                vec![Doc {
                    id: "doc".to_owned(),
                    vector: Some(vec![1.0, 0.0]),
                    attributes: BTreeMap::new(),
                }],
                Vec::new(),
                BTreeMap::new(),
            )
            .await
            .expect("upsert");
        coordinator.force_flush("tombstones").await.expect("flush");
        let before_delete = coordinator.loaded_memory_bytes().await;

        coordinator
            .upsert(
                "tombstones",
                Vec::new(),
                vec!["doc".to_owned()],
                BTreeMap::new(),
            )
            .await
            .expect("delete");
        assert!(coordinator.loaded_memory_bytes().await > before_delete);
    }

    #[tokio::test]
    async fn unlimited_default_preserves_hot_cache_behavior_without_accounting() {
        let root = tempdir().expect("tempdir");
        let store = std::sync::Arc::new(LocalDirStore::new(root.path()).expect("store"));
        let shared_store: SharedStore = store;
        let coordinator = Coordinator::new(shared_store);
        for index in 0..=HOT_CACHE_CAPACITY {
            coordinator
                .upsert(
                    &format!("ns-{index}"),
                    vec![Doc {
                        id: "doc".to_owned(),
                        vector: Some(vec![1.0, 0.0]),
                        attributes: BTreeMap::new(),
                    }],
                    Vec::new(),
                    BTreeMap::new(),
                )
                .await
                .expect("upsert");
        }

        assert_eq!(coordinator.loaded_memory_bytes().await, 0);
        assert!(coordinator.loaded_namespaces().await.expect("loaded").len() <= HOT_CACHE_CAPACITY);
        coordinator
            .query(
                "ns-0",
                &Query {
                    vector: Some(vec![1.0, 0.0]),
                    text: None,
                    filter: None,
                    top_k: 1,
                    include_attributes: false,
                },
            )
            .await
            .expect("reload after capacity eviction");
    }

    #[tokio::test]
    async fn dropping_coordinator_drops_worker_registry_references() {
        let root = tempdir().expect("tempdir");
        let store = std::sync::Arc::new(LocalDirStore::new(root.path()).expect("store"));
        let shared_store: SharedStore = store;
        let coordinator = std::sync::Arc::new(Coordinator::new(shared_store));
        coordinator
            .upsert(
                "drop-check",
                vec![Doc {
                    id: "doc".to_owned(),
                    vector: None,
                    attributes: BTreeMap::new(),
                }],
                Vec::new(),
                BTreeMap::new(),
            )
            .await
            .expect("upsert");
        let weak_registry = std::sync::Arc::downgrade(&coordinator.registry);
        let entry = coordinator
            .registry
            .lock()
            .await
            .entries
            .get("drop-check")
            .cloned()
            .expect("entry");
        let weak_entry = std::sync::Arc::downgrade(&entry);
        drop(entry);

        drop(coordinator);

        assert!(weak_registry.upgrade().is_none());
        assert!(weak_entry.upgrade().is_none());
    }
}

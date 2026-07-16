//! Namespace lifecycle coordination and bounded hot-cache admission.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex as StdMutex, Weak};

use tokio::sync::{watch, Mutex, Notify, OwnedSemaphorePermit, RwLock, Semaphore};
use tokio::task::JoinHandle;

#[cfg(test)]
use std::sync::OnceLock;

use crate::namespace::{Namespace, Query, QueryResult, Schema, SharedStore, WriteSummary};
use crate::{Error, Result};

pub(crate) const HOT_CACHE_CAPACITY: usize = 8;
pub(crate) const COLD_LOAD_CONCURRENCY: usize = 4;

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
}

/// Coordinates namespace loads, lifecycle transitions, and the hot cache.
pub(crate) struct Coordinator {
    store: SharedStore,
    registry: Arc<Mutex<Registry>>,
    cold_load_admission: Arc<Semaphore>,
}

impl Coordinator {
    pub(crate) fn new(store: SharedStore) -> Self {
        Self {
            store,
            registry: Arc::new(Mutex::new(Registry {
                entries: HashMap::new(),
                loading: HashMap::new(),
                transitions: HashMap::new(),
                lru: VecDeque::new(),
            })),
            cold_load_admission: Arc::new(Semaphore::new(COLD_LOAD_CONCURRENCY)),
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
        let entry = self.get_or_open(namespace).await?;
        let loaded = entry.namespace.read().await;
        ensure_open(&entry, namespace)?;
        loaded.query_with_ef_search(query, ef_search)
    }

    pub(crate) async fn force_flush(&self, namespace: &str) -> Result<()> {
        let entry = self.get_or_open(namespace).await?;
        let _operation = entry.operation.lock().await;
        ensure_open(&entry, namespace)?;
        flush_and_compact(&entry).await
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
                    let closing = registry
                        .transitions
                        .get(&name)
                        .is_some_and(|transition| transition.kind == LifecycleKind::Delete);
                    let entry = Arc::new(NamespaceEntry {
                        namespace: Arc::new(RwLock::new(namespace)),
                        operation: Mutex::new(()),
                        closing: AtomicBool::new(closing),
                        worker: StdMutex::new(None),
                    });
                    entry.start_worker();
                    registry.entries.insert(name.clone(), entry.clone());
                    touch_locked(&mut registry, &name);

                    let evicted = if registry.lru.len() > HOT_CACHE_CAPACITY {
                        registry.lru.pop_front().and_then(|evicted_name| {
                            let entry = registry.entries.get(&evicted_name).cloned()?;
                            entry.closing.store(true, Ordering::Release);
                            let transition =
                                Arc::new(LifecycleTransition::new(LifecycleKind::Evict));
                            registry
                                .transitions
                                .insert(evicted_name.clone(), transition.clone());
                            let loading = registry.loading.get(&evicted_name).cloned();
                            if let Some(loading) = &loading {
                                loading.cancel();
                            }
                            Some((
                                evicted_name,
                                LifecycleAction {
                                    transition,
                                    entry: Some(entry),
                                    loading,
                                },
                            ))
                        })
                    } else {
                        None
                    };
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
                    None,
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
                None,
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
    if let Some((evicted_name, action)) = evicted {
        start_evict_transition(registry, evicted_name, action);
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
        start_open_load, AdmissionWaitHook, ColdLoadContext, Coordinator, LoadFinalizeHook,
        LoadingSlot, WorkerRaceHook, ADMISSION_WAIT_HOOK, COLD_LOAD_CONCURRENCY,
        LOAD_FINALIZE_HOOK, WORKER_RACE_HOOK,
    };
    use crate::engine::Engine;
    use crate::namespace::{Query, SharedStore};
    use crate::store::LocalDirStore;
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
}

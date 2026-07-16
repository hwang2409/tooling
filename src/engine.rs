//! Public engine facade for namespace lifecycle coordination and search.

use std::path::Path;
use std::sync::Arc;

use crate::lifecycle::Coordinator;
use crate::namespace::{Query, QueryResult, Schema, SharedStore, WriteSummary};
use crate::store::{LocalDirStore, ObjectStore};
use crate::Result;

/// The process-wide namespace registry and hot namespace cache.
pub struct Engine {
    coordinator: Coordinator,
}

impl Engine {
    /// Open an engine backed by a local object-store directory.
    pub fn new(root: impl AsRef<Path>) -> Result<Self> {
        let store = Arc::new(LocalDirStore::new(root)?);
        Self::try_with_store(store)
    }

    /// Construct an engine over a caller-provided object store.
    pub fn with_store<S>(store: Arc<S>) -> Self
    where
        S: ObjectStore + Send + Sync + 'static,
    {
        Self::try_with_store(store).expect("invalid PUFFERCLONE_MEMORY_BUDGET_BYTES")
    }

    /// Construct an engine over a caller-provided object store, validating
    /// the process memory-budget configuration.
    pub fn try_with_store<S>(store: Arc<S>) -> Result<Self>
    where
        S: ObjectStore + Send + Sync + 'static,
    {
        let store: SharedStore = store;
        Ok(Self {
            coordinator: Coordinator::from_env(store)?,
        })
    }

    pub fn hot_cache_capacity() -> usize {
        Coordinator::hot_cache_capacity()
    }

    pub fn cold_load_concurrency() -> usize {
        Coordinator::cold_load_concurrency()
    }

    /// Upsert documents and/or delete IDs, creating the namespace on demand.
    pub async fn upsert(
        &self,
        namespace: &str,
        upserts: Vec<crate::Doc>,
        deletes: Vec<String>,
        schema: Schema,
    ) -> Result<WriteSummary> {
        self.coordinator
            .upsert(namespace, upserts, deletes, schema)
            .await
    }

    pub async fn query(&self, namespace: &str, query: &Query) -> Result<Vec<QueryResult>> {
        self.coordinator.query(namespace, query).await
    }

    /// Query a namespace with an explicit HNSW traversal breadth.
    pub async fn query_with_ef_search(
        &self,
        namespace: &str,
        query: &Query,
        ef_search: usize,
    ) -> Result<Vec<QueryResult>> {
        self.coordinator
            .query_with_ef_search(namespace, query, ef_search)
            .await
    }

    pub async fn force_flush(&self, namespace: &str) -> Result<()> {
        self.coordinator.force_flush(namespace).await
    }

    /// Inspect the most recent background flush failure for a loaded
    /// namespace. The namespace remains retryable after an error.
    pub async fn last_flush_error(&self, namespace: &str) -> Result<Option<String>> {
        self.coordinator.last_flush_error(namespace).await
    }

    /// List namespaces from manifests in the store, including cold namespaces.
    pub fn list_namespaces(&self) -> Result<Vec<String>> {
        self.coordinator.list_namespaces()
    }

    /// Delete every object under a namespace prefix and evict it from RAM.
    pub async fn delete_namespace(&self, namespace: &str) -> Result<()> {
        self.coordinator.delete_namespace(namespace).await
    }

    /// Drop one loaded namespace from the hot cache without touching storage.
    pub async fn evict(&self, namespace: &str) -> Result<()> {
        self.coordinator.evict(namespace).await
    }

    pub async fn loaded_namespaces(&self) -> Result<Vec<String>> {
        self.coordinator.loaded_namespaces().await
    }

    /// Return the current estimate for all loaded namespaces.
    pub async fn loaded_memory_bytes(&self) -> usize {
        self.coordinator.loaded_memory_bytes().await
    }
}

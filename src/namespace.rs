//! Per-namespace write, flush, and query state.

use std::collections::{BTreeMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use tokio::sync::Notify;

use crate::index::filter::Filter;
use crate::index::hnsw::Hnsw;
use crate::index::text::{TextIndex, TextStats};
use crate::index::vector::ExactScan;
use crate::index::{rrf_default, sort_scores};
use crate::segment::{
    list_namespace_segment_objects, list_segment_objects, SegmentBuilder, SegmentReader,
};
use crate::store::ObjectStore;
use crate::wal::{wal_key, Manifest, SegmentMeta, WalBatch};
use crate::{AttrValue, Doc, Error, Result};

const DOC_FLUSH_THRESHOLD: usize = 1_000;
const WAL_FLUSH_THRESHOLD: usize = 4 * 1024 * 1024;
const HNSW_MIN_DOCS: usize = 256;
pub const DEFAULT_EF_SEARCH: usize = 64;
pub const MAX_EF_SEARCH: usize = 512;
static SEGMENT_ATTEMPT: AtomicU64 = AtomicU64::new(0);

/// A store handle shared by the engine and its namespaces.
pub type SharedStore = Arc<dyn ObjectStore + Send + Sync>;

/// A schema hint for a document attribute.
pub type Schema = BTreeMap<String, bool>;

/// The result of one accepted write batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WriteSummary {
    pub upserted: usize,
    pub deleted: usize,
}

/// A query accepted by the engine.
#[derive(Debug, Clone, PartialEq)]
pub struct Query {
    pub vector: Option<Vec<f32>>,
    pub text: Option<String>,
    pub filter: Option<Filter>,
    pub top_k: usize,
    pub include_attributes: bool,
}

/// One ranked query result.
#[derive(Debug, Clone, PartialEq)]
pub struct QueryResult {
    pub id: String,
    pub score: f32,
    pub attributes: Option<BTreeMap<String, AttrValue>>,
}

#[derive(Debug, Clone)]
struct MemEntry {
    seq: u64,
    doc: Doc,
}

struct LoadedSegment {
    meta: SegmentMeta,
    docs: Vec<Doc>,
    vectors: ExactScan,
    hnsw: Option<Hnsw>,
    text: TextIndex,
    indexed_fields: Vec<String>,
    tombstones: BTreeMap<String, u64>,
}

pub(crate) struct CompactionPlan {
    candidate: Manifest,
    output: LoadedSegment,
    input_objects: Vec<String>,
    selected: Vec<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Source {
    Segment(usize),
    Memtable,
}

/// Mutable state for one namespace.
pub struct Namespace {
    name: String,
    store: SharedStore,
    pub manifest: Manifest,
    memtable: BTreeMap<String, MemEntry>,
    /// Tombstones are retained after flush. The segment format stores live
    /// documents only, so dropping a tombstone would let an older segment
    /// resurrect the deleted ID.
    tombstones: BTreeMap<String, u64>,
    segments: Vec<LoadedSegment>,
    pending_wal_bytes: usize,
    pending_retire_through: Option<u64>,
    last_flush_error: Option<String>,
    next_seq: u64,
    pub(crate) flush_notify: Arc<Notify>,
}

impl Namespace {
    pub(crate) fn unpersisted(store: SharedStore, name: impl Into<String>) -> Result<Self> {
        let name = validate_namespace(name.into())?;
        let manifest = Manifest {
            vector_dim: None,
            full_text_fields: Vec::new(),
            segments: Vec::new(),
            last_wal_seq: 0,
        };
        Ok(Self {
            name,
            store,
            manifest,
            memtable: BTreeMap::new(),
            tombstones: BTreeMap::new(),
            segments: Vec::new(),
            pending_wal_bytes: 0,
            pending_retire_through: None,
            last_flush_error: None,
            next_seq: 1,
            flush_notify: Arc::new(Notify::new()),
        })
    }

    /// Create and persist an empty namespace manifest.
    pub fn create(store: SharedStore, name: impl Into<String>) -> Result<Self> {
        let namespace = Self::unpersisted(store, name)?;
        namespace
            .manifest
            .store(namespace.store.as_ref(), &namespace.name)?;
        Ok(namespace)
    }

    /// Cold-load a namespace manifest, segments, and its unsegmented WAL tail.
    pub fn open(store: SharedStore, name: impl Into<String>) -> Result<Self> {
        let name = validate_namespace(name.into())?;
        let manifest = Manifest::load(store.as_ref(), &name)?;
        cleanup_orphan_segments(store.as_ref(), &name, &manifest)?;
        let mut segments = Vec::with_capacity(manifest.segments.len());
        let mut tombstones: BTreeMap<String, u64> = BTreeMap::new();
        for meta in &manifest.segments {
            let segment = load_segment(store.as_ref(), &name, meta, &manifest.full_text_fields)?;
            for (id, seq) in &segment.tombstones {
                tombstones
                    .entry(id.clone())
                    .and_modify(|current| *current = (*current).max(*seq))
                    .or_insert(*seq);
            }
            segments.push(segment);
        }

        let covered_seq = manifest
            .segments
            .iter()
            .map(|segment| segment.last_wal_seq)
            .max()
            .unwrap_or(0);
        let mut wal_batches = Vec::new();
        for key in store.list(&format!("ns/{name}/wal/"))? {
            if wal_key_seq(&key).is_some_and(|seq| seq <= covered_seq) {
                store.delete(&key)?;
                continue;
            }
            let batch = WalBatch::decode(&store.get(&key)?)?;
            if batch.seq > covered_seq {
                wal_batches.push((batch.seq, batch));
            }
        }
        wal_batches.sort_by_key(|(seq, _)| *seq);

        let mut namespace = Self {
            name,
            store,
            next_seq: manifest.last_wal_seq.saturating_add(1),
            manifest,
            memtable: BTreeMap::new(),
            tombstones,
            segments,
            pending_wal_bytes: 0,
            pending_retire_through: None,
            last_flush_error: None,
            flush_notify: Arc::new(Notify::new()),
        };
        for (seq, batch) in wal_batches {
            namespace.apply_batch(seq, &batch);
            namespace.pending_wal_bytes = namespace
                .pending_wal_bytes
                .saturating_add(batch.encode()?.len());
            namespace.next_seq = namespace.next_seq.max(seq.saturating_add(1));
            namespace.manifest.last_wal_seq = namespace.manifest.last_wal_seq.max(seq);
        }
        Ok(namespace)
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// Apply a durable upsert/delete batch and return only after the WAL and
    /// manifest have both been persisted.
    pub fn upsert(
        &mut self,
        upserts: Vec<Doc>,
        deletes: Vec<String>,
        schema: Schema,
    ) -> Result<WriteSummary> {
        validate_batch(&upserts, &deletes)?;
        let mut candidate = self.candidate_manifest(&upserts, &schema)?;
        self.retry_wal_retirement()?;
        let seq = self.next_seq;
        let next_seq = seq
            .checked_add(1)
            .ok_or_else(|| Error::Store("WAL sequence exhausted".to_owned()))?;
        let batch = WalBatch {
            seq,
            upserts,
            deletes,
        };
        let bytes = batch.encode()?;
        self.store.put(&wal_key(&self.name, seq), &bytes)?;

        candidate.last_wal_seq = seq;
        if let Err(error) = candidate.store(self.store.as_ref(), &self.name) {
            let _ = self.store.delete(&wal_key(&self.name, seq));
            return Err(error);
        }
        self.manifest = candidate;

        let summary = WriteSummary {
            upserted: batch.upserts.len(),
            deleted: batch.deletes.len(),
        };
        self.apply_batch(seq, &batch);
        self.pending_wal_bytes = self.pending_wal_bytes.saturating_add(bytes.len());
        self.next_seq = next_seq;
        if self.should_flush() {
            self.flush_notify.notify_one();
        }
        Ok(summary)
    }

    /// Force a synchronous segment build. This is primarily useful to tests
    /// and operational callers that need a durable segment immediately.
    pub fn force_flush(&mut self) -> Result<()> {
        self.flush_inner()?;
        self.compact_if_needed()
    }

    /// Return the most recent background flush error, if any. A subsequent
    /// force flush or successful background flush clears it.
    pub fn last_flush_error(&self) -> Option<&str> {
        self.last_flush_error.as_deref()
    }

    pub(crate) fn record_flush_error(&mut self, error: &Error) {
        self.last_flush_error = Some(error.to_string());
    }

    /// Execute a strongly consistent query over segment and memtable state.
    pub fn query(&self, query: &Query) -> Result<Vec<QueryResult>> {
        self.query_with_ef_search(query, DEFAULT_EF_SEARCH)
    }

    /// Query with an explicit HNSW traversal breadth. Exact scans and the
    /// memtable ignore this parameter.
    pub fn query_with_ef_search(
        &self,
        query: &Query,
        ef_search: usize,
    ) -> Result<Vec<QueryResult>> {
        validate_query(query, self.manifest.vector_dim)?;
        let ef_search = clamp_ef_search(ef_search, query.top_k);
        let logical = self.logical_documents();
        let live_ids = logical.keys().cloned().collect::<HashSet<_>>();
        let allowed = logical
            .iter()
            .filter(|(_, (doc, _))| {
                query
                    .filter
                    .as_ref()
                    .is_none_or(|filter| filter.eval(&doc.attributes))
            })
            .map(|(id, _)| id.clone())
            .collect::<HashSet<_>>();

        let vector_results = query
            .vector
            .as_deref()
            .map(|vector| self.vector_search(vector, query.top_k, ef_search, &allowed, &logical));
        let text_results = query
            .text
            .as_deref()
            .map(|text| self.text_search(text, query.top_k, &allowed, &live_ids, &logical));

        let ranked = match (vector_results, text_results) {
            (Some(vector), Some(text)) => {
                let vector_ids = vector.iter().map(|(id, _)| id.clone()).collect::<Vec<_>>();
                let text_ids = text.iter().map(|(id, _)| id.clone()).collect::<Vec<_>>();
                rrf_default(&[vector_ids, text_ids])
            }
            (Some(results), None) | (None, Some(results)) => results,
            (None, None) => unreachable!("validate_query rejects an empty query"),
        };

        Ok(ranked
            .into_iter()
            .take(query.top_k)
            .filter_map(|(id, score)| {
                logical.get(&id).map(|(doc, _)| QueryResult {
                    id,
                    score,
                    attributes: query.include_attributes.then(|| doc.attributes.clone()),
                })
            })
            .collect())
    }

    fn candidate_manifest(&self, upserts: &[Doc], schema: &Schema) -> Result<Manifest> {
        let mut candidate = self.manifest.clone();
        for field in schema.keys() {
            if field.is_empty() {
                return Err(Error::Validation(
                    "schema field names must not be empty".to_owned(),
                ));
            }
        }
        for (field, indexed) in schema {
            if *indexed && !candidate.full_text_fields.contains(field) {
                candidate.full_text_fields.push(field.clone());
            }
        }
        candidate.full_text_fields.sort();
        candidate.full_text_fields.dedup();

        for doc in upserts {
            for field in &candidate.full_text_fields {
                if let Some(value) = doc.attributes.get(field) {
                    if !matches!(value, AttrValue::String(_)) {
                        return Err(Error::Validation(format!(
                            "full-text field '{field}' must be a string"
                        )));
                    }
                }
            }
        }
        let candidate_dim = candidate.vector_dim.or_else(|| {
            upserts
                .iter()
                .find_map(|doc| doc.vector.as_ref().map(Vec::len))
        });
        if let Some(dim) = candidate_dim {
            for doc in upserts {
                if let Some(vector) = &doc.vector {
                    if vector.len() != dim {
                        return Err(Error::Validation(format!(
                            "vector dimension mismatch: expected {dim}, got {}",
                            vector.len()
                        )));
                    }
                }
            }
        }
        candidate.vector_dim = candidate_dim;
        Ok(candidate)
    }

    fn apply_batch(&mut self, seq: u64, batch: &WalBatch) {
        for doc in &batch.upserts {
            self.tombstones
                .get(&doc.id)
                .is_some_and(|tombstone| *tombstone <= seq)
                .then(|| self.tombstones.remove(&doc.id));
            self.memtable.insert(
                doc.id.clone(),
                MemEntry {
                    seq,
                    doc: doc.clone(),
                },
            );
        }
        for id in &batch.deletes {
            self.memtable.remove(id);
            self.tombstones.insert(id.clone(), seq);
        }
    }

    pub(crate) fn should_flush(&self) -> bool {
        self.memtable.len() >= DOC_FLUSH_THRESHOLD || self.pending_wal_bytes >= WAL_FLUSH_THRESHOLD
    }

    fn flush_inner(&mut self) -> Result<()> {
        self.retry_wal_retirement()?;
        if self.memtable.is_empty() && self.pending_wal_bytes == 0 {
            return Ok(());
        }
        let last_seq = self.manifest.last_wal_seq;
        let first_seq = self
            .manifest
            .segments
            .iter()
            .map(|segment| segment.last_wal_seq)
            .max()
            .unwrap_or(0)
            .saturating_add(1);
        if first_seq > last_seq {
            return Ok(());
        }

        let docs = self
            .memtable
            .values()
            .filter(|entry| entry.seq <= last_seq)
            .map(|entry| entry.doc.clone())
            .collect::<Vec<_>>();
        let vectors = ExactScan::build(docs.iter().filter_map(|doc| {
            doc.vector
                .as_ref()
                .map(|vector| (doc.id.clone(), vector.clone()))
        }));
        let text = build_text_index(&docs, &self.manifest.full_text_fields);
        let vector_bytes = vectors.to_bytes()?;
        let text_bytes = text.to_bytes()?;
        let tombstone_bytes = bincode::serialize(&self.tombstones)?;
        let mut sections = vec![
            ("vectors", vector_bytes),
            ("text", text_bytes),
            ("tombstones", tombstone_bytes),
        ];
        if docs.len() >= HNSW_MIN_DOCS {
            let hnsw = Hnsw::build(docs.iter().filter_map(|doc| {
                doc.vector
                    .as_ref()
                    .map(|vector| (doc.id.clone(), vector.clone()))
            }));
            sections.push(("hnsw", hnsw.to_bytes()?));
        }
        let segment_id = format!(
            "seg-{first_seq}-{last_seq}-{}-{}",
            std::process::id(),
            SEGMENT_ATTEMPT.fetch_add(1, Ordering::Relaxed)
        );
        let meta = SegmentBuilder::new(self.store.as_ref(), &self.name, (first_seq, last_seq))
            .with_id(segment_id)
            .build_allow_empty(docs, sections)
            .map_err(|error| Error::Store(error.to_string()))?;
        let loaded = load_segment(
            self.store.as_ref(),
            &self.name,
            &meta,
            &self.manifest.full_text_fields,
        )?;
        let mut candidate = self.manifest.clone();
        candidate.segments.push(meta.clone());
        if let Err(error) = candidate.store(self.store.as_ref(), &self.name) {
            self.last_flush_error = Some(error.to_string());
            return Err(error);
        }
        self.manifest = candidate;
        self.segments.push(loaded);
        self.memtable.retain(|_, entry| entry.seq > last_seq);
        self.pending_wal_bytes = 0;
        self.pending_retire_through = Some(last_seq);
        if let Err(error) = self.retry_wal_retirement() {
            self.last_flush_error = Some(error.to_string());
            return Err(error);
        }
        self.last_flush_error = None;
        Ok(())
    }

    /// Run the intentionally simple v1 compaction policy. Once a trigger
    /// fires, every segment participates in one full merge. Leveled and
    /// tiered policies are future work.
    fn compact_if_needed(&mut self) -> Result<()> {
        let Some(plan) = self.prepare_compaction()? else {
            return Ok(());
        };
        let input_objects = self.publish_compaction(plan)?;
        if let Err(error) = self.finish_compaction(&input_objects) {
            self.last_flush_error = Some(error.to_string());
            return Err(error);
        }
        self.last_flush_error = None;
        Ok(())
    }

    pub(crate) fn flush_only(&mut self) -> Result<()> {
        self.flush_inner()
    }

    /// Build and validate the replacement segment without taking the
    /// namespace write lock. The caller publishes the returned plan under a
    /// short write section after all merge I/O has completed.
    pub(crate) fn prepare_compaction(&self) -> Result<Option<CompactionPlan>> {
        let Some(selected) = self.compaction_selection() else {
            return Ok(None);
        };

        // Compaction is invoked after flush. Keep this guard so a future
        // caller cannot publish a compacted manifest while a WAL tail is
        // still represented only by the memtable.
        if !self.memtable.is_empty() {
            return Ok(None);
        }

        let logical = self.logical_documents();
        if logical
            .values()
            .any(|(_, source)| matches!(source, Source::Memtable))
        {
            return Ok(None);
        }

        let input_metas = selected
            .iter()
            .map(|&index| self.segments[index].meta.clone())
            .collect::<Vec<_>>();
        let input_objects = input_metas
            .iter()
            .try_fold(Vec::new(), |mut objects, meta| {
                let keys = list_segment_objects(self.store.as_ref(), &self.name, &meta.id)
                    .map_err(|error| Error::Store(error.to_string()))?;
                objects.extend(keys);
                Ok::<_, Error>(objects)
            })?;
        let first_wal_seq = input_metas
            .iter()
            .map(|meta| meta.first_wal_seq)
            .min()
            .expect("compaction selection is non-empty");
        let last_wal_seq = input_metas
            .iter()
            .map(|meta| meta.last_wal_seq)
            .max()
            .expect("compaction selection is non-empty");
        let docs = logical
            .into_values()
            .map(|(doc, _)| doc)
            .collect::<Vec<_>>();
        let vectors = ExactScan::build(docs.iter().filter_map(|doc| {
            doc.vector
                .as_ref()
                .map(|vector| (doc.id.clone(), vector.clone()))
        }));
        let text = build_text_index(&docs, &self.manifest.full_text_fields);
        let tombstone_bytes = bincode::serialize(&BTreeMap::<String, u64>::new())?;
        let mut sections = vec![
            ("vectors", vectors.to_bytes()?),
            ("text", text.to_bytes()?),
            ("tombstones", tombstone_bytes),
        ];
        if docs.len() >= HNSW_MIN_DOCS {
            let hnsw = Hnsw::build(docs.iter().filter_map(|doc| {
                doc.vector
                    .as_ref()
                    .map(|vector| (doc.id.clone(), vector.clone()))
            }));
            sections.push(("hnsw", hnsw.to_bytes()?));
        }

        let segment_id = format!(
            "compact-{first_wal_seq}-{last_wal_seq}-{}-{}",
            std::process::id(),
            SEGMENT_ATTEMPT.fetch_add(1, Ordering::Relaxed)
        );
        let output_meta = SegmentBuilder::new(
            self.store.as_ref(),
            &self.name,
            (first_wal_seq, last_wal_seq),
        )
        .with_id(segment_id)
        .build_allow_empty(docs, sections)
        .map_err(|error| Error::Store(error.to_string()))?;
        let output = load_segment(
            self.store.as_ref(),
            &self.name,
            &output_meta,
            &self.manifest.full_text_fields,
        )?;

        let selected_set = selected.iter().copied().collect::<HashSet<_>>();
        let first_selected = selected[0];
        let mut candidate = self.manifest.clone();
        candidate.segments.clear();
        for (index, meta) in self.manifest.segments.iter().enumerate() {
            if index == first_selected {
                candidate.segments.push(output_meta.clone());
            }
            if !selected_set.contains(&index) {
                candidate.segments.push(meta.clone());
            }
        }
        // The output covers exactly the sequence range of the replaced
        // segments; the namespace's WAL high-water mark is unchanged.
        candidate.last_wal_seq = self.manifest.last_wal_seq;
        Ok(Some(CompactionPlan {
            candidate,
            output,
            input_objects,
            selected,
        }))
    }

    /// Publish a validated compaction plan and return the old objects for
    /// deletion. Manifest publication is the commit point; no in-memory state
    /// changes before it succeeds.
    pub(crate) fn publish_compaction(&mut self, plan: CompactionPlan) -> Result<Vec<String>> {
        let CompactionPlan {
            candidate,
            output,
            input_objects,
            selected,
        } = plan;
        candidate.store(self.store.as_ref(), &self.name)?;

        let selected_set = selected.iter().copied().collect::<HashSet<_>>();
        let first_selected = selected[0];
        let old_segments = std::mem::take(&mut self.segments);
        let mut output = Some(output);
        let mut next_segments = Vec::with_capacity(candidate.segments.len());
        for (index, segment) in old_segments.into_iter().enumerate() {
            if index == first_selected {
                next_segments.push(output.take().expect("output inserted once"));
            }
            if !selected_set.contains(&index) {
                next_segments.push(segment);
            }
        }
        self.manifest = candidate;
        self.segments = next_segments;
        // All current segments were selected by the full-compaction policy,
        // so no older object can still resurrect an ID after this swap.
        self.tombstones.clear();
        Ok(input_objects)
    }

    /// Delete input objects after the manifest swap. This method only needs a
    /// shared namespace reference, so queries can continue while cleanup runs.
    pub(crate) fn finish_compaction(&self, input_objects: &[String]) -> Result<()> {
        for key in input_objects {
            self.store.delete(key)?;
        }
        Ok(())
    }

    fn compaction_selection(&self) -> Option<Vec<usize>> {
        if self.segments.is_empty() {
            return None;
        }
        let logical = self.logical_documents();
        let has_stale_segment = self.segments.iter().enumerate().any(|(index, segment)| {
            let dead_docs = segment
                .docs
                .iter()
                .filter(|doc| {
                    !logical
                        .get(&doc.id)
                        .is_some_and(|(_, source)| *source == Source::Segment(index))
                })
                .count();
            dead_docs > segment.docs.len() / 2
        });
        (self.segments.len() >= 4 || has_stale_segment).then(|| (0..self.segments.len()).collect())
    }

    fn retry_wal_retirement(&mut self) -> Result<()> {
        let Some(through) = self.pending_retire_through else {
            return Ok(());
        };
        let keys = self.store.list(&format!("ns/{}/wal/", self.name))?;
        for key in keys {
            if wal_key_seq(&key).is_some_and(|seq| seq <= through) {
                self.store.delete(&key)?;
            }
        }
        self.pending_retire_through = None;
        Ok(())
    }

    fn logical_documents(&self) -> BTreeMap<String, (Doc, Source)> {
        let mut logical = BTreeMap::new();
        for (segment_index, segment) in self.segments.iter().enumerate() {
            for doc in &segment.docs {
                if self
                    .tombstones
                    .get(&doc.id)
                    .is_some_and(|tombstone| *tombstone > segment.meta.last_wal_seq)
                {
                    continue;
                }
                logical.insert(
                    doc.id.clone(),
                    (doc.clone(), Source::Segment(segment_index)),
                );
            }
        }
        for (id, entry) in &self.memtable {
            if self
                .tombstones
                .get(id)
                .is_some_and(|tombstone| *tombstone >= entry.seq)
            {
                continue;
            }
            logical.insert(id.clone(), (entry.doc.clone(), Source::Memtable));
        }
        logical
    }

    fn vector_search(
        &self,
        query: &[f32],
        top_k: usize,
        ef_search: usize,
        allowed: &HashSet<String>,
        logical: &BTreeMap<String, (Doc, Source)>,
    ) -> Vec<(String, f32)> {
        let mut results = Vec::new();
        for (index, segment) in self.segments.iter().enumerate() {
            let ids = source_ids(logical, allowed, Source::Segment(index));
            if let Some(hnsw) = &segment.hnsw {
                results.extend(hnsw.search_filtered_with_ef(query, top_k, ef_search, Some(&ids)));
            } else {
                results.extend(segment.vectors.search_filtered(query, top_k, Some(&ids)));
            }
        }
        let memtable_index = ExactScan::build(logical.iter().filter_map(|(id, (doc, source))| {
            (*source == Source::Memtable)
                .then(|| {
                    doc.vector
                        .as_ref()
                        .map(|vector| (id.clone(), vector.clone()))
                })
                .flatten()
        }));
        let mem_ids = source_ids(logical, allowed, Source::Memtable);
        results.extend(memtable_index.search_filtered(query, top_k, Some(&mem_ids)));
        sort_scores(&mut results);
        results.truncate(top_k);
        results
    }

    fn text_search(
        &self,
        query: &str,
        top_k: usize,
        allowed: &HashSet<String>,
        live_ids: &HashSet<String>,
        logical: &BTreeMap<String, (Doc, Source)>,
    ) -> Vec<(String, f32)> {
        let mut indexes = Vec::with_capacity(self.segments.len() + 1);
        let mut source_allowed = Vec::with_capacity(self.segments.len() + 1);
        let mut source_live = Vec::with_capacity(self.segments.len() + 1);
        for (index, segment) in self.segments.iter().enumerate() {
            indexes.push(
                if segment.indexed_fields == self.manifest.full_text_fields {
                    segment.text.clone()
                } else {
                    build_text_index(&segment.docs, &self.manifest.full_text_fields)
                },
            );
            source_allowed.push(source_ids(logical, allowed, Source::Segment(index)));
            source_live.push(source_ids(logical, live_ids, Source::Segment(index)));
        }
        let mem_docs = logical
            .iter()
            .filter_map(|(_, (doc, source))| (*source == Source::Memtable).then_some(doc.clone()))
            .collect::<Vec<_>>();
        indexes.push(build_text_index(&mem_docs, &self.manifest.full_text_fields));
        source_allowed.push(source_ids(logical, allowed, Source::Memtable));
        source_live.push(source_ids(logical, live_ids, Source::Memtable));

        let mut stats = TextStats::default();
        for (index, ids) in indexes.iter().zip(&source_live) {
            stats.merge(&index.stats_filtered(Some(ids)));
        }
        let mut results = Vec::new();
        for (index, ids) in indexes.iter().zip(source_allowed) {
            results.extend(index.search_with_stats_filtered(query, top_k, &stats, Some(&ids)));
        }
        sort_scores(&mut results);
        results.truncate(top_k);
        results
    }
}

fn load_segment(
    store: &dyn ObjectStore,
    namespace: &str,
    meta: &SegmentMeta,
    fields: &[String],
) -> Result<LoadedSegment> {
    let reader = SegmentReader::open(store, namespace, &meta.id)
        .map_err(|error| Error::Store(error.to_string()))?;
    let docs = reader.documents().cloned().collect::<Vec<_>>();
    let vectors = match reader.section_bytes("vectors") {
        Ok(bytes) => {
            ExactScan::from_bytes(&bytes).map_err(|error| Error::Store(error.to_string()))?
        }
        Err(crate::segment::SegmentError::SectionNotFound(_)) => {
            ExactScan::build(docs.iter().filter_map(|doc| {
                doc.vector
                    .as_ref()
                    .map(|vector| (doc.id.clone(), vector.clone()))
            }))
        }
        Err(error) => return Err(Error::Store(error.to_string())),
    };
    let hnsw = match reader.section_bytes("hnsw") {
        Ok(bytes) => {
            Some(Hnsw::from_bytes(&bytes).map_err(|error| Error::Store(error.to_string()))?)
        }
        Err(crate::segment::SegmentError::SectionNotFound(_)) => None,
        Err(error) => return Err(Error::Store(error.to_string())),
    };
    let _persisted_text = match reader.section_bytes("text") {
        Ok(bytes) => {
            TextIndex::from_bytes(&bytes).map_err(|error| Error::Store(error.to_string()))?
        }
        Err(crate::segment::SegmentError::SectionNotFound(_)) => build_text_index(&docs, fields),
        Err(error) => return Err(Error::Store(error.to_string())),
    };
    let tombstones = match reader.section_bytes("tombstones") {
        Ok(bytes) => {
            bincode::deserialize(&bytes).map_err(|error| Error::Store(error.to_string()))?
        }
        Err(crate::segment::SegmentError::SectionNotFound(_)) => BTreeMap::new(),
        Err(error) => return Err(Error::Store(error.to_string())),
    };
    let text = build_text_index(&docs, fields);
    Ok(LoadedSegment {
        meta: meta.clone(),
        docs,
        vectors,
        hnsw,
        text,
        indexed_fields: fields.to_vec(),
        tombstones,
    })
}

fn cleanup_orphan_segments(
    store: &dyn ObjectStore,
    namespace: &str,
    manifest: &Manifest,
) -> Result<()> {
    let referenced = manifest
        .segments
        .iter()
        .map(|segment| segment.id.as_str())
        .collect::<HashSet<_>>();
    let keys = list_namespace_segment_objects(store, namespace)
        .map_err(|error| Error::Store(error.to_string()))?;
    let prefix = format!("ns/{namespace}/segments/");
    for key in keys {
        let Some(segment_id) = key
            .strip_prefix(&prefix)
            .and_then(|suffix| suffix.split('/').next())
        else {
            continue;
        };
        if !referenced.contains(segment_id) {
            store.delete(&key)?;
        }
    }
    Ok(())
}

fn build_text_index(docs: &[Doc], fields: &[String]) -> TextIndex {
    TextIndex::build(docs.iter().flat_map(|doc| {
        fields.iter().filter_map(|field| {
            doc.attributes.get(field).and_then(|value| match value {
                AttrValue::String(text) => Some((doc.id.clone(), text.clone())),
                _ => None,
            })
        })
    }))
}

fn wal_key_seq(key: &str) -> Option<u64> {
    key.strip_suffix(".wal")?.rsplit('/').next()?.parse().ok()
}

fn source_ids(
    logical: &BTreeMap<String, (Doc, Source)>,
    allowed: &HashSet<String>,
    source: Source,
) -> HashSet<String> {
    logical
        .iter()
        .filter(|(id, (_, candidate))| *candidate == source && allowed.contains(*id))
        .map(|(id, _)| id.clone())
        .collect()
}

fn validate_namespace(name: String) -> Result<String> {
    if name.is_empty() || name == "." || name == ".." || name.contains('/') || name.contains('\\') {
        return Err(Error::InvalidKey(format!("invalid namespace: {name}")));
    }
    Ok(name)
}

fn validate_batch(upserts: &[Doc], deletes: &[String]) -> Result<()> {
    if upserts.is_empty() && deletes.is_empty() {
        return Err(Error::Validation("batch must not be empty".to_owned()));
    }
    for doc in upserts {
        if doc.id.is_empty() {
            return Err(Error::Validation(
                "document IDs must not be empty".to_owned(),
            ));
        }
        if doc
            .vector
            .as_ref()
            .is_some_and(|vector| vector.iter().any(|value| !value.is_finite()))
        {
            return Err(Error::Validation(format!(
                "vector for '{}' contains a non-finite value",
                doc.id
            )));
        }
    }
    if deletes.iter().any(String::is_empty) {
        return Err(Error::Validation(
            "deleted IDs must not be empty".to_owned(),
        ));
    }
    Ok(())
}

fn validate_query(query: &Query, vector_dim: Option<usize>) -> Result<()> {
    if query.vector.is_none() && query.text.is_none() {
        return Err(Error::Validation(
            "query must include vector or text".to_owned(),
        ));
    }
    if query.top_k == 0 {
        return Err(Error::Validation(
            "top_k must be greater than zero".to_owned(),
        ));
    }
    if query.top_k > MAX_EF_SEARCH {
        return Err(Error::Validation(format!(
            "top_k must not exceed {MAX_EF_SEARCH}"
        )));
    }
    if let (Some(expected), Some(vector)) = (vector_dim, &query.vector) {
        if vector.len() != expected {
            return Err(Error::Validation(format!(
                "vector dimension mismatch: expected {expected}, got {}",
                vector.len()
            )));
        }
    }
    if query
        .vector
        .as_ref()
        .is_some_and(|vector| vector.iter().any(|value| !value.is_finite()))
    {
        return Err(Error::Validation(
            "query vector contains a non-finite value".to_owned(),
        ));
    }
    Ok(())
}

fn clamp_ef_search(requested: usize, top_k: usize) -> usize {
    requested.clamp(top_k.max(1), MAX_EF_SEARCH)
}

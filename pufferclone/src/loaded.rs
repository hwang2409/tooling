//! Loaded immutable segments and their memory-accounting estimate.

use std::collections::{BTreeMap, HashSet};

use serde::Serialize;

use crate::index::hnsw::Hnsw;
use crate::index::text::TextIndex;
use crate::index::vector::ExactScan;
use crate::segment::{list_namespace_segment_objects, SegmentError, SegmentReader};
use crate::store::ObjectStore;
use crate::wal::SegmentMeta;
use crate::{AttrValue, Doc, Error, Result};

/// The immutable data and indexes retained for one loaded segment.
pub(crate) struct LoadedSegment {
    meta: SegmentMeta,
    docs: Vec<Doc>,
    vectors: ExactScan,
    hnsw: Option<Hnsw>,
    text: TextIndex,
    indexed_fields: Vec<String>,
    tombstones: BTreeMap<String, u64>,
    doc_bytes: usize,
    index_bytes: usize,
}

impl LoadedSegment {
    /// Load one segment and rebuild any index section absent from older data.
    pub(crate) fn load(
        store: &dyn ObjectStore,
        namespace: &str,
        meta: &SegmentMeta,
        fields: &[String],
    ) -> Result<Self> {
        let reader = SegmentReader::open(store, namespace, &meta.id)
            .map_err(|error| Error::Store(error.to_string()))?;
        let doc_bytes = reader.document_bytes();
        let docs = reader.documents().cloned().collect::<Vec<_>>();
        let (vectors, vectors_bytes) = match reader.section_bytes("vectors") {
            Ok(bytes) => {
                let size = bytes.len();
                (
                    ExactScan::from_bytes(&bytes)
                        .map_err(|error| Error::Store(error.to_string()))?,
                    size,
                )
            }
            Err(SegmentError::SectionNotFound(_)) => {
                let index = ExactScan::build(docs.iter().filter_map(|doc| {
                    doc.vector
                        .as_ref()
                        .map(|vector| (doc.id.clone(), vector.clone()))
                }));
                let size = index.to_bytes()?.len();
                (index, size)
            }
            Err(error) => return Err(Error::Store(error.to_string())),
        };
        let (hnsw, hnsw_bytes) = match reader.section_bytes("hnsw") {
            Ok(bytes) => {
                let size = bytes.len();
                (
                    Some(
                        Hnsw::from_bytes(&bytes)
                            .map_err(|error| Error::Store(error.to_string()))?,
                    ),
                    size,
                )
            }
            Err(SegmentError::SectionNotFound(_)) => (None, 0),
            Err(error) => return Err(Error::Store(error.to_string())),
        };
        let (_persisted_text, persisted_text_bytes) = match reader.section_bytes("text") {
            Ok(bytes) => {
                let size = bytes.len();
                (
                    TextIndex::from_bytes(&bytes)
                        .map_err(|error| Error::Store(error.to_string()))?,
                    size,
                )
            }
            Err(SegmentError::SectionNotFound(_)) => (build_text_index(&docs, fields), 0),
            Err(error) => return Err(Error::Store(error.to_string())),
        };
        let text = build_text_index(&docs, fields);
        let text_bytes = if persisted_text_bytes == 0 {
            text.to_bytes()?.len()
        } else {
            persisted_text_bytes
        };
        let (tombstones, tombstone_bytes) = match reader.section_bytes("tombstones") {
            Ok(bytes) => {
                let size = bytes.len();
                (
                    bincode::deserialize(&bytes)
                        .map_err(|error| Error::Store(error.to_string()))?,
                    size,
                )
            }
            Err(SegmentError::SectionNotFound(_)) => {
                let tombstones = BTreeMap::new();
                let size = bincode::serialized_size(&tombstones)? as usize;
                (tombstones, size)
            }
            Err(error) => return Err(Error::Store(error.to_string())),
        };
        Ok(Self {
            meta: meta.clone(),
            docs,
            vectors,
            hnsw,
            text,
            indexed_fields: fields.to_vec(),
            tombstones,
            doc_bytes,
            index_bytes: vectors_bytes
                .saturating_add(hnsw_bytes)
                .saturating_add(text_bytes)
                .saturating_add(tombstone_bytes),
        })
    }

    pub(crate) fn meta(&self) -> &SegmentMeta {
        &self.meta
    }

    pub(crate) fn docs(&self) -> &[Doc] {
        &self.docs
    }

    pub(crate) fn vectors(&self) -> &ExactScan {
        &self.vectors
    }

    pub(crate) fn hnsw(&self) -> Option<&Hnsw> {
        self.hnsw.as_ref()
    }

    pub(crate) fn text(&self) -> &TextIndex {
        &self.text
    }

    pub(crate) fn indexed_fields(&self) -> &[String] {
        &self.indexed_fields
    }

    pub(crate) fn tombstones(&self) -> &BTreeMap<String, u64> {
        &self.tombstones
    }

    fn memory_bytes(&self) -> usize {
        self.doc_bytes.saturating_add(self.index_bytes)
    }
}

/// Estimate immutable segments plus the mutable memtable and tombstones.
pub(crate) fn estimate_memory_bytes<T: Serialize>(
    segments: &[LoadedSegment],
    memtable: &T,
    tombstones: &BTreeMap<String, u64>,
) -> usize {
    let segment_bytes = segments
        .iter()
        .map(LoadedSegment::memory_bytes)
        .fold(0usize, usize::saturating_add);
    let memtable_bytes = bincode::serialized_size(memtable).unwrap_or(0) as usize;
    let tombstone_bytes = bincode::serialized_size(tombstones).unwrap_or(0) as usize;
    segment_bytes
        .saturating_add(memtable_bytes)
        .saturating_add(tombstone_bytes)
}

pub(crate) fn cleanup_orphan_segments(
    store: &dyn ObjectStore,
    namespace: &str,
    manifest: &crate::wal::Manifest,
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

pub(crate) fn build_text_index(docs: &[Doc], fields: &[String]) -> TextIndex {
    TextIndex::build(docs.iter().flat_map(|doc| {
        fields.iter().filter_map(|field| {
            doc.attributes.get(field).and_then(|value| match value {
                AttrValue::String(text) => Some((doc.id.clone(), text.clone())),
                _ => None,
            })
        })
    }))
}

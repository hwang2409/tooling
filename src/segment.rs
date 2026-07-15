//! Immutable, object-store-backed document segments.
//!
//! A segment stores its documents in docs.bin and keeps index implementations
//! deliberately out of this module. Indexes are opaque named byte sections,
//! which lets each index own its serialization format.

use std::collections::BTreeMap;
use std::ops::RangeInclusive;

use crc32fast::Hasher;
use serde::{Deserialize, Serialize};

use crate::store::ObjectStore;
use crate::wal::SegmentMeta;
use crate::{AttrValue, Doc};

type EncodedDoc = (Option<Vec<f32>>, BTreeMap<String, AttrValue>);
type EncodedDocs = BTreeMap<String, EncodedDoc>;

/// The result type for segment build and read operations.
pub type Result<T> = std::result::Result<T, SegmentError>;

/// Errors raised while writing or reading an immutable segment.
#[derive(Debug, thiserror::Error)]
pub enum SegmentError {
    /// An object-store operation failed.
    #[error("object-store error: {0}")]
    Store(#[from] crate::Error),

    /// A segment object could not be serialized or deserialized.
    #[error("bincode error: {0}")]
    Bincode(#[from] Box<bincode::ErrorKind>),

    /// A segment metadata object could not be serialized or deserialized.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// The segment input violates a container invariant.
    #[error("invalid segment: {0}")]
    InvalidSegment(String),

    /// A named section is not valid or collides with a reserved file name.
    #[error("invalid section name: {0}")]
    InvalidSectionName(String),

    /// A requested section is not present in the segment.
    #[error("section not found: {0}")]
    SectionNotFound(String),

    /// A persisted object has a different length from its metadata.
    #[error("size mismatch for {name}: expected {expected}, got {actual}")]
    SizeMismatch {
        /// The logical object name.
        name: String,
        /// The length recorded in metadata.
        expected: u64,
        /// The actual object length.
        actual: u64,
    },

    /// A persisted object has a different checksum from its metadata.
    #[error("checksum mismatch for {name}: expected {expected}, got {actual}")]
    ChecksumMismatch {
        /// The logical object name.
        name: String,
        /// The checksum recorded in metadata.
        expected: u32,
        /// The actual checksum.
        actual: u32,
    },
}

/// A WAL sequence range accepted by SegmentBuilder::new.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WalSeqRange {
    /// The first sequence included in the segment.
    pub first: u64,
    /// The last sequence included in the segment.
    pub last: u64,
}

impl From<RangeInclusive<u64>> for WalSeqRange {
    fn from(range: RangeInclusive<u64>) -> Self {
        let (first, last) = range.into_inner();
        Self { first, last }
    }
}

impl From<(u64, u64)> for WalSeqRange {
    fn from((first, last): (u64, u64)) -> Self {
        Self { first, last }
    }
}

/// Builds one immutable segment under a namespace's segment prefix.
///
/// Segment identifiers are sequence-based: seg-<first>-<last>. Rebuilding
/// the same WAL range therefore targets the same write-once objects and is
/// rejected by the object store rather than silently replacing a segment.
pub struct SegmentBuilder<'a> {
    store: &'a dyn ObjectStore,
    namespace: String,
    wal_range: WalSeqRange,
}

impl<'a> SegmentBuilder<'a> {
    /// Create a builder for namespace and the WAL range it will cover.
    pub fn new<R>(store: &'a dyn ObjectStore, namespace: impl Into<String>, range: R) -> Self
    where
        R: Into<WalSeqRange>,
    {
        Self {
            store,
            namespace: namespace.into(),
            wal_range: range.into(),
        }
    }

    /// Write the documents and opaque index sections, returning manifest metadata.
    pub fn build<I, S, N, B>(self, docs: I, sections: S) -> Result<SegmentMeta>
    where
        I: IntoIterator<Item = Doc>,
        S: IntoIterator<Item = (N, B)>,
        N: AsRef<str>,
        B: AsRef<[u8]>,
    {
        if self.namespace.is_empty() {
            return Err(SegmentError::InvalidSegment(
                "namespace must not be empty".to_owned(),
            ));
        }
        if self.wal_range.first > self.wal_range.last {
            return Err(SegmentError::InvalidSegment(
                "WAL range must be ascending".to_owned(),
            ));
        }

        let mut encoded_docs = BTreeMap::new();
        for doc in docs {
            let id = doc.id;
            if encoded_docs
                .insert(id.clone(), (doc.vector, doc.attributes))
                .is_some()
            {
                return Err(SegmentError::InvalidSegment(format!(
                    "duplicate document id: {id}"
                )));
            }
        }
        if encoded_docs.is_empty() {
            return Err(SegmentError::InvalidSegment(
                "cannot build an empty segment".to_owned(),
            ));
        }

        let mut section_bytes = BTreeMap::new();
        for (name, bytes) in sections {
            let name = name.as_ref().to_owned();
            validate_section_name(&name)?;
            if section_bytes
                .insert(name.clone(), bytes.as_ref().to_vec())
                .is_some()
            {
                return Err(SegmentError::InvalidSegment(format!(
                    "duplicate section name: {name}"
                )));
            }
        }

        let id = segment_id(self.wal_range);
        let prefix = segment_prefix(&self.namespace, &id);
        let docs_bytes = bincode::serialize(&encoded_docs)?;

        self.store
            .put(&format!("{prefix}/docs.bin"), &docs_bytes)
            .map_err(SegmentError::Store)?;

        for (name, bytes) in &section_bytes {
            self.store
                .put(&format!("{prefix}/{name}.bin"), bytes)
                .map_err(SegmentError::Store)?;
        }

        let meta = SegmentMeta {
            id,
            doc_count: encoded_docs.len(),
            first_wal_seq: self.wal_range.first,
            last_wal_seq: self.wal_range.last,
            sections: section_bytes.keys().cloned().collect(),
        };
        let disk_meta = SegmentDiskMeta {
            id: meta.id.clone(),
            doc_count: meta.doc_count,
            first_wal_seq: meta.first_wal_seq,
            last_wal_seq: meta.last_wal_seq,
            docs: FileMeta::for_bytes("docs", &docs_bytes),
            sections: section_bytes
                .iter()
                .map(|(name, bytes)| FileMeta::for_bytes(name, bytes))
                .collect(),
        };
        let meta_bytes = serde_json::to_vec_pretty(&disk_meta)?;
        self.store
            .put(&format!("{prefix}/meta.json"), &meta_bytes)
            .map_err(SegmentError::Store)?;

        Ok(meta)
    }

    /// Build a document-only segment without opaque index sections.
    pub fn build_docs<I>(self, docs: I) -> Result<SegmentMeta>
    where
        I: IntoIterator<Item = Doc>,
    {
        self.build(docs, std::iter::empty::<(String, Vec<u8>)>())
    }
}

/// Reads the documents and opaque sections of one immutable segment.
pub struct SegmentReader<'a> {
    store: &'a dyn ObjectStore,
    namespace: String,
    meta: SegmentMeta,
    docs: Vec<Doc>,
}

impl<'a> SegmentReader<'a> {
    /// Alias for SegmentReader::open.
    pub fn new(
        store: &'a dyn ObjectStore,
        namespace: impl Into<String>,
        segment_id: impl Into<String>,
    ) -> Result<Self> {
        Self::open(store, namespace, segment_id)
    }

    /// Load and verify a segment's metadata and document payload.
    pub fn open(
        store: &'a dyn ObjectStore,
        namespace: impl Into<String>,
        segment_id: impl Into<String>,
    ) -> Result<Self> {
        let namespace = namespace.into();
        let segment_id = segment_id.into();
        if namespace.is_empty() || segment_id.is_empty() {
            return Err(SegmentError::InvalidSegment(
                "namespace and segment id must not be empty".to_owned(),
            ));
        }

        let prefix = segment_prefix(&namespace, &segment_id);
        let disk_meta: SegmentDiskMeta =
            serde_json::from_slice(&store.get(&format!("{prefix}/meta.json"))?)?;
        if disk_meta.id != segment_id {
            return Err(SegmentError::InvalidSegment(format!(
                "metadata id {} does not match requested segment {segment_id}",
                disk_meta.id
            )));
        }

        let docs_bytes = store.get(&format!("{prefix}/docs.bin"))?;
        verify_file(&disk_meta.docs, &docs_bytes)?;
        let encoded_docs: EncodedDocs = bincode::deserialize(&docs_bytes)?;
        if encoded_docs.len() != disk_meta.doc_count {
            return Err(SegmentError::InvalidSegment(format!(
                "metadata says {} documents but docs.bin contains {}",
                disk_meta.doc_count,
                encoded_docs.len()
            )));
        }
        let docs = encoded_docs
            .into_iter()
            .map(|(id, (vector, attributes))| Doc {
                id,
                vector,
                attributes,
            })
            .collect();

        Ok(Self {
            store,
            namespace,
            meta: SegmentMeta {
                id: disk_meta.id,
                doc_count: disk_meta.doc_count,
                first_wal_seq: disk_meta.first_wal_seq,
                last_wal_seq: disk_meta.last_wal_seq,
                sections: disk_meta
                    .sections
                    .iter()
                    .map(|section| section.name.clone())
                    .collect(),
            },
            docs,
        })
    }

    /// Return manifest-compatible metadata for this segment.
    pub fn meta(&self) -> &SegmentMeta {
        &self.meta
    }

    /// Iterate over all documents in deterministic ID order.
    pub fn documents(&self) -> impl Iterator<Item = &Doc> {
        self.docs.iter()
    }

    /// Alias for SegmentReader::documents.
    pub fn iter_docs(&self) -> impl Iterator<Item = &Doc> {
        self.documents()
    }

    /// Alias for SegmentReader::documents.
    pub fn iter(&self) -> impl Iterator<Item = &Doc> {
        self.documents()
    }

    /// Find a document by its ID.
    pub fn get(&self, id: &str) -> Option<&Doc> {
        self.docs.iter().find(|doc| doc.id == id)
    }

    /// Alias for SegmentReader::get.
    pub fn get_doc(&self, id: &str) -> Option<&Doc> {
        self.get(id)
    }

    /// Alias for SegmentReader::get.
    pub fn doc(&self, id: &str) -> Option<&Doc> {
        self.get(id)
    }

    /// Load and verify an opaque index section.
    pub fn section_bytes(&self, name: &str) -> Result<Vec<u8>> {
        validate_section_name(name)?;
        let disk_meta = self.load_disk_meta()?;
        let file_meta = disk_meta
            .sections
            .iter()
            .find(|section| section.name == name)
            .ok_or_else(|| SegmentError::SectionNotFound(name.to_owned()))?;
        let bytes = self.store.get(&format!(
            "{}/{name}.bin",
            segment_prefix(&self.namespace, &self.meta.id)
        ))?;
        verify_file(file_meta, &bytes)?;
        Ok(bytes)
    }

    /// Alias for SegmentReader::section_bytes.
    pub fn section(&self, name: &str) -> Result<Vec<u8>> {
        self.section_bytes(name)
    }

    /// Alias for SegmentReader::section_bytes.
    pub fn get_section(&self, name: &str) -> Result<Vec<u8>> {
        self.section_bytes(name)
    }

    fn load_disk_meta(&self) -> Result<SegmentDiskMeta> {
        Ok(serde_json::from_slice(&self.store.get(&format!(
            "{}/meta.json",
            segment_prefix(&self.namespace, &self.meta.id)
        ))?)?)
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct SegmentDiskMeta {
    id: String,
    doc_count: usize,
    first_wal_seq: u64,
    last_wal_seq: u64,
    docs: FileMeta,
    sections: Vec<FileMeta>,
}

#[derive(Debug, Serialize, Deserialize)]
struct FileMeta {
    name: String,
    size: u64,
    checksum: u32,
}

impl FileMeta {
    fn for_bytes(name: impl Into<String>, bytes: &[u8]) -> Self {
        Self {
            name: name.into(),
            size: bytes.len() as u64,
            checksum: checksum(bytes),
        }
    }
}

fn validate_section_name(name: &str) -> Result<()> {
    if name.is_empty()
        || name == "."
        || name == ".."
        || name == "docs"
        || name == "meta"
        || name == "docs.bin"
        || name == "meta.json"
        || name.contains('/')
        || name.contains('\\')
    {
        return Err(SegmentError::InvalidSectionName(name.to_owned()));
    }
    Ok(())
}

fn verify_file(expected: &FileMeta, bytes: &[u8]) -> Result<()> {
    let actual_size = bytes.len() as u64;
    if expected.size != actual_size {
        return Err(SegmentError::SizeMismatch {
            name: expected.name.clone(),
            expected: expected.size,
            actual: actual_size,
        });
    }
    let actual_checksum = checksum(bytes);
    if expected.checksum != actual_checksum {
        return Err(SegmentError::ChecksumMismatch {
            name: expected.name.clone(),
            expected: expected.checksum,
            actual: actual_checksum,
        });
    }
    Ok(())
}

fn checksum(bytes: &[u8]) -> u32 {
    let mut hasher = Hasher::new();
    hasher.update(bytes);
    hasher.finalize()
}

fn segment_id(range: WalSeqRange) -> String {
    format!("seg-{}-{}", range.first, range.last)
}

fn segment_prefix(namespace: &str, segment_id: &str) -> String {
    format!("ns/{namespace}/segments/{segment_id}")
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::fs;

    use tempfile::tempdir;

    use super::{SegmentBuilder, SegmentError, SegmentReader};
    use crate::store::LocalDirStore;
    use crate::{AttrValue, Doc};

    fn docs() -> Vec<Doc> {
        let mut attributes = BTreeMap::new();
        attributes.insert("title".to_owned(), AttrValue::String("Hello".to_owned()));
        attributes.insert("rank".to_owned(), AttrValue::Int(7));
        vec![
            Doc {
                id: "a".to_owned(),
                vector: Some(vec![1.0, 2.0]),
                attributes,
            },
            Doc {
                id: "b".to_owned(),
                vector: None,
                attributes: BTreeMap::new(),
            },
        ]
    }

    #[test]
    fn build_read_roundtrip() {
        let root = tempdir().expect("tempdir");
        let store = LocalDirStore::new(root.path()).expect("store");
        let mut sections = BTreeMap::new();
        sections.insert("text".to_owned(), vec![1, 2, 3]);
        sections.insert("vector".to_owned(), vec![4, 5]);

        let meta = SegmentBuilder::new(&store, "demo", 10..=12)
            .build(docs(), sections.clone())
            .expect("build");
        assert_eq!(meta.id, "seg-10-12");
        assert_eq!(meta.doc_count, 2);
        assert_eq!(meta.first_wal_seq, 10);
        assert_eq!(meta.last_wal_seq, 12);
        assert_eq!(meta.sections, vec!["text", "vector"]);

        let reader = SegmentReader::open(&store, "demo", &meta.id).expect("open");
        assert_eq!(reader.meta(), &meta);
        assert_eq!(reader.documents().count(), 2);
        assert_eq!(reader.get("a"), Some(&docs()[0]));
        assert_eq!(reader.get("missing"), None);
        assert_eq!(reader.section_bytes("text").expect("text"), vec![1, 2, 3]);
        assert_eq!(reader.section("vector").expect("vector"), vec![4, 5]);
    }

    #[test]
    fn detects_corrupt_section() {
        let root = tempdir().expect("tempdir");
        let store = LocalDirStore::new(root.path()).expect("store");
        let meta = SegmentBuilder::new(&store, "demo", (1, 1))
            .build(docs(), [("text".to_owned(), vec![1, 2, 3])])
            .expect("build");
        let path = root
            .path()
            .join("ns/demo/segments")
            .join(&meta.id)
            .join("text.bin");
        let mut bytes = fs::read(&path).expect("read section");
        bytes[0] ^= 1;
        fs::write(path, bytes).expect("corrupt section");

        let reader = SegmentReader::open(&store, "demo", &meta.id).expect("open");
        assert!(matches!(
            reader.section_bytes("text"),
            Err(SegmentError::ChecksumMismatch { name, .. }) if name == "text"
        ));
    }

    #[test]
    fn detects_corrupt_docs_on_open() {
        let root = tempdir().expect("tempdir");
        let store = LocalDirStore::new(root.path()).expect("store");
        let meta = SegmentBuilder::new(&store, "demo", (1, 1))
            .build_docs(docs())
            .expect("build");
        let path = root
            .path()
            .join("ns/demo/segments")
            .join(&meta.id)
            .join("docs.bin");
        let mut bytes = fs::read(&path).expect("read docs");
        bytes[0] ^= 1;
        fs::write(path, bytes).expect("corrupt docs");

        assert!(matches!(
            SegmentReader::open(&store, "demo", &meta.id),
            Err(SegmentError::ChecksumMismatch { name, .. }) if name == "docs"
        ));
    }

    #[test]
    fn detects_swapped_section_bytes() {
        let root = tempdir().expect("tempdir");
        let store = LocalDirStore::new(root.path()).expect("store");
        let meta = SegmentBuilder::new(&store, "demo", (1, 1))
            .build(docs(), [("text", vec![1, 2, 3, 4])])
            .expect("build");
        let path = root
            .path()
            .join("ns/demo/segments")
            .join(&meta.id)
            .join("text.bin");
        let mut bytes = fs::read(&path).expect("read section");
        bytes.swap(0, 1);
        fs::write(path, bytes).expect("swap section bytes");

        let reader = SegmentReader::open(&store, "demo", &meta.id).expect("open");
        assert!(matches!(
            reader.section_bytes("text"),
            Err(SegmentError::ChecksumMismatch { name, .. }) if name == "text"
        ));
    }

    #[test]
    fn rejects_empty_batch_and_reserved_sections() {
        let root = tempdir().expect("tempdir");
        let store = LocalDirStore::new(root.path()).expect("store");
        let empty = SegmentBuilder::new(&store, "demo", (1, 1)).build_docs(Vec::new());
        assert!(matches!(empty, Err(SegmentError::InvalidSegment(_))));

        for name in ["docs", "meta", "docs.bin", "meta.json"] {
            let result = SegmentBuilder::new(&store, "demo", (1, 1))
                .build(docs(), [(name.to_owned(), Vec::new())]);
            assert!(matches!(result, Err(SegmentError::InvalidSectionName(_))));
        }
    }
}

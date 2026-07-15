//! Write-ahead log and namespace manifest serialization.

use serde::{Deserialize, Serialize};

use crate::store::ObjectStore;
use crate::{Doc, Result};

/// A durable batch of document upserts and deletes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WalBatch {
    /// The monotonically increasing WAL sequence number.
    pub seq: u64,
    /// Documents to insert or replace.
    pub upserts: Vec<Doc>,
    /// IDs to remove from the namespace.
    pub deletes: Vec<String>,
}

impl WalBatch {
    /// Serialize this batch using bincode.
    pub fn encode(&self) -> Result<Vec<u8>> {
        Ok(bincode::serialize(self)?)
    }

    /// Deserialize a batch encoded by [`WalBatch::encode`].
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        Ok(bincode::deserialize(bytes)?)
    }
}

/// Return the object-store key for a namespace WAL sequence.
///
/// The sequence is zero-padded to 20 decimal digits, which is wide enough for
/// every `u64` and preserves lexical ordering.
pub fn wal_key(namespace: &str, seq: u64) -> String {
    format!("ns/{namespace}/wal/{seq:020}.wal")
}

/// Metadata for one persisted index segment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SegmentMeta {
    /// The segment's stable identifier.
    pub id: String,
    /// The number of documents represented by the segment.
    pub doc_count: usize,
}

/// The atomically swapped namespace manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    /// The namespace vector dimension, once established.
    pub vector_dim: Option<usize>,
    /// Attribute names indexed for full-text search.
    pub full_text_fields: Vec<String>,
    /// Persisted index segments in this namespace.
    pub segments: Vec<SegmentMeta>,
    /// The highest WAL sequence included by this manifest.
    pub last_wal_seq: u64,
}

impl Manifest {
    /// Load a namespace manifest from an object store.
    pub fn load(store: &dyn ObjectStore, namespace: &str) -> Result<Self> {
        let bytes = store.get(&manifest_key(namespace))?;
        Ok(serde_json::from_slice(&bytes)?)
    }

    /// Store this namespace manifest as human-readable JSON.
    pub fn store(&self, store: &dyn ObjectStore, namespace: &str) -> Result<()> {
        let bytes = serde_json::to_vec_pretty(self)?;
        store.put(&manifest_key(namespace), &bytes)
    }
}

/// Return the object-store key for a namespace manifest.
pub fn manifest_key(namespace: &str) -> String {
    format!("ns/{namespace}/MANIFEST.json")
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use tempfile::tempdir;

    use super::{manifest_key, Manifest, SegmentMeta, WalBatch};
    use crate::store::{LocalDirStore, ObjectStore};
    use crate::{AttrValue, Doc};

    #[test]
    fn wal_roundtrip() {
        let mut attributes = BTreeMap::new();
        attributes.insert("active".to_owned(), AttrValue::Bool(true));
        attributes.insert("rank".to_owned(), AttrValue::Int(7));
        let batch = WalBatch {
            seq: 42,
            upserts: vec![Doc {
                id: "doc-1".to_owned(),
                vector: Some(vec![f32::NAN, -0.0, f32::INFINITY, f32::NEG_INFINITY]),
                attributes,
            }],
            deletes: vec!["doc-0".to_owned()],
        };

        let encoded = batch.encode().expect("encode");
        let decoded = WalBatch::decode(&encoded).expect("decode");
        assert_eq!(decoded.seq, batch.seq);
        assert_eq!(decoded.deletes, batch.deletes);
        assert_eq!(decoded.upserts[0].id, batch.upserts[0].id);
        assert_eq!(
            decoded.upserts[0]
                .vector
                .as_ref()
                .expect("decoded vector")
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            batch.upserts[0]
                .vector
                .as_ref()
                .expect("original vector")
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            decoded.upserts[0].attributes["active"],
            AttrValue::Bool(true)
        );
        assert_eq!(decoded.upserts[0].attributes["rank"], AttrValue::Int(7));
        assert_eq!(
            super::wal_key("demo", 42),
            "ns/demo/wal/00000000000000000042.wal"
        );
        assert_eq!(
            super::wal_key("demo", u64::MAX),
            "ns/demo/wal/18446744073709551615.wal"
        );
    }

    #[test]
    fn attr_value_roundtrip_preserves_edge_float_bits() {
        let mut attributes = BTreeMap::new();
        attributes.insert("string".to_owned(), AttrValue::String("value".to_owned()));
        attributes.insert("float".to_owned(), AttrValue::Float(-0.0));
        attributes.insert(
            "float-nan".to_owned(),
            AttrValue::Float(f64::from_bits(0x7ff8_0000_0000_0001)),
        );
        attributes.insert("float-infinity".to_owned(), AttrValue::Float(f64::INFINITY));
        attributes.insert(
            "strings".to_owned(),
            AttrValue::StringList(vec!["one".to_owned(), "two".to_owned()]),
        );
        let batch = WalBatch {
            seq: 1,
            upserts: vec![Doc {
                id: "edge-values".to_owned(),
                vector: None,
                attributes,
            }],
            deletes: Vec::new(),
        };

        let decoded = WalBatch::decode(&batch.encode().expect("encode")).expect("decode");
        let decoded_attributes = &decoded.upserts[0].attributes;
        assert_eq!(
            decoded_attributes["string"],
            AttrValue::String("value".to_owned())
        );
        assert_eq!(
            decoded_attributes["strings"],
            AttrValue::StringList(vec!["one".to_owned(), "two".to_owned()])
        );
        assert_eq!(
            match &decoded_attributes["float"] {
                AttrValue::Float(value) => value.to_bits(),
                value => panic!("expected float, got {value:?}"),
            },
            (-0.0f64).to_bits()
        );
        assert_eq!(
            match &decoded_attributes["float-nan"] {
                AttrValue::Float(value) => value.to_bits(),
                value => panic!("expected float, got {value:?}"),
            },
            0x7ff8_0000_0000_0001
        );
        assert_eq!(
            decoded_attributes["float-infinity"],
            AttrValue::Float(f64::INFINITY)
        );
    }

    #[test]
    fn manifest_load_and_swap_roundtrip() {
        let root = tempdir().expect("tempdir");
        let store = LocalDirStore::new(root.path()).expect("store");
        let first = Manifest {
            vector_dim: Some(3),
            full_text_fields: vec!["title".to_owned()],
            segments: vec![SegmentMeta {
                id: "segment-1".to_owned(),
                doc_count: 10,
            }],
            last_wal_seq: 4,
        };
        first.store(&store, "demo").expect("store manifest");
        assert!(
            String::from_utf8(store.get(&manifest_key("demo")).expect("get manifest"))
                .expect("utf8")
                .contains("\n")
        );
        assert_eq!(
            Manifest::load(&store, "demo").expect("load manifest"),
            first
        );

        let second = Manifest {
            last_wal_seq: 5,
            ..first.clone()
        };
        second.store(&store, "demo").expect("swap manifest");
        assert_eq!(
            Manifest::load(&store, "demo").expect("load swapped manifest"),
            second
        );
    }
}

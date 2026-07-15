//! Exact, in-memory vector search.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::index::sort_scores;
use crate::Result;

/// An index that searches vectors and returns cosine-similarity rankings.
pub trait VectorIndex: Sized {
    /// Build an index from document IDs and vectors.
    fn build<I, D, V>(documents: I) -> Self
    where
        I: IntoIterator<Item = (D, V)>,
        D: Into<String>,
        V: Into<Vec<f32>>;

    /// Search for the nearest vectors using cosine similarity.
    fn search(&self, query: &[f32], top_k: usize) -> Vec<(String, f32)>;

    /// Search while restricting results to IDs in `allowed` when supplied.
    fn search_filtered(
        &self,
        query: &[f32],
        top_k: usize,
        allowed: Option<&HashSet<String>>,
    ) -> Vec<(String, f32)>;

    /// Serialize the index into an opaque byte blob.
    fn to_bytes(&self) -> Result<Vec<u8>>;

    /// Deserialize an index from an opaque byte blob.
    fn from_bytes(bytes: &[u8]) -> Result<Self>;
}

/// A brute-force vector index suitable for the v0 implementation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExactScan {
    vectors: std::collections::BTreeMap<String, Vec<f32>>,
}

impl ExactScan {
    /// Build an exact-scan index. A repeated ID replaces its earlier vector.
    pub fn build<I, D, V>(documents: I) -> Self
    where
        I: IntoIterator<Item = (D, V)>,
        D: Into<String>,
        V: Into<Vec<f32>>,
    {
        Self {
            vectors: documents
                .into_iter()
                .map(|(doc_id, vector)| (doc_id.into(), vector.into()))
                .collect(),
        }
    }

    /// Search by cosine similarity, sorting by score descending and ID
    /// ascending. A query with a different dimension matches no documents.
    /// Zero-norm or non-finite vectors have a score of zero rather than
    /// producing a NaN result.
    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<(String, f32)> {
        self.search_filtered(query, top_k, None)
    }

    /// Search by cosine similarity, considering only IDs in `allowed` when
    /// supplied. Filtering happens before `top_k` truncation.
    pub fn search_filtered(
        &self,
        query: &[f32],
        top_k: usize,
        allowed: Option<&HashSet<String>>,
    ) -> Vec<(String, f32)> {
        if top_k == 0 {
            return Vec::new();
        }

        let mut results: Vec<_> = self
            .vectors
            .iter()
            .filter(|(_, vector)| vector.len() == query.len())
            .filter(|(doc_id, _)| allowed.is_none_or(|allowed| allowed.contains(*doc_id)))
            .map(|(doc_id, vector)| (doc_id.clone(), cosine_similarity(query, vector)))
            .collect();
        sort_scores(&mut results);
        results.truncate(top_k);
        results
    }

    /// Serialize this index using bincode.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        Ok(bincode::serialize(self)?)
    }

    /// Deserialize an index encoded by [`ExactScan::to_bytes`].
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        Ok(bincode::deserialize(bytes)?)
    }
}

impl VectorIndex for ExactScan {
    fn build<I, D, V>(documents: I) -> Self
    where
        I: IntoIterator<Item = (D, V)>,
        D: Into<String>,
        V: Into<Vec<f32>>,
    {
        Self::build(documents)
    }

    fn search(&self, query: &[f32], top_k: usize) -> Vec<(String, f32)> {
        Self::search(self, query, top_k)
    }

    fn search_filtered(
        &self,
        query: &[f32],
        top_k: usize,
        allowed: Option<&HashSet<String>>,
    ) -> Vec<(String, f32)> {
        Self::search_filtered(self, query, top_k, allowed)
    }

    fn to_bytes(&self) -> Result<Vec<u8>> {
        Self::to_bytes(self)
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self> {
        Self::from_bytes(bytes)
    }
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
    let mut dot = 0.0f64;
    let mut left_norm = 0.0f64;
    let mut right_norm = 0.0f64;
    for (&left_value, &right_value) in left.iter().zip(right) {
        if !left_value.is_finite() || !right_value.is_finite() {
            return 0.0;
        }
        let left_value = f64::from(left_value);
        let right_value = f64::from(right_value);
        dot += left_value * right_value;
        left_norm += left_value * left_value;
        right_norm += right_value * right_value;
    }
    if left_norm == 0.0 || right_norm == 0.0 {
        0.0
    } else {
        (dot / (left_norm.sqrt() * right_norm.sqrt())) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::ExactScan;

    #[test]
    fn cosine_scores_match_hand_computation() {
        let index = ExactScan::build(vec![
            ("diagonal", vec![1.0, 1.0]),
            ("x", vec![1.0, 0.0]),
            ("zero", vec![0.0, 0.0]),
        ]);
        let results = index.search(&[1.0, 0.0], 3);
        assert_eq!(results[0].0, "x");
        assert!((results[0].1 - 1.0).abs() < f32::EPSILON);
        assert_eq!(results[1].0, "diagonal");
        assert!((results[1].1 - 1.0 / 2.0f32.sqrt()).abs() < 1e-6);
        assert_eq!(results[2], ("zero".to_owned(), 0.0));
    }

    #[test]
    fn ties_are_deterministic_and_mismatched_dimensions_are_skipped() {
        let index = ExactScan::build(vec![
            ("b", vec![1.0, 0.0]),
            ("a", vec![1.0, 0.0]),
            ("wrong", vec![1.0]),
        ]);
        assert_eq!(
            index.search(&[1.0, 0.0], 3),
            vec![("a".to_owned(), 1.0), ("b".to_owned(), 1.0)]
        );
    }

    #[test]
    fn filtering_happens_before_top_k_truncation() {
        let index = ExactScan::build(vec![
            ("best", vec![1.0, 0.0]),
            ("allowed-a", vec![0.9, 0.1]),
            ("allowed-b", vec![0.8, 0.2]),
        ]);
        let allowed = ["allowed-a".to_owned(), "allowed-b".to_owned()]
            .into_iter()
            .collect();
        assert_eq!(
            index.search_filtered(&[1.0, 0.0], 2, Some(&allowed)),
            vec![
                ("allowed-a".to_owned(), 0.9938837),
                ("allowed-b".to_owned(), 0.9701425),
            ]
        );
    }

    #[test]
    fn serialization_preserves_search_results() {
        let index = ExactScan::build(vec![("a", vec![1.0, 2.0]), ("b", vec![2.0, 1.0])]);
        let decoded = ExactScan::from_bytes(&index.to_bytes().expect("encode")).expect("decode");
        assert_eq!(decoded.search(&[1.0, 2.0], 2), index.search(&[1.0, 2.0], 2));
    }
}

//! In-memory inverted index with BM25 ranking.

use std::collections::{BTreeMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::index::sort_scores;
use crate::Result;

const K1: f64 = 1.2;
const B: f64 = 0.75;

/// BM25 corpus statistics that can be merged across independent indexes.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TextStats {
    /// Number of documents in the corpus.
    pub doc_count: usize,
    /// Sum of token counts across all documents in the corpus.
    pub total_len: usize,
    /// Number of documents containing each term.
    pub doc_freq: BTreeMap<String, usize>,
}

impl TextStats {
    /// Merge statistics from another disjoint corpus into this corpus.
    pub fn merge(&mut self, other: &Self) {
        self.doc_count = self.doc_count.saturating_add(other.doc_count);
        self.total_len = self.total_len.saturating_add(other.total_len);
        for (term, &frequency) in &other.doc_freq {
            let entry = self.doc_freq.entry(term.clone()).or_default();
            *entry = entry.saturating_add(frequency);
        }
    }

    /// Return the corpus-average document length.
    pub fn avgdl(&self) -> f64 {
        if self.doc_count == 0 {
            0.0
        } else {
            self.total_len as f64 / self.doc_count as f64
        }
    }
}

/// A tokenized inverted index using BM25 (k1 = 1.2, b = 0.75).
///
/// Each input pair is a full-text field. Multiple pairs with the same document
/// ID are concatenated with a space before tokenization, so term frequency and
/// document length span all full-text fields for that document.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TextIndex {
    postings: BTreeMap<String, BTreeMap<String, u32>>,
    doc_lengths: BTreeMap<String, usize>,
    avgdl: f64,
}

impl TextIndex {
    /// Build an inverted index from document IDs and full-text field values.
    /// A repeated document ID contributes another concatenated field.
    pub fn build<I, D, T>(documents: I) -> Self
    where
        I: IntoIterator<Item = (D, T)>,
        D: Into<String>,
        T: Into<String>,
    {
        let mut fields = BTreeMap::<String, String>::new();
        for (doc_id, field_text) in documents {
            let doc_id = doc_id.into();
            let field_text = field_text.into();
            let combined = fields.entry(doc_id).or_default();
            if !combined.is_empty() && !field_text.is_empty() {
                combined.push(' ');
            }
            combined.push_str(&field_text);
        }

        let mut postings = BTreeMap::<String, BTreeMap<String, u32>>::new();
        let mut doc_lengths = BTreeMap::new();
        for (doc_id, text) in fields {
            let mut length = 0usize;
            let mut term_frequencies = BTreeMap::<String, u32>::new();
            for token in tokenize(&text) {
                length += 1;
                let frequency = term_frequencies.entry(token).or_default();
                *frequency = frequency.saturating_add(1);
            }
            for (term, frequency) in term_frequencies {
                postings
                    .entry(term)
                    .or_default()
                    .insert(doc_id.clone(), frequency);
            }
            doc_lengths.insert(doc_id, length);
        }

        let avgdl = if doc_lengths.is_empty() {
            0.0
        } else {
            doc_lengths.values().sum::<usize>() as f64 / doc_lengths.len() as f64
        };
        Self {
            postings,
            doc_lengths,
            avgdl,
        }
    }

    /// Search tokenized query text with BM25, sorting by score descending and
    /// document ID ascending. Documents without a query term are omitted.
    pub fn search(&self, query_text: &str, top_k: usize) -> Vec<(String, f32)> {
        self.search_filtered(query_text, top_k, None)
    }

    /// Search with BM25 corpus statistics supplied by the caller. This is the
    /// cross-segment-safe API: use merged statistics for all segments and the
    /// memtable, rather than each segment's local statistics.
    pub fn search_with_stats(
        &self,
        query_text: &str,
        top_k: usize,
        stats: &TextStats,
    ) -> Vec<(String, f32)> {
        self.search_with_stats_filtered(query_text, top_k, stats, None)
    }

    /// Search with caller-supplied corpus statistics and optional candidate
    /// restriction. Filtering occurs before `top_k` truncation.
    pub fn search_with_stats_filtered(
        &self,
        query_text: &str,
        top_k: usize,
        stats: &TextStats,
        allowed: Option<&HashSet<String>>,
    ) -> Vec<(String, f32)> {
        let avgdl = stats.avgdl();
        if top_k == 0 || self.doc_lengths.is_empty() || avgdl == 0.0 || stats.doc_count == 0 {
            return Vec::new();
        }

        let document_count = stats.doc_count as f64;
        let mut scores = BTreeMap::<String, f32>::new();
        for term in tokenize(query_text) {
            let Some(posting) = self.postings.get(&term) else {
                continue;
            };
            let document_frequency = stats.doc_freq.get(&term).copied().unwrap_or(0) as f64;
            let idf = bm25_idf(document_count, document_frequency);
            for (doc_id, &term_frequency) in posting {
                if !allowed.is_none_or(|allowed| allowed.contains(doc_id)) {
                    continue;
                }
                let document_length = self.doc_lengths[doc_id] as f64;
                let normalization = K1 * (1.0 - B + B * document_length / avgdl);
                let term_frequency = f64::from(term_frequency);
                let term_score =
                    idf * term_frequency * (K1 + 1.0) / (term_frequency + normalization);
                *scores.entry(doc_id.clone()).or_default() += term_score as f32;
            }
        }

        let mut results: Vec<_> = scores.into_iter().collect();
        sort_scores(&mut results);
        results.truncate(top_k);
        results
    }

    /// Search with this index's local statistics and an optional candidate
    /// restriction. Filtering occurs before `top_k` truncation.
    pub fn search_filtered(
        &self,
        query_text: &str,
        top_k: usize,
        allowed: Option<&HashSet<String>>,
    ) -> Vec<(String, f32)> {
        let stats = self.stats();
        self.search_with_stats_filtered(query_text, top_k, &stats, allowed)
    }

    /// Return mergeable corpus statistics for all documents in this index.
    pub fn stats(&self) -> TextStats {
        self.stats_filtered(None)
    }

    /// Return mergeable corpus statistics for the selected documents. Pass a
    /// segment's live-ID set when aggregating segments that may contain
    /// superseded versions; `None` includes every document in this index.
    /// The inputs to [`TextStats::merge`] must be deduplicated across indexes.
    pub fn stats_filtered(&self, allowed: Option<&HashSet<String>>) -> TextStats {
        let is_allowed = |doc_id: &String| allowed.is_none_or(|allowed| allowed.contains(doc_id));
        let doc_count = self
            .doc_lengths
            .keys()
            .filter(|doc_id| is_allowed(doc_id))
            .count();
        let total_len = self
            .doc_lengths
            .iter()
            .filter(|(doc_id, _)| is_allowed(doc_id))
            .map(|(_, length)| length)
            .sum();
        let doc_freq = self
            .postings
            .iter()
            .filter_map(|(term, posting)| {
                let frequency = posting.keys().filter(|doc_id| is_allowed(doc_id)).count();
                (frequency > 0).then(|| (term.clone(), frequency))
            })
            .collect();
        TextStats {
            doc_count,
            total_len,
            doc_freq,
        }
    }

    /// Serialize this index using bincode.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        Ok(bincode::serialize(self)?)
    }

    /// Deserialize an index encoded by [`TextIndex::to_bytes`].
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        Ok(bincode::deserialize(bytes)?)
    }
}

/// Compute Lucene's non-negative BM25 IDF variant. The `+1` inside the
/// logarithm keeps IDF non-negative even when document frequency exceeds half
/// of the corpus document count.
fn bm25_idf(document_count: f64, document_frequency: f64) -> f64 {
    (1.0 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)).ln()
}

/// Tokenize full-text input by lowercasing and splitting on non-alphanumeric
/// characters. Empty tokens are discarded.
pub fn tokenize(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split(|character: char| !character.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(str::to_lowercase)
}

/// Compatibility name for callers that refer to the implementation as an
/// inverted index.
pub type InvertedIndex = TextIndex;

#[cfg(test)]
mod tests {
    use super::TextIndex;

    #[test]
    fn bm25_scores_match_hand_computation() {
        let index = TextIndex::build(vec![
            ("doc-1", "the quick brown fox"),
            ("doc-2", "the quick"),
        ]);
        let results = index.search("quick fox", 2);
        let quick_idf = (1.0f64 + 0.5 / 2.5).ln();
        let fox_idf = (1.0f64 + 1.5 / 1.5).ln();
        let expected_doc_1 = ((quick_idf + fox_idf) * (2.2 / 2.5)) as f32;
        let expected_doc_2 = (quick_idf * (2.2 / 1.9)) as f32;
        assert_eq!(results[0].0, "doc-1");
        assert!((results[0].1 - expected_doc_1).abs() < 1e-6);
        assert_eq!(results[1].0, "doc-2");
        assert!((results[1].1 - expected_doc_2).abs() < 1e-6);
    }

    #[test]
    fn repeated_fields_are_concatenated_and_ties_break_by_id() {
        let index = TextIndex::build(vec![("b", "alpha"), ("a", "alpha"), ("a", "beta")]);
        let results = index.search("beta", 2);
        assert_eq!(results[0].0, "a");
        assert_eq!(index.search("missing", 2), Vec::<(String, f32)>::new());
    }

    #[test]
    fn merged_stats_make_scores_comparable_across_segments() {
        let first = TextIndex::build(vec![("first", "same")]);
        let second = TextIndex::build(vec![("second", "same")]);
        let mut stats = first.stats();
        stats.merge(&second.stats());

        let first_results = first.search_with_stats("same", 1, &stats);
        let second_results = second.search_with_stats("same", 1, &stats);
        assert_eq!(first_results[0].1, second_results[0].1);
        assert!((first_results[0].1 - (1.2f32).ln()).abs() < 1e-6);
    }

    #[test]
    fn filtered_stats_deduplicate_overlapping_segments() {
        let old_segment = TextIndex::build(vec![("doc", "stale"), ("old-only", "shared")]);
        let new_segment = TextIndex::build(vec![("doc", "shared"), ("new-only", "other")]);
        let old_live = ["old-only".to_owned()].into_iter().collect();
        let new_live = ["doc".to_owned(), "new-only".to_owned()]
            .into_iter()
            .collect();

        let mut merged = old_segment.stats_filtered(Some(&old_live));
        merged.merge(&new_segment.stats_filtered(Some(&new_live)));
        assert_eq!(merged.doc_count, 3);
        assert_eq!(merged.total_len, 3);
        assert_eq!(merged.doc_freq["shared"], 2);

        let logical_corpus = TextIndex::build(vec![
            ("old-only", "shared"),
            ("doc", "shared"),
            ("new-only", "other"),
        ]);
        let segment_results = new_segment.search_with_stats("shared", 1, &merged);
        let corpus_results = logical_corpus.search("shared", 1);
        assert_eq!(segment_results, corpus_results);
    }

    #[test]
    fn filtering_happens_before_top_k_truncation() {
        let index = TextIndex::build(vec![
            ("best", "common common"),
            ("allowed-a", "common"),
            ("allowed-b", "common"),
        ]);
        let allowed = ["allowed-a".to_owned(), "allowed-b".to_owned()]
            .into_iter()
            .collect();
        let results = index.search_filtered("common", 2, Some(&allowed));
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|(doc_id, _)| allowed.contains(doc_id)));
    }

    #[test]
    fn common_terms_have_non_negative_documented_idf() {
        let index = TextIndex::build(vec![("a", "common"), ("b", "common"), ("c", "other")]);
        let results = index.search("common", 2);
        let expected = (1.0f32 + (3.0 - 2.0 + 0.5) / (2.0 + 0.5)).ln();
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|(_, score)| *score >= 0.0));
        assert!((results[0].1 - expected).abs() < 1e-6);
    }

    #[test]
    fn serialization_preserves_search_results() {
        let index = TextIndex::build(vec![("a", "Rust systems"), ("b", "systems search")]);
        let decoded = TextIndex::from_bytes(&index.to_bytes().expect("encode")).expect("decode");
        assert_eq!(decoded.search("systems", 2), index.search("systems", 2));
    }
}

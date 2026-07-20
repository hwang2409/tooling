//! In-memory indexes used by the v0 query engine.

use std::collections::{BTreeMap, BTreeSet};

pub mod filter;
pub mod hnsw;
pub mod text;
pub mod vector;

/// The default reciprocal-rank-fusion constant.
pub const DEFAULT_RRF_K: u32 = 60;

/// Combine ranked document lists using reciprocal rank fusion.
///
/// Each ranking contributes at most once per document. Ranks are one-based,
/// and ties in the resulting score are resolved by ascending document ID.
pub fn rrf(rankings: &[Vec<String>], k: u32) -> Vec<(String, f32)> {
    let mut scores = BTreeMap::<String, f32>::new();
    for ranking in rankings {
        let mut seen = BTreeSet::new();
        for (zero_based_rank, doc_id) in ranking.iter().enumerate() {
            if !seen.insert(doc_id) {
                continue;
            }
            let rank = zero_based_rank as u32 + 1;
            let contribution = 1.0 / (k.saturating_add(rank) as f32);
            *scores.entry(doc_id.clone()).or_default() += contribution;
        }
    }

    let mut results: Vec<_> = scores.into_iter().collect();
    sort_scores(&mut results);
    results
}

/// Combine ranked document lists using the standard v0 RRF constant.
pub fn rrf_default(rankings: &[Vec<String>]) -> Vec<(String, f32)> {
    rrf(rankings, DEFAULT_RRF_K)
}

pub(crate) fn sort_scores(results: &mut Vec<(String, f32)>) {
    // Non-finite values cannot participate in a meaningful ranking. Drop
    // them before sorting so they can never displace finite scores.
    let finite_len = results
        .iter()
        .filter(|(_, score)| score.is_finite())
        .count();
    results.retain(|(_, score)| score.is_finite());
    debug_assert_eq!(results.len(), finite_len);
    results.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
}

#[cfg(test)]
mod tests {
    use super::{rrf, rrf_default, DEFAULT_RRF_K};

    #[test]
    fn rrf_combines_rankings_and_breaks_ties_by_id() {
        let rankings = vec![
            vec!["b".to_owned(), "a".to_owned()],
            vec!["a".to_owned(), "b".to_owned()],
        ];
        let results = rrf(&rankings, 0);
        assert_eq!(results[0].0, "a");
        assert_eq!(results[1].0, "b");
        assert!((results[0].1 - 1.5).abs() < f32::EPSILON);
        assert!((results[1].1 - 1.5).abs() < f32::EPSILON);
    }

    #[test]
    fn rrf_default_uses_sixty() {
        let results = rrf_default(&[vec!["doc".to_owned()]]);
        assert_eq!(
            results,
            vec![("doc".to_owned(), 1.0 / (DEFAULT_RRF_K + 1) as f32)]
        );
    }

    #[test]
    fn non_finite_scores_are_dropped_before_ranking() {
        let mut results = vec![
            ("nan".to_owned(), f32::NAN),
            ("finite".to_owned(), 1.0),
            ("infinity".to_owned(), f32::INFINITY),
        ];
        super::sort_scores(&mut results);
        assert_eq!(results, vec![("finite".to_owned(), 1.0)]);
    }
}

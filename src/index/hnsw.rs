//! Deterministic, hand-rolled HNSW vector search.
//!
//! This implementation intentionally uses the simple "keep the closest M"
//! neighbor rule instead of the paper's diversity heuristic. It keeps the
//! learning surface small and is sufficient for the v1 segment index. During
//! filtered search, rejected nodes remain in the traversal candidate set so
//! they can still route the search, but they are never added to the returned
//! result candidates. This improves filtered recall over post-filtering a
//! truncated unfiltered result, at the cost of the usual approximate-search
//! recall tradeoff when the allowed set is sparse. The index assumes one
//! vector dimension per namespace; the namespace validates that invariant,
//! while this index stores the dimension for constant-time query validation.

use std::cmp::{Ordering, Reverse};
use std::collections::{BTreeMap, BinaryHeap, HashSet};

use serde::{Deserialize, Serialize};

use crate::index::sort_scores;
use crate::index::vector::{cosine_similarity, VectorIndex};
use crate::{Error, Result};

/// The default search breadth for an HNSW query.
pub const DEFAULT_EF_SEARCH: usize = 64;

const DEFAULT_M: usize = 16;
const DEFAULT_M0: usize = 32;
const DEFAULT_EF_CONSTRUCTION: usize = 200;
const MAX_LEVEL: usize = 32;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct HnswParams {
    m: usize,
    m0: usize,
    ef_construction: usize,
    ef_search: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct Node {
    id: String,
    vector: Vec<f32>,
    level: usize,
    neighbors: Vec<Vec<usize>>,
    backbone_prev: Option<usize>,
    backbone_next: Option<usize>,
}

#[derive(Debug, Clone)]
struct Candidate {
    node: usize,
    score: f32,
    id: String,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.node == other.node
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .total_cmp(&other.score)
            .then_with(|| other.id.cmp(&self.id))
    }
}

/// A deterministic hierarchical navigable small-world graph.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Hnsw {
    params: HnswParams,
    nodes: Vec<Node>,
    dimension: Option<usize>,
    entry_point: Option<usize>,
    max_level: usize,
}

impl Hnsw {
    /// Build a deterministic HNSW index. Repeated IDs replace earlier
    /// vectors, matching [`crate::index::vector::ExactScan`].
    pub fn build<I, D, V>(documents: I) -> Self
    where
        I: IntoIterator<Item = (D, V)>,
        D: Into<String>,
        V: Into<Vec<f32>>,
    {
        let documents = documents
            .into_iter()
            .map(|(id, vector)| (id.into(), vector.into()))
            .collect::<BTreeMap<_, _>>();
        let mut index = Self {
            params: HnswParams {
                m: DEFAULT_M,
                m0: DEFAULT_M0,
                ef_construction: DEFAULT_EF_CONSTRUCTION,
                ef_search: DEFAULT_EF_SEARCH,
            },
            nodes: Vec::with_capacity(documents.len()),
            dimension: None,
            entry_point: None,
            max_level: 0,
        };
        for (id, vector) in documents {
            index.insert(id, vector);
        }
        index
    }

    fn insert(&mut self, id: String, vector: Vec<f32>) {
        match self.dimension {
            Some(dimension) if dimension != vector.len() => return,
            None => self.dimension = Some(vector.len()),
            Some(_) => {}
        }
        let level = level_for_id(&id);
        let node = self.nodes.len();
        self.nodes.push(Node {
            id,
            vector,
            level,
            neighbors: (0..=level).map(|_| Vec::new()).collect(),
            backbone_prev: node.checked_sub(1),
            backbone_next: None,
        });

        let Some(mut entry) = self.entry_point else {
            self.entry_point = Some(node);
            self.max_level = level;
            return;
        };

        let previous_max_level = self.max_level;
        for layer in ((level + 1)..=previous_max_level).rev() {
            entry = self.greedy_search(&self.nodes[node].vector, entry, layer);
        }

        for layer in (0..=level.min(previous_max_level)).rev() {
            let beam = self.search_layer(
                &self.nodes[node].vector,
                entry,
                layer,
                self.params.ef_construction,
                None,
            );
            let next_entry = beam.first().copied().unwrap_or(entry);
            let selected = beam
                .into_iter()
                .take(self.max_neighbors(layer))
                .collect::<Vec<_>>();
            for neighbor in selected {
                if !self.nodes[node].neighbors[layer].contains(&neighbor) {
                    self.nodes[node].neighbors[layer].push(neighbor);
                }
                if !self.nodes[neighbor].neighbors[layer].contains(&node) {
                    self.nodes[neighbor].neighbors[layer].push(node);
                }
                self.prune_neighbors(neighbor, layer);
            }
            self.prune_neighbors(node, layer);
            entry = next_entry;
        }

        if let Some(previous) = node.checked_sub(1) {
            if !self.nodes[node].neighbors[0].contains(&previous) {
                self.nodes[node].neighbors[0].push(previous);
            }
            self.prune_neighbors(node, 0);
            self.nodes[previous].backbone_next = Some(node);
            if !self.nodes[previous].neighbors[0].contains(&node) {
                self.nodes[previous].neighbors[0].push(node);
            }
            self.prune_neighbors(previous, 0);
        }

        if level > self.max_level {
            self.max_level = level;
            self.entry_point = Some(node);
        }
    }

    /// Search with a caller-selected traversal breadth.
    pub fn search_with_ef(
        &self,
        query: &[f32],
        top_k: usize,
        ef_search: usize,
    ) -> Vec<(String, f32)> {
        self.search_filtered_with_ef(query, top_k, ef_search, None)
    }

    /// Search with a caller-selected breadth and an allow-list. Disallowed
    /// nodes are traversed as routing nodes but are not result candidates.
    pub fn search_filtered_with_ef(
        &self,
        query: &[f32],
        top_k: usize,
        ef_search: usize,
        allowed: Option<&HashSet<String>>,
    ) -> Vec<(String, f32)> {
        if top_k == 0 {
            return Vec::new();
        }
        if self
            .dimension
            .is_some_and(|dimension| dimension != query.len())
        {
            return Vec::new();
        }
        let Some(entry_point) = self.entry_point else {
            return Vec::new();
        };

        let mut entry = entry_point;
        for level in (1..=self.max_level).rev() {
            entry = self.greedy_search(query, entry, level);
        }

        let ef = ef_search.max(top_k).max(1);
        let candidates = self.search_layer(query, entry, 0, ef, allowed);
        let mut results = candidates
            .into_iter()
            .map(|node| {
                (
                    self.nodes[node].id.clone(),
                    cosine_similarity(query, &self.nodes[node].vector),
                )
            })
            .collect::<Vec<_>>();
        sort_scores(&mut results);
        results.truncate(top_k);
        results
    }

    /// Serialize this index as a self-contained opaque blob.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        self.validate()?;
        Ok(bincode::serialize(self)?)
    }

    /// Deserialize and validate an index encoded by [`Hnsw::to_bytes`].
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let index: Self = bincode::deserialize(bytes)?;
        index.validate()?;
        Ok(index)
    }

    fn greedy_search(&self, query: &[f32], mut current: usize, layer: usize) -> usize {
        loop {
            let mut best_score = cosine_similarity(query, &self.nodes[current].vector);
            let mut best = current;
            for &neighbor in &self.nodes[current].neighbors[layer] {
                let candidate_score = cosine_similarity(query, &self.nodes[neighbor].vector);
                if better_node(
                    candidate_score,
                    &self.nodes[neighbor].id,
                    best_score,
                    &self.nodes[best].id,
                ) {
                    best_score = candidate_score;
                    best = neighbor;
                }
            }
            if best == current {
                return current;
            }
            current = best;
        }
    }

    fn search_layer(
        &self,
        query: &[f32],
        entry: usize,
        layer: usize,
        ef: usize,
        allowed: Option<&HashSet<String>>,
    ) -> Vec<usize> {
        if let Some(allowed) = allowed {
            return self.search_layer_filtered(query, entry, layer, ef, allowed);
        }
        let ef = ef.max(1);
        let mut visited = HashSet::new();
        let entry_candidate = self.candidate(query, entry);
        let mut frontier = BinaryHeap::from([entry_candidate.clone()]);
        let mut results = BinaryHeap::from([Reverse(entry_candidate)]);
        visited.insert(entry);

        while let Some(current) = frontier.pop() {
            if results.len() >= ef
                && current
                    .cmp(&results.peek().expect("non-empty result beam").0)
                    .is_lt()
            {
                break;
            }

            for &neighbor in &self.nodes[current.node].neighbors[layer] {
                if !visited.insert(neighbor) {
                    continue;
                }
                let candidate = self.candidate(query, neighbor);
                let should_add = results.len() < ef
                    || candidate
                        .cmp(&results.peek().expect("non-empty result beam").0)
                        .is_gt();
                if should_add {
                    frontier.push(candidate.clone());
                    results.push(Reverse(candidate));
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }

        candidates_to_nodes(results)
    }

    /// Filtered traversal uses allowed results, rather than the unfiltered
    /// beam, as its termination condition. Disallowed nodes remain routing
    /// candidates. The expansion cap bounds sparse-filter work at eight times
    /// ef_search.
    fn search_layer_filtered(
        &self,
        query: &[f32],
        entry: usize,
        layer: usize,
        ef: usize,
        allowed: &HashSet<String>,
    ) -> Vec<usize> {
        let ef = ef.max(1);
        let expansion_cap = ef.max(1).saturating_mul(8).max(1);
        let mut visited = HashSet::new();
        let entry_candidate = self.candidate(query, entry);
        let mut frontier = BinaryHeap::from([entry_candidate.clone()]);
        let mut results = BinaryHeap::new();
        let mut disallowed_frontier = 0;
        if allowed.contains(&entry_candidate.id) {
            results.push(Reverse(entry_candidate));
        } else {
            disallowed_frontier = 1;
        }
        visited.insert(entry);
        let mut expansions = 0;

        while let Some(current) = frontier.pop() {
            if expansions >= expansion_cap {
                break;
            }
            expansions += 1;
            if !allowed.contains(&current.id) {
                disallowed_frontier -= 1;
            }
            if results.len() >= ef
                && disallowed_frontier == 0
                && current
                    .cmp(&results.peek().expect("non-empty result beam").0)
                    .is_lt()
            {
                break;
            }

            for &neighbor in &self.nodes[current.node].neighbors[layer] {
                if !visited.insert(neighbor) {
                    continue;
                }
                let candidate = self.candidate(query, neighbor);
                if allowed.contains(&candidate.id) {
                    let should_add = results.len() < ef
                        || candidate
                            .cmp(&results.peek().expect("non-empty result beam").0)
                            .is_gt();
                    if should_add {
                        frontier.push(candidate.clone());
                        results.push(Reverse(candidate));
                        if results.len() > ef {
                            results.pop();
                        }
                    }
                } else {
                    frontier.push(candidate);
                    disallowed_frontier += 1;
                }
            }
        }

        candidates_to_nodes(results)
    }

    fn candidate(&self, query: &[f32], node: usize) -> Candidate {
        Candidate {
            node,
            score: cosine_similarity(query, &self.nodes[node].vector),
            id: self.nodes[node].id.clone(),
        }
    }

    fn prune_neighbors(&mut self, node: usize, layer: usize) {
        let query = self.nodes[node].vector.clone();
        let mut neighbors = std::mem::take(&mut self.nodes[node].neighbors[layer]);
        neighbors.dedup();
        let mut required = Vec::new();
        if layer == 0 {
            if let Some(previous) = self.nodes[node].backbone_prev {
                required.push(previous);
            }
            if let Some(next) = self.nodes[node].backbone_next {
                required.push(next);
            }
        }
        for &neighbor in &required {
            if !neighbors.contains(&neighbor) {
                neighbors.push(neighbor);
            }
        }
        neighbors.retain(|neighbor| !required.contains(neighbor));
        neighbors.sort_by(|left, right| compare_node_ids(*left, *right, &query, &self.nodes));
        let remaining = self.max_neighbors(layer).saturating_sub(required.len());
        neighbors.truncate(remaining);
        required.extend(neighbors);
        required.sort_by(|left, right| compare_node_ids(*left, *right, &query, &self.nodes));
        self.nodes[node].neighbors[layer] = required;
    }

    fn max_neighbors(&self, layer: usize) -> usize {
        if layer == 0 {
            self.params.m0
        } else {
            self.params.m
        }
    }

    fn validate(&self) -> Result<()> {
        if self.params.m == 0
            || self.params.m0 == 0
            || self.params.ef_construction == 0
            || self.params.ef_search == 0
        {
            return Err(invalid_index("parameters must be non-zero"));
        }
        if self.dimension.is_none() && !self.nodes.is_empty()
            || self.dimension.is_some_and(|dimension| {
                self.nodes.iter().any(|node| node.vector.len() != dimension)
            })
        {
            return Err(invalid_index(
                "vectors do not share the configured dimension",
            ));
        }
        let mut ids = HashSet::new();
        let mut max_level = 0;
        for (node_index, node) in self.nodes.iter().enumerate() {
            if !ids.insert(&node.id) {
                return Err(invalid_index("duplicate document ID"));
            }
            if node.level > MAX_LEVEL || node.neighbors.len() != node.level + 1 {
                return Err(invalid_index("invalid node level"));
            }
            max_level = max_level.max(node.level);
            for (layer, neighbors) in node.neighbors.iter().enumerate() {
                if neighbors.len() > self.max_neighbors(layer) {
                    return Err(invalid_index("neighbor list exceeds configured M"));
                }
                for &neighbor in neighbors {
                    if neighbor >= self.nodes.len() || neighbor == node_index {
                        return Err(invalid_index("invalid neighbor reference"));
                    }
                    if self.nodes[neighbor].level < layer {
                        return Err(invalid_index("neighbor is missing a referenced layer"));
                    }
                }
            }
            if let Some(previous) = node.backbone_prev {
                if previous.checked_add(1) != Some(node_index)
                    || !node.neighbors[0].contains(&previous)
                {
                    return Err(invalid_index("invalid backbone predecessor"));
                }
            } else if node_index != 0 {
                return Err(invalid_index("missing backbone predecessor"));
            }
            if let Some(next) = node.backbone_next {
                if next != node_index.saturating_add(1)
                    || next >= self.nodes.len()
                    || !node.neighbors[0].contains(&next)
                {
                    return Err(invalid_index("invalid backbone successor"));
                }
            } else if node_index + 1 < self.nodes.len() {
                return Err(invalid_index("missing backbone successor"));
            }
        }
        if self.max_level != max_level {
            return Err(invalid_index("invalid maximum level"));
        }
        match self.entry_point {
            Some(entry)
                if entry < self.nodes.len() && self.nodes[entry].level == self.max_level => {}
            None if self.nodes.is_empty() => {}
            _ => return Err(invalid_index("invalid entry point")),
        }
        if let Some(entry) = self.entry_point {
            let mut reachable = HashSet::new();
            let mut frontier = vec![entry];
            reachable.insert(entry);
            while let Some(node) = frontier.pop() {
                for &neighbor in &self.nodes[node].neighbors[0] {
                    if reachable.insert(neighbor) {
                        frontier.push(neighbor);
                    }
                }
            }
            if reachable.len() != self.nodes.len() {
                return Err(invalid_index("graph is not reachable from entry point"));
            }
        }
        Ok(())
    }
}

impl VectorIndex for Hnsw {
    fn build<I, D, V>(documents: I) -> Self
    where
        I: IntoIterator<Item = (D, V)>,
        D: Into<String>,
        V: Into<Vec<f32>>,
    {
        Self::build(documents)
    }

    fn search(&self, query: &[f32], top_k: usize) -> Vec<(String, f32)> {
        self.search_with_ef(query, top_k, self.params.ef_search)
    }

    fn search_filtered(
        &self,
        query: &[f32],
        top_k: usize,
        allowed: Option<&HashSet<String>>,
    ) -> Vec<(String, f32)> {
        self.search_filtered_with_ef(query, top_k, self.params.ef_search, allowed)
    }

    fn to_bytes(&self) -> Result<Vec<u8>> {
        Self::to_bytes(self)
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self> {
        Self::from_bytes(bytes)
    }
}

fn compare_node_ids(
    left: usize,
    right: usize,
    query: &[f32],
    nodes: &[Node],
) -> std::cmp::Ordering {
    let left_score = cosine_similarity(query, &nodes[left].vector);
    let right_score = cosine_similarity(query, &nodes[right].vector);
    right_score
        .total_cmp(&left_score)
        .then_with(|| nodes[left].id.cmp(&nodes[right].id))
}

fn candidates_to_nodes(results: BinaryHeap<Reverse<Candidate>>) -> Vec<usize> {
    let mut results = results
        .into_iter()
        .map(|Reverse(candidate)| candidate)
        .collect::<Vec<_>>();
    results.sort_by(|left, right| right.cmp(left));
    results
        .into_iter()
        .map(|candidate| candidate.node)
        .collect()
}

fn better_node(left_score: f32, left_id: &str, right_score: f32, right_id: &str) -> bool {
    left_score.total_cmp(&right_score).is_gt()
        || (left_score.total_cmp(&right_score).is_eq() && left_id < right_id)
}

fn level_for_id(id: &str) -> usize {
    let mut hash = 0xcbf29ce484222325_u64 ^ 0x9e3779b97f4a7c15_u64;
    for byte in id.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3_u64);
    }

    let mut level = 0;
    let mut value = hash;
    while level < MAX_LEVEL && value & 0x0f == 0 {
        level += 1;
        value >>= 4;
    }
    level
}

fn invalid_index(message: &str) -> Error {
    Error::Store(format!("invalid HNSW index: {message}"))
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::Hnsw;
    use crate::index::vector::{ExactScan, VectorIndex};

    #[test]
    fn deterministic_build_and_serialization() {
        let documents = vec![
            ("b", vec![0.0, 1.0]),
            ("a", vec![1.0, 0.0]),
            ("c", vec![1.0, 1.0]),
        ];
        let first = Hnsw::build(documents.clone());
        let second = Hnsw::build(documents.into_iter().rev());
        assert_eq!(
            first.to_bytes().expect("encode"),
            second.to_bytes().expect("encode")
        );
    }

    #[test]
    fn serialization_preserves_search_results() {
        let index = Hnsw::build([
            ("a", vec![1.0, 0.0]),
            ("b", vec![0.0, 1.0]),
            ("c", vec![1.0, 1.0]),
        ]);
        let decoded = Hnsw::from_bytes(&index.to_bytes().expect("encode")).expect("decode");
        assert_eq!(decoded.search(&[1.0, 0.0], 3), index.search(&[1.0, 0.0], 3));
    }

    #[test]
    fn roundtrip_256_vector_random_and_identical_inputs() {
        let random = Hnsw::build((0..256).map(|value| {
            (
                format!("doc-{value:03}"),
                (0..8)
                    .map(|dimension| ((value * 17 + dimension * 5) as f32).sin())
                    .collect::<Vec<_>>(),
            )
        }));
        let identical =
            Hnsw::build((0..256).map(|value| (format!("doc-{value:03}"), vec![1.0, 0.0])));
        let decoded_random = Hnsw::from_bytes(&random.to_bytes().expect("encode")).expect("decode");
        let random_query = (0..8)
            .map(|dimension| (dimension as f32 * 0.37).cos())
            .collect::<Vec<_>>();
        assert_eq!(
            decoded_random.search(&random_query, 10),
            random.search(&random_query, 10)
        );

        let decoded_identical =
            Hnsw::from_bytes(&identical.to_bytes().expect("encode")).expect("decode");
        assert_eq!(
            decoded_identical.search(&[1.0, 0.0], 10),
            identical.search(&[1.0, 0.0], 10)
        );
    }

    #[test]
    fn filtered_search_returns_allowed_neighbors_beyond_raw_top_k() {
        let index = Hnsw::build((0..1_000).map(|value| {
            let angle = value as f32 * 0.01;
            (format!("doc-{value:03}"), vec![angle.cos(), angle.sin()])
        }));
        let allowed = HashSet::from(["doc-100".to_owned()]);
        let exact = ExactScan::build((0..1_000).map(|value| {
            let angle = value as f32 * 0.01;
            (format!("doc-{value:03}"), vec![angle.cos(), angle.sin()])
        }));
        assert_eq!(
            index.search_filtered_with_ef(&[1.0, 0.0], 1, 64, Some(&allowed)),
            exact.search_filtered(&[1.0, 0.0], 1, Some(&allowed))
        );

        let all_allowed = (0..1_000)
            .map(|value| format!("doc-{value:03}"))
            .collect::<HashSet<_>>();
        assert_eq!(
            index.search_filtered_with_ef(&[1.0, 0.0], 10, 64, Some(&all_allowed)),
            index.search_with_ef(&[1.0, 0.0], 10, 64)
        );
    }

    #[test]
    fn recall_is_high_on_seeded_vectors() {
        let mut state = 0x1234_5678_9abc_def0_u64;
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            ((state >> 32) as u32) as f32 / u32::MAX as f32 * 2.0 - 1.0
        };
        let documents = (0..2_000)
            .map(|value| {
                (
                    format!("doc-{value:04}"),
                    (0..32).map(|_| next()).collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>();
        let index = Hnsw::build(documents.clone());
        let exact = ExactScan::build(documents);

        let mut hits = 0;
        let mut queries = 0;
        for query_number in 0..20 {
            let query = (0..32)
                .map(|dimension| ((query_number * 37 + dimension * 11) as f32).sin())
                .collect::<Vec<_>>();
            let expected = exact
                .search(&query, 10)
                .into_iter()
                .map(|(id, _)| id)
                .collect::<HashSet<_>>();
            let actual = index
                .search_with_ef(&query, 10, 64)
                .into_iter()
                .map(|(id, _)| id)
                .collect::<HashSet<_>>();
            hits += expected.intersection(&actual).count();
            queries += expected.len();
        }
        assert!(
            hits as f32 / queries as f32 >= 0.9,
            "recall was {hits}/{queries}"
        );
    }
}

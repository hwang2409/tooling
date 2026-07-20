//! Small deterministic corpus generators used by benchmarks and examples.
//!
//! The generators deliberately use a local, fixed-seed PRNG. They never read
//! the clock, operating-system entropy, or process-specific state, so corpus
//! construction is reproducible across benchmark runs.

use std::collections::BTreeMap;

use crate::Doc;

const VECTOR_SEED: u64 = 0x4d595df4d0f33173;
const QUERY_SEED: u64 = 0x9e3779b97f4a7c15;

#[derive(Clone, Copy)]
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 32) as u32
    }

    fn next_f32(&mut self) -> f32 {
        let unit = (self.next_u32() >> 8) as f32 / 16_777_216.0;
        unit * 2.0 - 1.0
    }
}

/// Build a deterministic vector corpus with stable document IDs.
pub fn vector_corpus(doc_count: usize, dimension: usize) -> Vec<Doc> {
    let mut rng = Rng::new(VECTOR_SEED ^ (doc_count as u64).rotate_left(17) ^ dimension as u64);
    (0..doc_count)
        .map(|index| Doc {
            id: format!("doc-{index:06}"),
            vector: Some((0..dimension).map(|_| rng.next_f32()).collect()),
            attributes: BTreeMap::new(),
        })
        .collect()
}

/// Build a deterministic short-document corpus with a stable term mix.
pub fn text_corpus(doc_count: usize) -> Vec<(String, String)> {
    const TERMS: [&str; 8] = [
        "rust", "engine", "vector", "search", "segment", "storage", "query", "index",
    ];
    let mut rng = Rng::new(VECTOR_SEED ^ (doc_count as u64).rotate_left(29));
    (0..doc_count)
        .map(|index| {
            let terms = (0..8)
                .map(|_| TERMS[(rng.next_u32() as usize) % TERMS.len()])
                .collect::<Vec<_>>();
            (format!("doc-{index:06}"), terms.join(" "))
        })
        .collect()
}

/// Derive a repeatable query near a selected corpus vector.
pub fn query_near(corpus: &[Doc], index: usize) -> Vec<f32> {
    let document = &corpus[index % corpus.len()];
    let mut query = document
        .vector
        .as_ref()
        .expect("vector corpus document")
        .clone();
    let mut rng = Rng::new(QUERY_SEED ^ index as u64);
    for value in &mut query {
        *value += rng.next_f32() * 0.005;
    }
    query
}

/// Return deterministic query vectors sampled from a corpus.
pub fn query_set(corpus: &[Doc], count: usize) -> Vec<Vec<f32>> {
    (0..count)
        .map(|index| query_near(corpus, index * 997))
        .collect()
}

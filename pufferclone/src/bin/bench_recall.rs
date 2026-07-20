use std::collections::HashSet;

use pufferclone::index::hnsw::Hnsw;
use pufferclone::index::vector::ExactScan;
use pufferclone::testkit::{query_set, vector_corpus};

fn main() {
    let corpus = vector_corpus(10_000, 128);
    let vectors = corpus.iter().filter_map(|doc| {
        doc.vector
            .as_ref()
            .map(|vector| (doc.id.clone(), vector.clone()))
    });
    let exact = ExactScan::build(vectors.clone());
    let hnsw = Hnsw::build(vectors);
    let queries = query_set(&corpus, 32);

    println!("ef_search\trecall@10\trecall@100");
    for ef_search in [16, 32, 64, 128, 256] {
        let mut recall_at_10 = 0.0;
        let mut recall_at_100 = 0.0;
        for query in &queries {
            let exact_10 = exact.search(query, 10);
            let exact_100 = exact.search(query, 100);
            let approximate_10 = hnsw.search_with_ef(query, 10, ef_search);
            let approximate_100 = hnsw.search_with_ef(query, 100, ef_search);
            recall_at_10 += recall(&exact_10, &approximate_10);
            recall_at_100 += recall(&exact_100, &approximate_100);
        }
        let query_count = queries.len() as f64;
        println!(
            "{ef_search}\t\t{:.3}\t\t{:.3}",
            recall_at_10 / query_count,
            recall_at_100 / query_count
        );
    }
}

fn recall(expected: &[(String, f32)], actual: &[(String, f32)]) -> f64 {
    if expected.is_empty() {
        return 1.0;
    }
    let expected = expected.iter().map(|(id, _)| id).collect::<HashSet<_>>();
    let matches = actual
        .iter()
        .filter(|(id, _)| expected.contains(id))
        .count();
    matches as f64 / expected.len() as f64
}

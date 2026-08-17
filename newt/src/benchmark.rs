//! Helpers shared by the standard benchmark harness and its tests.

/// Return a percentile using the nearest-rank rule on sorted samples.
pub fn nearest_rank_percentile(samples: &[u128], fraction: f64) -> u128 {
    assert!(!samples.is_empty(), "percentile needs at least one sample");
    let rank = (fraction * samples.len() as f64).ceil() as usize;
    let index = rank.saturating_sub(1).min(samples.len() - 1);
    samples[index]
}

#[cfg(test)]
mod tests {
    use super::nearest_rank_percentile;

    #[test]
    fn percentile_uses_nearest_rank() {
        let samples: Vec<u128> = (1..=15).map(|value| value * 10).collect();

        assert_eq!(nearest_rank_percentile(&samples, 0.1), 20);
        assert_eq!(nearest_rank_percentile(&samples, 0.5), 80);
        assert_eq!(nearest_rank_percentile(&samples, 0.9), 140);
    }
}

use std::collections::BTreeSet;

use regex::Regex;

use crate::format::PartitionIndex;
use crate::model::{Matcher, Series, StoreError};

pub(crate) struct CompiledMatcher {
    label: String,
    kind: MatcherKind,
}

enum MatcherKind {
    Exact(String),
    Regex(Regex),
}

pub(crate) fn compile(matchers: &[Matcher]) -> Result<Vec<CompiledMatcher>, StoreError> {
    matchers
        .iter()
        .map(|matcher| match matcher {
            Matcher::Exact { label, value } => Ok(CompiledMatcher {
                label: label.clone(),
                kind: MatcherKind::Exact(value.clone()),
            }),
            Matcher::Regex { label, pattern } => Regex::new(&format!("^(?:{pattern})$"))
                .map(|regex| CompiledMatcher {
                    label: label.clone(),
                    kind: MatcherKind::Regex(regex),
                })
                .map_err(|error| StoreError::InvalidMatcher(error.to_string())),
        })
        .collect()
}

pub(crate) fn matches(series: &Series, matchers: &[CompiledMatcher]) -> bool {
    matchers.iter().all(|matcher| {
        let actual = if matcher.label == "__name__" {
            Some(series.name.as_str())
        } else {
            series.labels.get(&matcher.label).map(String::as_str)
        };
        match (&matcher.kind, actual) {
            (MatcherKind::Exact(expected), Some(actual)) => expected == actual,
            (MatcherKind::Regex(regex), Some(actual)) => regex.is_match(actual),
            _ => false,
        }
    })
}

pub(crate) fn candidates(index: &PartitionIndex, matchers: &[CompiledMatcher]) -> BTreeSet<u64> {
    if matchers.is_empty() {
        return index.offsets.keys().copied().collect();
    }
    let mut result: Option<BTreeSet<u64>> = None;
    for matcher in matchers {
        let mut matching = BTreeSet::new();
        match &matcher.kind {
            MatcherKind::Exact(value) => {
                if let Some(ids) = index.labels.get(&(matcher.label.clone(), value.clone())) {
                    matching.extend(ids);
                }
            }
            MatcherKind::Regex(regex) => {
                for ((label, value), ids) in &index.labels {
                    if label == &matcher.label && regex.is_match(value) {
                        matching.extend(ids);
                    }
                }
            }
        }
        result = Some(match result {
            Some(previous) => previous.intersection(&matching).copied().collect(),
            None => matching,
        });
    }
    result.unwrap_or_default()
}

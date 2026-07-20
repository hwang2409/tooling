use std::collections::{BTreeMap, BTreeSet};
use std::io;

use thiserror::Error;

pub const DEFAULT_PARTITION_DURATION_MS: i64 = 2 * 60 * 60 * 1000;
pub const DEFAULT_OUT_OF_ORDER_TOLERANCE_MS: i64 = 60 * 1000;
pub const DEFAULT_RETENTION_MS: i64 = 30 * 24 * 60 * 60 * 1000;

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct Series {
    pub name: String,
    pub labels: BTreeMap<String, String>,
}

impl Series {
    pub fn new(name: impl Into<String>, labels: BTreeMap<String, String>) -> Self {
        Self {
            name: name.into(),
            labels,
        }
    }

    pub fn from_labels<I, K, V>(name: impl Into<String>, labels: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        Self::new(
            name,
            labels
                .into_iter()
                .map(|(key, value)| (key.into(), value.into()))
                .collect(),
        )
    }

    pub fn canonical(&self) -> String {
        let mut result = String::new();
        append_string(&mut result, &self.name);
        for (name, value) in &self.labels {
            append_string(&mut result, name);
            append_string(&mut result, value);
        }
        result
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sample {
    pub timestamp: i64,
    pub value: f64,
}

impl Sample {
    pub const fn new(timestamp: i64, value: f64) -> Self {
        Self { timestamp, value }
    }

    pub const fn timestamp_millis(self) -> i64 {
        self.timestamp
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SeriesSamples {
    pub series: Series,
    pub samples: Vec<Sample>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Matcher {
    Exact { label: String, value: String },
    Regex { label: String, pattern: String },
}

impl Matcher {
    pub fn exact(label: impl Into<String>, value: impl Into<String>) -> Self {
        Self::Exact {
            label: label.into(),
            value: value.into(),
        }
    }

    pub fn regex(label: impl Into<String>, pattern: impl Into<String>) -> Self {
        Self::Regex {
            label: label.into(),
            pattern: pattern.into(),
        }
    }

    pub fn metric_name(value: impl Into<String>) -> Self {
        Self::exact("__name__", value)
    }
}

#[derive(Clone, Debug)]
pub struct StoreConfig {
    pub partition_duration_ms: i64,
    pub out_of_order_tolerance_ms: i64,
    pub retention_ms: i64,
}

pub type Config = StoreConfig;

impl Default for StoreConfig {
    fn default() -> Self {
        Self {
            partition_duration_ms: DEFAULT_PARTITION_DURATION_MS,
            out_of_order_tolerance_ms: DEFAULT_OUT_OF_ORDER_TOLERANCE_MS,
            retention_ms: DEFAULT_RETENTION_MS,
        }
    }
}

impl StoreConfig {
    pub fn with_partition_duration_ms(mut self, value: i64) -> Self {
        self.partition_duration_ms = value;
        self
    }

    pub fn with_out_of_order_tolerance_ms(mut self, value: i64) -> Self {
        self.out_of_order_tolerance_ms = value;
        self
    }

    pub fn with_retention_ms(mut self, value: i64) -> Self {
        self.retention_ms = value;
        self
    }

    pub(crate) fn validate(&self) -> Result<(), StoreError> {
        if self.partition_duration_ms <= 0 {
            return Err(StoreError::InvalidConfig(
                "partition duration must be positive".to_owned(),
            ));
        }
        if self.out_of_order_tolerance_ms < 0 || self.retention_ms < 0 {
            return Err(StoreError::InvalidConfig(
                "tolerance and retention must not be negative".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct StoreStats {
    pub rejected_out_of_order: u64,
    pub flushed_partitions: u64,
    pub deleted_partitions: u64,
}

#[derive(Debug, Error)]
pub enum StoreError {
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("invalid series: {0}")]
    InvalidSeries(String),
    #[error("invalid store configuration: {0}")]
    InvalidConfig(String),
    #[error("sample at {timestamp} is older than the out-of-order tolerance; newest is {newest}")]
    OutOfOrder { timestamp: i64, newest: i64 },
    #[error("invalid matcher: {0}")]
    InvalidMatcher(String),
    #[error("corrupt storage: {0}")]
    CorruptStorage(String),
    #[error("sample codec error: {0}")]
    Codec(String),
    #[error("store lock was poisoned")]
    LockPoisoned,
}

pub(crate) fn validate_series(series: &Series) -> Result<(), StoreError> {
    const MAX_STRING_SIZE: usize = 16 * 1024 * 1024;
    if series.name.is_empty() {
        return Err(StoreError::InvalidSeries("metric name is empty".to_owned()));
    }
    if series.name.len() > MAX_STRING_SIZE
        || series
            .labels
            .iter()
            .any(|(name, value)| name.len() > MAX_STRING_SIZE || value.len() > MAX_STRING_SIZE)
    {
        return Err(StoreError::InvalidSeries(
            "series field is too large".to_owned(),
        ));
    }
    Ok(())
}

pub(crate) fn append_string(target: &mut String, value: &str) {
    target.push_str(&value.len().to_string());
    target.push(':');
    target.push_str(value);
}

pub(crate) fn series_id(series: &Series) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in series.canonical().as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

pub(crate) fn partition_start(timestamp: i64, duration: i64) -> i64 {
    timestamp.div_euclid(duration).saturating_mul(duration)
}

pub(crate) type LabelKey = (String, String);
pub(crate) type LabelIndex = BTreeMap<LabelKey, BTreeSet<u64>>;

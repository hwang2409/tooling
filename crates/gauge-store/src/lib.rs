//! Prometheus-lite time-series storage for gauge.

mod codec;
mod format;
mod matcher;
mod model;
mod store;
mod wal;

pub use model::{
    Config, DEFAULT_OUT_OF_ORDER_TOLERANCE_MS, DEFAULT_PARTITION_DURATION_MS, DEFAULT_RETENTION_MS,
    Matcher, Sample, Series, SeriesSamples, StoreConfig, StoreError, StoreStats,
};
pub use store::{GaugeStore, ReadHandle, SelectIter, Store, WriteHandle};

pub fn encode_samples(samples: &[Sample]) -> Result<Vec<u8>, StoreError> {
    codec::encode(samples).map_err(|error| StoreError::Codec(error.to_string()))
}

pub fn decode_samples(bytes: &[u8]) -> Result<Vec<Sample>, StoreError> {
    codec::decode(bytes).map_err(|error| StoreError::Codec(error.to_string()))
}

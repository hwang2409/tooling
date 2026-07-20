//! Scraping server for gauge.

pub mod config;
pub mod parser;
pub mod scrape;
pub mod server;

pub use config::{Config, ConfigError, StorageConfig, TargetConfig, parse_duration};
pub use parser::{MetricSample, ParseResult, parse_exposition};
pub use scrape::{JITTER_FRACTION, jittered_interval};

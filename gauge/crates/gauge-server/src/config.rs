use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::Deserialize;

#[derive(Clone, Debug, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub targets: Vec<TargetConfig>,
    #[serde(default = "default_scrape_timeout", alias = "timeout")]
    pub scrape_timeout: DurationValue,
    #[serde(default)]
    pub storage: StorageConfig,
}

#[derive(Clone, Debug, Deserialize)]
pub struct TargetConfig {
    pub name: String,
    pub url: String,
    #[serde(default = "default_interval")]
    pub interval: DurationValue,
}

#[derive(Clone, Debug, Deserialize)]
pub struct StorageConfig {
    #[serde(default = "default_data_dir")]
    pub data_dir: PathBuf,
    #[serde(default = "default_retention")]
    pub retention: DurationValue,
    #[serde(default)]
    pub partition_duration: Option<DurationValue>,
    #[serde(default)]
    pub out_of_order_tolerance: Option<DurationValue>,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            data_dir: default_data_dir(),
            retention: default_retention(),
            partition_duration: None,
            out_of_order_tolerance: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct DurationValue(pub Duration);

impl<'de> Deserialize<'de> for DurationValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Value {
            String(String),
            Seconds(u64),
        }
        match Value::deserialize(deserializer)? {
            Value::String(value) => parse_duration(&value)
                .map(DurationValue)
                .map_err(serde::de::Error::custom),
            Value::Seconds(value) => Ok(DurationValue(Duration::from_secs(value))),
        }
    }
}

impl Default for DurationValue {
    fn default() -> Self {
        default_interval()
    }
}

impl DurationValue {
    pub fn as_duration(&self) -> Duration {
        self.0
    }
}

fn default_interval() -> DurationValue {
    DurationValue(Duration::from_secs(15))
}

fn default_scrape_timeout() -> DurationValue {
    DurationValue(Duration::from_secs(5))
}

fn default_retention() -> DurationValue {
    DurationValue(Duration::from_secs(30 * 24 * 60 * 60))
}

fn default_data_dir() -> PathBuf {
    PathBuf::from("./gauge-data")
}

pub fn parse_duration(value: &str) -> Result<Duration, String> {
    let value = value.trim();
    if value.is_empty() {
        return Err("duration must not be empty".to_owned());
    }
    let split = value
        .find(|character: char| !character.is_ascii_digit() && character != '.')
        .ok_or_else(|| "duration needs a unit (ms, s, m, h, or d)".to_owned())?;
    let (number, unit) = value.split_at(split);
    let amount: f64 = number
        .parse()
        .map_err(|_| format!("invalid duration: {value}"))?;
    if !amount.is_finite() || amount < 0.0 {
        return Err(format!("invalid duration: {value}"));
    }
    let multiplier = match unit {
        "ms" => 0.001,
        "s" => 1.0,
        "m" => 60.0,
        "h" => 3600.0,
        "d" => 86400.0,
        _ => return Err(format!("unknown duration unit in {value}")),
    };
    let seconds = amount * multiplier;
    if seconds > Duration::MAX.as_secs_f64() {
        return Err(format!("duration is too large: {value}"));
    }
    Ok(Duration::from_secs_f64(seconds))
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("could not read config {path}: {source}")]
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("invalid TOML: {0}")]
    Toml(#[from] toml::de::Error),
    #[error("config validation failed: {0}")]
    Invalid(String),
}

impl Config {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|source| ConfigError::Read {
            path: path.to_owned(),
            source,
        })?;
        let mut config: Self = toml::from_str(&text)?;
        config.validate()?;
        if config.storage.data_dir.is_relative() {
            let base = path.parent().unwrap_or_else(|| Path::new("."));
            config.storage.data_dir = base.join(&config.storage.data_dir);
        }
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.targets.is_empty() {
            return Err(ConfigError::Invalid(
                "at least one target is required".to_owned(),
            ));
        }
        if self.scrape_timeout.0.is_zero() {
            return Err(ConfigError::Invalid(
                "scrape timeout must be positive".to_owned(),
            ));
        }
        let mut names = std::collections::BTreeSet::new();
        for target in &self.targets {
            if target.name.is_empty() || target.name.contains('"') {
                return Err(ConfigError::Invalid(format!(
                    "target name must be non-empty and quote-free: {}",
                    target.name
                )));
            }
            if !names.insert(&target.name) {
                return Err(ConfigError::Invalid(format!(
                    "duplicate target name: {}",
                    target.name
                )));
            }
            if !target.url.starts_with("http://") {
                return Err(ConfigError::Invalid(format!(
                    "target {} URL must use http://",
                    target.name
                )));
            }
            if target.interval.0.is_zero() {
                return Err(ConfigError::Invalid(format!(
                    "target {} interval must be positive",
                    target.name
                )));
            }
        }
        if self.storage.retention.0.is_zero() {
            return Err(ConfigError::Invalid(
                "storage retention must be positive".to_owned(),
            ));
        }
        for (name, value) in [
            (
                "partition_duration",
                self.storage.partition_duration.as_ref(),
            ),
            (
                "out_of_order_tolerance",
                self.storage.out_of_order_tolerance.as_ref(),
            ),
        ] {
            if value.is_some_and(|duration| duration.0.is_zero()) {
                return Err(ConfigError::Invalid(format!(
                    "storage {name} must be positive"
                )));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_duration_units() {
        assert_eq!(
            parse_duration("1500ms").unwrap(),
            Duration::from_millis(1500)
        );
        assert_eq!(parse_duration("2.5m").unwrap(), Duration::from_secs(150));
        assert!(parse_duration("15").is_err());
    }

    #[test]
    fn rejects_bad_config() {
        let result: Result<Config, _> = toml::from_str(
            r#"scrape_timeout = "0s"
            [[targets]]
            name = "one"
            url = "https://example.com/metrics"
            "#,
        );
        assert!(result.unwrap().validate().is_err());
    }
}

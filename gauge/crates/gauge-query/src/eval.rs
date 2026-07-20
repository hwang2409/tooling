use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use gauge_store::{GaugeStore, Matcher, Sample, SeriesSamples};
use regex::Regex;

use crate::parser::{AggregateOp, BinaryOp, Expr, MatchOp, ParseError, Selector, UnaryOp};

pub const DEFAULT_LOOKBACK_MS: i64 = 5 * 60 * 1_000;
pub const MAX_RANGE_POINTS: i64 = 11_000;

#[derive(Clone, Debug, PartialEq)]
pub struct InstantSample {
    pub metric: BTreeMap<String, String>,
    pub timestamp: i64,
    pub value: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RangeSeries {
    pub metric: BTreeMap<String, String>,
    pub values: Vec<InstantSample>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EvalError {
    Store(String),
    InvalidRange(String),
    InvalidExpression(String),
}

impl fmt::Display for EvalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Store(message) => write!(f, "storage error: {message}"),
            Self::InvalidRange(message) | Self::InvalidExpression(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for EvalError {}

#[derive(Debug)]
pub enum QueryError {
    Parse(ParseError),
    Eval(EvalError),
}

impl fmt::Display for QueryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse(error) => error.fmt(f),
            Self::Eval(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for QueryError {}

#[derive(Clone)]
pub struct QueryEngine {
    store: GaugeStore,
    lookback_ms: i64,
}

impl QueryEngine {
    pub fn new(store: GaugeStore) -> Self {
        Self {
            store,
            lookback_ms: DEFAULT_LOOKBACK_MS,
        }
    }

    pub fn with_lookback_ms(mut self, lookback_ms: i64) -> Self {
        self.lookback_ms = lookback_ms.max(0);
        self
    }

    pub fn lookback_ms(&self) -> i64 {
        self.lookback_ms
    }

    pub fn store(&self) -> &GaugeStore {
        &self.store
    }

    pub fn query(
        &self,
        expression: &str,
        timestamp: i64,
    ) -> Result<Vec<InstantSample>, QueryError> {
        let expr = crate::parser::parse(expression).map_err(QueryError::Parse)?;
        self.eval_instant(&expr, timestamp)
            .map_err(QueryError::Eval)
    }

    pub fn query_range(
        &self,
        expression: &str,
        start: i64,
        end: i64,
        step: i64,
    ) -> Result<Vec<RangeSeries>, QueryError> {
        let expr = crate::parser::parse(expression).map_err(QueryError::Parse)?;
        self.eval_range(&expr, start, end, step)
            .map_err(QueryError::Eval)
    }

    pub fn eval_instant(
        &self,
        expression: &Expr,
        timestamp: i64,
    ) -> Result<Vec<InstantSample>, EvalError> {
        let value = self.eval_expression(expression, timestamp)?;
        let mut result = match value {
            Value::Scalar(value) => vec![VectorSample {
                metric: Metric::default(),
                value,
            }],
            Value::Vector(vector) => vector,
        };
        result.sort_by(|left, right| left.metric.cmp(&right.metric));
        Ok(result
            .into_iter()
            .map(|sample| InstantSample {
                metric: sample.metric.as_labels(),
                timestamp,
                value: sample.value,
            })
            .collect())
    }

    pub fn eval_range(
        &self,
        expression: &Expr,
        start: i64,
        end: i64,
        step: i64,
    ) -> Result<Vec<RangeSeries>, EvalError> {
        if end < start {
            return Err(EvalError::InvalidRange(
                "range end must be greater than or equal to start".to_owned(),
            ));
        }
        if step <= 0 {
            return Err(EvalError::InvalidRange(
                "range step must be positive".to_owned(),
            ));
        }
        let point_count = (i128::from(end) - i128::from(start)) / i128::from(step) + 1;
        if point_count > i128::from(MAX_RANGE_POINTS) {
            return Err(EvalError::InvalidRange(format!(
                "range query exceeds maximum of {MAX_RANGE_POINTS} points"
            )));
        }

        let mut series: BTreeMap<Metric, Vec<InstantSample>> = BTreeMap::new();
        let mut timestamp = start;
        loop {
            for sample in self.eval_instant(expression, timestamp)? {
                let metric = Metric::from_labels(&sample.metric);
                series.entry(metric).or_default().push(sample);
            }
            let Some(next) = timestamp.checked_add(step) else {
                break;
            };
            if next > end {
                break;
            }
            timestamp = next;
        }

        Ok(series
            .into_iter()
            .map(|(metric, values)| RangeSeries {
                metric: metric.as_labels(),
                values,
            })
            .collect())
    }

    pub fn series(&self, selector: &Selector) -> Result<Vec<BTreeMap<String, String>>, EvalError> {
        let rows = self.select_rows(selector, i64::MIN, i64::MAX)?;
        let mut result = BTreeSet::new();
        for row in rows {
            result.insert(Metric::from_series(&row.series).as_labels());
        }
        Ok(result.into_iter().collect())
    }

    fn eval_expression(&self, expression: &Expr, timestamp: i64) -> Result<Value, EvalError> {
        match expression {
            Expr::Number(value) => Ok(Value::Scalar(*value)),
            Expr::Selector(selector) => self.eval_selector(selector, timestamp),
            Expr::Rate {
                selector,
                window_ms,
            } => self.eval_rate(selector, *window_ms, timestamp),
            Expr::Aggregate { op, expr, by } => self.eval_aggregate(*op, expr, by, timestamp),
            Expr::Binary { left, op, right } => {
                let left = self.eval_expression(left, timestamp)?;
                let right = self.eval_expression(right, timestamp)?;
                apply_binary(left, *op, right)
            }
            Expr::Unary { op, expr } => {
                let value = self.eval_expression(expr, timestamp)?;
                apply_unary(value, *op)
            }
        }
    }

    fn eval_selector(&self, selector: &Selector, timestamp: i64) -> Result<Value, EvalError> {
        let start = timestamp.saturating_sub(self.lookback_ms);
        let rows = self.select_rows(selector, start, timestamp)?;
        let vector = rows
            .into_iter()
            .filter_map(|row| {
                latest_value(&row.samples).map(|value| VectorSample {
                    metric: Metric::from_series(&row.series),
                    value,
                })
            })
            .collect();
        Ok(Value::Vector(vector))
    }

    fn eval_rate(
        &self,
        selector: &Selector,
        window_ms: i64,
        timestamp: i64,
    ) -> Result<Value, EvalError> {
        if window_ms <= 0 {
            return Err(EvalError::InvalidExpression(
                "rate window must be positive".to_owned(),
            ));
        }
        let start = timestamp.saturating_sub(window_ms);
        let rows = self.select_rows(selector, start, timestamp)?;
        let vector = rows
            .into_iter()
            .filter_map(|row| {
                rate_value(&row.samples).map(|value| VectorSample {
                    metric: Metric::from_series(&row.series),
                    value,
                })
            })
            .collect();
        Ok(Value::Vector(vector))
    }

    fn eval_aggregate(
        &self,
        op: AggregateOp,
        expression: &Expr,
        by: &[String],
        timestamp: i64,
    ) -> Result<Value, EvalError> {
        let value = self.eval_expression(expression, timestamp)?;
        let vector = match value {
            Value::Scalar(value) => vec![VectorSample {
                metric: Metric::default(),
                value,
            }],
            Value::Vector(vector) => vector,
        };
        let mut groups: BTreeMap<BTreeMap<String, String>, Vec<f64>> = BTreeMap::new();
        for sample in vector {
            let labels = by
                .iter()
                .filter_map(|label| {
                    sample
                        .metric
                        .label_value(label)
                        .map(|value| (label.clone(), value.to_owned()))
                })
                .collect();
            groups.entry(labels).or_default().push(sample.value);
        }
        let vector = groups
            .into_iter()
            .map(|(labels, values)| VectorSample {
                metric: Metric { name: None, labels },
                value: aggregate_value(op, &values),
            })
            .collect();
        Ok(Value::Vector(vector))
    }

    fn select_rows(
        &self,
        selector: &Selector,
        start: i64,
        end: i64,
    ) -> Result<Vec<SeriesSamples>, EvalError> {
        let compiled = CompiledSelector::new(selector)?;
        let store_matchers = compiled.store_matchers();
        let rows = self
            .store
            .select(&store_matchers, start, end)
            .map_err(|error| EvalError::Store(error.to_string()))?;
        Ok(rows
            .into_iter()
            .filter(|row| compiled.matches(&row.series))
            .collect())
    }
}

#[derive(Clone, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
struct Metric {
    name: Option<String>,
    labels: BTreeMap<String, String>,
}

impl Metric {
    fn from_series(series: &gauge_store::Series) -> Self {
        Self {
            name: Some(series.name.clone()),
            labels: series.labels.clone(),
        }
    }

    fn from_labels(labels: &BTreeMap<String, String>) -> Self {
        let mut labels = labels.clone();
        let name = labels.remove("__name__");
        Self { name, labels }
    }

    fn as_labels(&self) -> BTreeMap<String, String> {
        let mut labels = self.labels.clone();
        if let Some(name) = &self.name {
            labels.insert("__name__".to_owned(), name.clone());
        }
        labels
    }

    fn label_value(&self, label: &str) -> Option<&str> {
        if label == "__name__" {
            self.name.as_deref()
        } else {
            self.labels.get(label).map(String::as_str)
        }
    }
}

#[derive(Clone, Debug)]
struct VectorSample {
    metric: Metric,
    value: f64,
}

enum Value {
    Scalar(f64),
    Vector(Vec<VectorSample>),
}

struct CompiledSelector {
    selector: Selector,
    regexes: Vec<Option<Regex>>,
}

impl CompiledSelector {
    fn new(selector: &Selector) -> Result<Self, EvalError> {
        let regexes = selector
            .matchers
            .iter()
            .map(|matcher| match matcher.op {
                MatchOp::Regex | MatchOp::NotRegex => {
                    Regex::new(&format!("^(?:{})$", matcher.value))
                        .map(Some)
                        .map_err(|error| {
                            EvalError::InvalidExpression(format!(
                                "invalid regex for label '{}': {error}",
                                matcher.label
                            ))
                        })
                }
                MatchOp::Eq | MatchOp::NotEq => Ok(None),
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            selector: selector.clone(),
            regexes,
        })
    }

    fn store_matchers(&self) -> Vec<Matcher> {
        let mut matchers = vec![Matcher::metric_name(self.selector.name.clone())];
        for matcher in &self.selector.matchers {
            match matcher.op {
                MatchOp::Eq => {
                    matchers.push(Matcher::exact(matcher.label.clone(), matcher.value.clone()))
                }
                MatchOp::Regex => {
                    matchers.push(Matcher::regex(matcher.label.clone(), matcher.value.clone()))
                }
                MatchOp::NotEq | MatchOp::NotRegex => {}
            }
        }
        matchers
    }

    fn matches(&self, series: &gauge_store::Series) -> bool {
        self.selector
            .matchers
            .iter()
            .enumerate()
            .all(|(index, matcher)| {
                let actual = if matcher.label == "__name__" {
                    Some(series.name.as_str())
                } else {
                    series.labels.get(&matcher.label).map(String::as_str)
                };
                match (matcher.op, actual) {
                    (MatchOp::Eq, Some(actual)) => actual == matcher.value,
                    (MatchOp::NotEq, Some(actual)) => actual != matcher.value,
                    (MatchOp::Regex, Some(actual)) => self.regexes[index]
                        .as_ref()
                        .is_some_and(|regex| regex.is_match(actual)),
                    (MatchOp::NotRegex, Some(actual)) => self.regexes[index]
                        .as_ref()
                        .is_some_and(|regex| !regex.is_match(actual)),
                    (_, None) => false,
                }
            })
    }
}

fn latest_value(samples: &[Sample]) -> Option<f64> {
    samples
        .last()
        .filter(|sample| !sample.value.is_nan())
        .map(|sample| sample.value)
}

fn rate_value(samples: &[Sample]) -> Option<f64> {
    let start = samples
        .iter()
        .rposition(|sample| sample.value.is_nan())
        .map_or(0, |index| index.saturating_add(1));
    let samples = &samples[start..];
    if samples.len() < 2 {
        return None;
    }
    let first = samples.first()?;
    let last = samples.last()?;
    let elapsed_ms = last.timestamp.checked_sub(first.timestamp)?;
    if elapsed_ms <= 0 {
        return None;
    }
    let mut increase = 0.0;
    let mut previous = first.value;
    let mut previous_max = first.value;
    for sample in &samples[1..] {
        if sample.value < previous {
            increase += sample.value - previous + previous_max;
            previous_max = sample.value;
        } else {
            increase += sample.value - previous;
            previous_max = previous_max.max(sample.value);
        }
        previous = sample.value;
    }
    // Deliberately use the first-to-last observed span. We do not extrapolate
    // sparse samples to the requested rate-window boundaries.
    Some(increase / (elapsed_ms as f64 / 1_000.0))
}

fn aggregate_value(op: AggregateOp, values: &[f64]) -> f64 {
    match op {
        AggregateOp::Sum => values.iter().sum(),
        AggregateOp::Avg => values.iter().sum::<f64>() / values.len() as f64,
        AggregateOp::Min => values.iter().copied().fold(f64::INFINITY, f64::min),
        AggregateOp::Max => values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        AggregateOp::Count => values.len() as f64,
    }
}

fn apply_unary(value: Value, op: UnaryOp) -> Result<Value, EvalError> {
    let sign = match op {
        UnaryOp::Plus => 1.0,
        UnaryOp::Minus => -1.0,
    };
    match value {
        Value::Scalar(value) => Ok(Value::Scalar(sign * value)),
        Value::Vector(vector) => Ok(Value::Vector(
            vector
                .into_iter()
                .map(|sample| VectorSample {
                    value: sign * sample.value,
                    ..sample
                })
                .collect(),
        )),
    }
}

fn apply_binary(left: Value, op: BinaryOp, right: Value) -> Result<Value, EvalError> {
    match (left, right) {
        (Value::Scalar(left), Value::Scalar(right)) => {
            Ok(Value::Scalar(apply_scalar_binary(left, op, right)))
        }
        (Value::Vector(vector), Value::Scalar(scalar)) => Ok(Value::Vector(
            vector
                .into_iter()
                .map(|sample| VectorSample {
                    value: apply_scalar_binary(sample.value, op, scalar),
                    ..sample
                })
                .collect(),
        )),
        (Value::Scalar(scalar), Value::Vector(vector)) => Ok(Value::Vector(
            vector
                .into_iter()
                .map(|sample| VectorSample {
                    value: apply_scalar_binary(scalar, op, sample.value),
                    ..sample
                })
                .collect(),
        )),
        (Value::Vector(_), Value::Vector(_)) => Err(EvalError::InvalidExpression(
            "arithmetic between two vector expressions is not supported".to_owned(),
        )),
    }
}

fn apply_scalar_binary(left: f64, op: BinaryOp, right: f64) -> f64 {
    match op {
        BinaryOp::Add => left + right,
        BinaryOp::Sub => left - right,
        BinaryOp::Mul => left * right,
        BinaryOp::Div => left / right,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    use gauge_store::{GaugeStore, Sample, Series};

    use super::*;
    use crate::parser::{MAX_PARSE_DEPTH, parse};

    static NEXT_PATH: AtomicU64 = AtomicU64::new(0);

    fn fixture_store(samples: &[(Series, i64, f64)]) -> (GaugeStore, std::path::PathBuf) {
        let path = std::env::temp_dir().join(format!(
            "gauge-query-test-{}-{}-{}",
            std::process::id(),
            samples.len(),
            NEXT_PATH.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&path);
        let store = GaugeStore::open(&path).unwrap();
        for (series, timestamp, value) in samples {
            store
                .append(series.clone(), Sample::new(*timestamp, *value))
                .unwrap();
        }
        (store, path)
    }

    fn series(name: &str, labels: &[(&str, &str)]) -> Series {
        Series::from_labels(
            name,
            labels
                .iter()
                .map(|(key, value)| ((*key).to_owned(), (*value).to_owned())),
        )
    }

    fn values(result: &[InstantSample]) -> Vec<(BTreeMap<String, String>, f64)> {
        result
            .iter()
            .map(|sample| (sample.metric.clone(), sample.value))
            .collect()
    }

    #[test]
    fn counter_rate_adds_previous_max_on_reset() {
        let (store, path) = fixture_store(&[
            (series("requests", &[("job", "api")]), 0, 10.0),
            (series("requests", &[("job", "api")]), 1_000, 15.0),
            (series("requests", &[("job", "api")]), 2_000, 3.0),
            (series("requests", &[("job", "api")]), 3_000, 8.0),
        ]);
        let engine = QueryEngine::new(store);
        let expr = parse("rate(requests[3s])").unwrap();
        let result = engine.eval_instant(&expr, 3_000).unwrap();
        assert_eq!(result.len(), 1);
        assert!(
            (result[0].value - 13.0 / 3.0).abs() < 1e-9,
            "got {}",
            result[0].value
        );
        fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn aggregation_and_scalar_arithmetic_are_deterministic() {
        let (store, path) = fixture_store(&[
            (
                series("cpu", &[("job", "api"), ("instance", "a")]),
                1_000,
                2.0,
            ),
            (
                series("cpu", &[("job", "api"), ("instance", "b")]),
                1_000,
                3.0,
            ),
            (
                series("cpu", &[("job", "worker"), ("instance", "a")]),
                1_000,
                7.0,
            ),
        ]);
        let engine = QueryEngine::new(store);
        let expr = parse("sum(cpu * 100) by (job)").unwrap();
        let result = engine.eval_instant(&expr, 1_000).unwrap();
        assert_eq!(
            values(&result),
            vec![
                (
                    BTreeMap::from([("job".to_owned(), "api".to_owned())]),
                    500.0,
                ),
                (
                    BTreeMap::from([("job".to_owned(), "worker".to_owned())]),
                    700.0,
                )
            ]
        );
        fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn all_aggregation_operators_match_hand_computed_values() {
        let (store, path) = fixture_store(&[
            (
                series("gauge", &[("job", "api"), ("instance", "a")]),
                1_000,
                2.0,
            ),
            (
                series("gauge", &[("job", "api"), ("instance", "b")]),
                2_000,
                6.0,
            ),
            (series("gauge", &[("job", "worker")]), 1_000, 9.0),
        ]);
        let engine = QueryEngine::new(store);
        for (expression, expected) in [
            ("sum(gauge) by (job)", [("api", 8.0), ("worker", 9.0)]),
            ("avg(gauge) by (job)", [("api", 4.0), ("worker", 9.0)]),
            ("min(gauge) by (job)", [("api", 2.0), ("worker", 9.0)]),
            ("max(gauge) by (job)", [("api", 6.0), ("worker", 9.0)]),
            ("count(gauge) by (job)", [("api", 2.0), ("worker", 1.0)]),
        ] {
            let result = engine
                .eval_instant(&parse(expression).unwrap(), 2_000)
                .unwrap();
            for (job, value) in expected {
                let sample = result
                    .iter()
                    .find(|sample| sample.metric.get("job").is_some_and(|actual| actual == job))
                    .unwrap();
                assert_eq!(sample.value, value, "{expression}");
            }
        }
        fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn range_uses_start_then_step_and_lookback() {
        let (store, path) = fixture_store(&[(series("gauge", &[]), 1_000, 4.0)]);
        let engine = QueryEngine::new(store).with_lookback_ms(500);
        let expr = parse("gauge").unwrap();
        let result = engine.eval_range(&expr, 1_500, 3_100, 1_000).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(
            result[0]
                .values
                .iter()
                .map(|sample| sample.timestamp)
                .collect::<Vec<_>>(),
            vec![1_500]
        );
        fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn range_with_step_larger_than_window_and_empty_window_is_deterministic() {
        let (store, path) = fixture_store(&[(series("gauge", &[]), 1_000, 4.0)]);
        let engine = QueryEngine::new(store).with_lookback_ms(100);
        let expr = parse("gauge").unwrap();
        let result = engine.eval_range(&expr, 2_000, 2_050, 1_000).unwrap();
        assert!(result.is_empty());
        fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn range_point_budget_has_a_clear_boundary() {
        let (store, path) = fixture_store(&[]);
        let engine = QueryEngine::new(store);
        let expr = parse("1").unwrap();
        let result = engine
            .eval_range(&expr, 0, MAX_RANGE_POINTS - 1, 1)
            .unwrap();
        assert_eq!(result[0].values.len(), MAX_RANGE_POINTS as usize);
        let error = engine
            .eval_range(&expr, 0, MAX_RANGE_POINTS, 1)
            .unwrap_err();
        assert!(error.to_string().contains("11000"));
        fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn sparse_rate_uses_observed_span_without_boundary_extrapolation() {
        let (store, path) = fixture_store(&[
            (series("requests", &[]), 1_000, 10.0),
            (series("requests", &[]), 3_000, 16.0),
        ]);
        let engine = QueryEngine::new(store);
        let expr = parse("rate(requests[10s])").unwrap();
        let result = engine.eval_instant(&expr, 5_000).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].value, 3.0);
        fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn largest_admitted_arithmetic_chain_evaluates_and_drops_safely() {
        let expression = vec!["1"; MAX_PARSE_DEPTH].join("+");
        let parsed = parse(&expression).unwrap();
        let (store, path) = fixture_store(&[]);
        let result = QueryEngine::new(store).eval_instant(&parsed, 0).unwrap();
        assert_eq!(result[0].value, MAX_PARSE_DEPTH as f64);
        drop(parsed);
        let too_deep = vec!["1"; MAX_PARSE_DEPTH + 1].join("+");
        let error = parse(&too_deep).unwrap_err();
        assert!(error.message.contains("maximum expression nesting depth"));
        assert!(error.position > 0);
        fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn staleness_marker_hides_old_value_until_new_sample() {
        let (store, path) = fixture_store(&[
            (series("gauge", &[]), 1_000, 4.0),
            (series("gauge", &[]), 2_000, f64::NAN),
            (series("gauge", &[]), 4_000, 9.0),
        ]);
        let engine = QueryEngine::new(store).with_lookback_ms(10_000);
        let expr = parse("gauge").unwrap();
        assert!(engine.eval_instant(&expr, 3_000).unwrap().is_empty());
        assert_eq!(engine.eval_instant(&expr, 4_000).unwrap()[0].value, 9.0);
        fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn negative_and_regex_matchers_are_anchored() {
        let (store, path) = fixture_store(&[
            (series("cpu", &[("job", "api-a")]), 1, 1.0),
            (series("cpu", &[("job", "api-b")]), 1, 2.0),
            (series("cpu", &[("job", "worker")]), 1, 3.0),
        ]);
        let engine = QueryEngine::new(store);
        let expr = parse(r#"cpu{job=~"api-." ,job!="api-b"}"#).unwrap();
        let result = engine.eval_instant(&expr, 1).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].value, 1.0);
        fs::remove_dir_all(path).unwrap();
    }
}

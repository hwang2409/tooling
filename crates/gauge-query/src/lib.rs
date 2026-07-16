//! Query engine and HTTP API for gauge's Prometheus-lite data model.
//!
//! The grammar is intentionally small and stable. It supports selectors such
//! as `http_requests_total{job="api",instance=~"web-.*"}`, counter-aware
//! `rate(selector[5m])`, `sum|avg|min|max|count (expr) by (label, ...)`, and
//! arithmetic between a query expression and a scalar.
//!
//! The HTTP response shapes are deliberately typed rather than copied from
//! Prometheus' tuple-shaped JSON:
//!
//! ```json
//! {
//!   "status": "success",
//!   "data": {
//!     "resultType": "vector",
//!     "result": [{
//!       "metric": {"__name__": "cpu", "job": "api"},
//!       "value": {"timestamp": 1000, "value": 0.5}
//!     }]
//!   }
//! }
//! ```
//!
//! Range responses use `resultType: "matrix"` and replace `value` with
//! `values: [{"timestamp": ..., "value": ...}]`. Series responses have
//! `data` as an array of metric-label maps. Errors always have this shape:
//! `{"status":"error","errorType":"parse_error|execution_error|bad_data",`
//! `"error":"...","position":null|<byte offset>}`.
//!
//! `QueryPoint.value` is an `f64` in Rust. Finite values are JSON numbers;
//! non-finite values use the symmetric string spellings `"NaN"`, `"+Inf"`,
//! and `"-Inf"` so typed clients never receive lossy JSON `null` values.
//! Range queries are limited to 11,000 evaluation points. `rate()` uses the
//! first-to-last observed sample span and deliberately does not extrapolate
//! sparse samples to the rate-window boundaries.

pub mod api;
pub mod eval;
pub mod parser;

pub use api::{
    ApiErrorResponse, InstantQueryData, InstantQueryResponse, InstantResult, QueryPoint,
    RangeQueryData, RangeQueryResponse, RangeResult, SeriesQueryResponse, router,
};
pub use eval::{
    DEFAULT_LOOKBACK_MS, EvalError, InstantSample, MAX_RANGE_POINTS, QueryEngine, QueryError,
    RangeSeries,
};
pub use parser::{
    AggregateOp, BinaryOp, Expr, LabelMatcher, MAX_PARSE_DEPTH, MAX_PARSE_NODES, MatchOp,
    ParseError, Selector, UnaryOp, parse, parse_selector,
};

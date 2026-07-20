use std::collections::BTreeMap;

use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use gauge_store::GaugeStore;
use serde::de::{self, Visitor};
use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;

use crate::eval::{EvalError, QueryEngine};
use crate::parser::{ParseError, parse, parse_selector};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct InstantQueryResponse {
    pub status: String,
    pub data: InstantQueryData,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct InstantQueryData {
    #[serde(rename = "resultType")]
    pub result_type: String,
    pub result: Vec<InstantResult>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct InstantResult {
    pub metric: BTreeMap<String, String>,
    pub value: QueryPoint,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct RangeQueryResponse {
    pub status: String,
    pub data: RangeQueryData,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct RangeQueryData {
    #[serde(rename = "resultType")]
    pub result_type: String,
    pub result: Vec<RangeResult>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct RangeResult {
    pub metric: BTreeMap<String, String>,
    pub values: Vec<QueryPoint>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct QueryPoint {
    pub timestamp: i64,
    pub value: f64,
}

impl Serialize for QueryPoint {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("QueryPoint", 2)?;
        state.serialize_field("timestamp", &self.timestamp)?;
        state.serialize_field("value", &WireValue(self.value))?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for QueryPoint {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct QueryPointWire {
            timestamp: i64,
            value: WireValue,
        }

        let wire = QueryPointWire::deserialize(deserializer)?;
        Ok(Self {
            timestamp: wire.timestamp,
            value: wire.value.0,
        })
    }
}

struct WireValue(f64);

impl Serialize for WireValue {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        if self.0.is_finite() {
            serializer.serialize_f64(self.0)
        } else if self.0.is_nan() {
            serializer.serialize_str("NaN")
        } else if self.0.is_sign_positive() {
            serializer.serialize_str("+Inf")
        } else {
            serializer.serialize_str("-Inf")
        }
    }
}

impl<'de> Deserialize<'de> for WireValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct WireValueVisitor;

        impl<'de> Visitor<'de> for WireValueVisitor {
            type Value = WireValue;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a finite JSON number or NaN/+Inf/-Inf string")
            }

            fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                if value.is_finite() {
                    Ok(WireValue(value))
                } else {
                    Err(E::custom(
                        "non-finite JSON numbers must use a special string",
                    ))
                }
            }

            fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(WireValue(value as f64))
            }

            fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                let value = value as f64;
                if value.is_finite() {
                    Ok(WireValue(value))
                } else {
                    Err(E::custom("JSON number is too large"))
                }
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                match value {
                    "NaN" => Ok(WireValue(f64::NAN)),
                    "+Inf" => Ok(WireValue(f64::INFINITY)),
                    "-Inf" => Ok(WireValue(f64::NEG_INFINITY)),
                    _ => Err(E::custom(
                        "special value must be exactly NaN, +Inf, or -Inf",
                    )),
                }
            }
        }

        deserializer.deserialize_any(WireValueVisitor)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct SeriesQueryResponse {
    pub status: String,
    pub data: Vec<BTreeMap<String, String>>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ApiErrorResponse {
    pub status: String,
    #[serde(rename = "errorType")]
    pub error_type: String,
    pub error: String,
    pub position: Option<usize>,
}

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    error_type: &'static str,
    message: String,
    position: Option<usize>,
}

impl ApiError {
    fn bad_data(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            error_type: "bad_data",
            message: message.into(),
            position: None,
        }
    }

    fn parse(error: ParseError) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            error_type: "parse_error",
            message: error.message,
            position: Some(error.position),
        }
    }

    fn eval(error: EvalError) -> Self {
        let (status, error_type) = match &error {
            EvalError::InvalidRange(_) => (StatusCode::BAD_REQUEST, "bad_data"),
            EvalError::Store(_) | EvalError::InvalidExpression(_) => {
                (StatusCode::UNPROCESSABLE_ENTITY, "execution_error")
            }
        };
        Self {
            status,
            error_type,
            message: error.to_string(),
            position: None,
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = ApiErrorResponse {
            status: "error".to_owned(),
            error_type: self.error_type.to_owned(),
            error: self.message,
            position: self.position,
        };
        (self.status, Json(body)).into_response()
    }
}

#[derive(Debug, Deserialize)]
struct QueryParams {
    expr: Option<String>,
    time: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RangeParams {
    expr: Option<String>,
    start: Option<String>,
    end: Option<String>,
    step: Option<String>,
}

#[derive(Debug, Deserialize)]
struct SeriesParams {
    #[serde(rename = "match")]
    matcher: Option<String>,
    #[serde(rename = "match[]")]
    matcher_array: Option<String>,
}

/// Build the mountable query API router.
pub fn router(store: GaugeStore) -> Router {
    Router::new()
        .route("/api/query", get(query))
        .route("/api/query_range", get(query_range))
        .route("/api/series", get(series))
        .with_state(store)
}

async fn query(
    State(store): State<GaugeStore>,
    Query(params): Query<QueryParams>,
) -> Result<Json<InstantQueryResponse>, ApiError> {
    let expression = required(params.expr, "expr")?;
    let timestamp = parse_i64(params.time, "time")?;
    let expression = parse(&expression).map_err(ApiError::parse)?;
    let result = QueryEngine::new(store)
        .eval_instant(&expression, timestamp)
        .map_err(ApiError::eval)?
        .into_iter()
        .map(|sample| InstantResult {
            metric: sample.metric,
            value: QueryPoint {
                timestamp: sample.timestamp,
                value: sample.value,
            },
        })
        .collect();
    Ok(Json(InstantQueryResponse {
        status: "success".to_owned(),
        data: InstantQueryData {
            result_type: "vector".to_owned(),
            result,
        },
    }))
}

async fn query_range(
    State(store): State<GaugeStore>,
    Query(params): Query<RangeParams>,
) -> Result<Json<RangeQueryResponse>, ApiError> {
    let expression = required(params.expr, "expr")?;
    let start = parse_i64(params.start, "start")?;
    let end = parse_i64(params.end, "end")?;
    let step = parse_i64(params.step, "step")?;
    let expression = parse(&expression).map_err(ApiError::parse)?;
    let result = QueryEngine::new(store)
        .eval_range(&expression, start, end, step)
        .map_err(ApiError::eval)?
        .into_iter()
        .map(|series| RangeResult {
            metric: series.metric,
            values: series
                .values
                .into_iter()
                .map(|sample| QueryPoint {
                    timestamp: sample.timestamp,
                    value: sample.value,
                })
                .collect(),
        })
        .collect();
    Ok(Json(RangeQueryResponse {
        status: "success".to_owned(),
        data: RangeQueryData {
            result_type: "matrix".to_owned(),
            result,
        },
    }))
}

async fn series(
    State(store): State<GaugeStore>,
    Query(params): Query<SeriesParams>,
) -> Result<Json<SeriesQueryResponse>, ApiError> {
    let expression = params
        .matcher
        .or(params.matcher_array)
        .ok_or_else(|| ApiError::bad_data("missing required query parameter 'match'"))?;
    let selector = parse_selector(&expression).map_err(ApiError::parse)?;
    let result = QueryEngine::new(store)
        .series(&selector)
        .map_err(ApiError::eval)?;
    Ok(Json(SeriesQueryResponse {
        status: "success".to_owned(),
        data: result,
    }))
}

fn required(value: Option<String>, name: &str) -> Result<String, ApiError> {
    value.ok_or_else(|| ApiError::bad_data(format!("missing required query parameter '{name}'")))
}

fn parse_i64(value: Option<String>, name: &str) -> Result<i64, ApiError> {
    let value = required(value, name)?;
    value
        .parse::<i64>()
        .map_err(|_| ApiError::bad_data(format!("query parameter '{name}' must be an integer")))
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    use axum::body::to_bytes;
    use axum::http::{Request, StatusCode};
    use gauge_store::{GaugeStore, Sample, Series};
    use tower::ServiceExt;

    use crate::parser::MAX_PARSE_DEPTH;

    use super::*;

    static NEXT_PATH: AtomicU64 = AtomicU64::new(0);

    fn fixture() -> (GaugeStore, std::path::PathBuf) {
        let path = std::env::temp_dir().join(format!(
            "gauge-api-test-{}-{}",
            std::process::id(),
            NEXT_PATH.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&path);
        let store = GaugeStore::open(&path).unwrap();
        for (job, value) in [("api", 2.0), ("worker", 3.0)] {
            store
                .append(
                    Series::from_labels("cpu", [("job", job)]),
                    Sample::new(1_000, value),
                )
                .unwrap();
        }
        (store, path)
    }

    #[tokio::test]
    async fn query_and_range_round_trip_as_typed_json() {
        let (store, path) = fixture();
        let app = router(store);
        let response = app
            .clone()
            .oneshot(
                Request::get("/api/query?expr=cpu%7Bjob%3D%22api%22%7D&time=1000")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let response: InstantQueryResponse = serde_json::from_slice(&body).unwrap();
        assert_eq!(response.data.result_type, "vector");
        assert_eq!(response.data.result[0].value.value, 2.0);

        let response = app
            .clone()
            .oneshot(
                Request::get("/api/query_range?expr=cpu&start=1000&end=2000&step=500")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let response: RangeQueryResponse = serde_json::from_slice(&body).unwrap();
        assert_eq!(response.data.result_type, "matrix");
        assert_eq!(response.data.result[0].values.len(), 3);
        fs::remove_dir_all(path).unwrap();
    }

    #[tokio::test]
    async fn parse_and_eval_errors_have_stable_json_shape_and_status() {
        let (store, path) = fixture();
        let app = router(store);
        let response = app
            .clone()
            .oneshot(
                Request::get("/api/query?expr=cpu%7Bjob%3D%22api%22&time=1000")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let error: ApiErrorResponse = serde_json::from_slice(&body).unwrap();
        assert_eq!(error.status, "error");
        assert_eq!(error.error_type, "parse_error");
        assert!(error.position.is_some());

        let response = app
            .clone()
            .oneshot(
                Request::get("/api/query_range?expr=cpu&start=1000&end=2000&step=0")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let error: ApiErrorResponse = serde_json::from_slice(&body).unwrap();
        assert_eq!(error.error_type, "bad_data");
        assert!(error.position.is_none());

        for path in [
            "/api/query?expr=cpu%7Bjob%3D~%22%5B%22%7D&time=1000",
            "/api/series?match=cpu%7Bjob%3D~%22%5B%22%7D",
        ] {
            let response = app
                .clone()
                .oneshot(Request::get(path).body(axum::body::Body::empty()).unwrap())
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
            let error: ApiErrorResponse = serde_json::from_slice(&body).unwrap();
            assert_eq!(error.error_type, "parse_error");
            assert!(error.position.is_some());
        }
        fs::remove_dir_all(path).unwrap();
    }

    #[tokio::test]
    async fn range_point_budget_is_a_bad_request_and_names_the_limit() {
        let (store, path) = fixture();
        let app = router(store);
        let response = app
            .oneshot(
                Request::get("/api/query_range?expr=1&start=0&end=11000&step=1")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let error: ApiErrorResponse = serde_json::from_slice(&body).unwrap();
        assert_eq!(error.error_type, "bad_data");
        assert!(error.error.contains("11000"));
        fs::remove_dir_all(path).unwrap();
    }

    #[tokio::test]
    async fn arithmetic_depth_limit_is_a_positioned_bad_request() {
        let (store, path) = fixture();
        let app = router(store);
        let expression = vec!["1"; MAX_PARSE_DEPTH + 1].join("%2B");
        let uri = format!("/api/query?expr={expression}&time=0");
        let response = app
            .oneshot(Request::get(uri).body(axum::body::Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let error: ApiErrorResponse = serde_json::from_slice(&body).unwrap();
        assert_eq!(error.error_type, "parse_error");
        assert!(error.position.is_some());
        fs::remove_dir_all(path).unwrap();
    }

    #[tokio::test]
    async fn series_endpoint_returns_label_maps() {
        let (store, path) = fixture();
        let app = router(store);
        let response = app
            .oneshot(
                Request::get("/api/series?match=cpu%7Bjob%3D~%22api.%2A%22%7D")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let response: SeriesQueryResponse = serde_json::from_slice(&body).unwrap();
        assert_eq!(response.data.len(), 1);
        assert_eq!(response.data[0]["__name__"], "cpu");
        fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn query_points_round_trip_finite_and_non_finite_values() {
        for value in [0.5, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let point = QueryPoint {
                timestamp: 1_000,
                value,
            };
            let json = serde_json::to_string(&point).unwrap();
            let decoded: QueryPoint = serde_json::from_str(&json).unwrap();
            if value.is_nan() {
                assert!(decoded.value.is_nan());
                assert!(json.contains("\"NaN\""));
            } else if value.is_infinite() && value.is_sign_positive() {
                assert_eq!(decoded.value, f64::INFINITY);
                assert!(json.contains("\"+Inf\""));
            } else if value.is_infinite() {
                assert_eq!(decoded.value, f64::NEG_INFINITY);
                assert!(json.contains("\"-Inf\""));
            } else {
                assert_eq!(decoded.value, value);
                assert!(json.contains("0.5"));
            }
        }
    }
}

//! JSON HTTP API for the v0 engine.

use std::collections::BTreeMap;
use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::engine::Engine;
use crate::index::filter::Filter;
use crate::namespace::{Query, QueryResult, DEFAULT_EF_SEARCH, MAX_EF_SEARCH};
use crate::{AttrValue, Doc, Error, Result};

/// Build the application router for an engine.
pub fn router(engine: Arc<Engine>) -> Router {
    Router::new()
        .route("/v1/namespaces", get(list_namespaces))
        .route(
            "/v1/namespaces/:namespace",
            post(upsert).delete(delete_namespace),
        )
        .route("/v1/namespaces/:namespace/query", post(query))
        .with_state(engine)
}

#[derive(Debug, Deserialize)]
pub struct UpsertRequest {
    #[serde(default)]
    upserts: Vec<ApiDoc>,
    #[serde(default)]
    deletes: Vec<String>,
    #[serde(default)]
    schema: BTreeMap<String, SchemaHint>,
}

#[derive(Debug, Deserialize)]
struct ApiDoc {
    id: String,
    vector: Option<Vec<f32>>,
    #[serde(default)]
    attributes: BTreeMap<String, Value>,
}

#[derive(Debug, Deserialize)]
struct SchemaHint {
    #[serde(default)]
    full_text_search: bool,
}

#[derive(Debug, Deserialize)]
pub struct QueryRequest {
    #[serde(default)]
    vector: Option<Vec<f32>>,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    filters: Option<Value>,
    top_k: Option<usize>,
    #[serde(default)]
    include_attributes: bool,
    #[serde(default)]
    ef_search: Option<usize>,
}

#[derive(Debug, Serialize)]
struct UpsertResponse {
    upserted: usize,
    deleted: usize,
}

#[derive(Debug, Serialize)]
struct NamespaceListResponse {
    namespaces: Vec<String>,
}

#[derive(Debug, Serialize)]
struct QueryResponse {
    results: Vec<ApiResult>,
}

#[derive(Debug, Serialize)]
struct ApiResult {
    id: String,
    score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    attributes: Option<BTreeMap<String, Value>>,
}

#[derive(Debug)]
struct ApiError(Error);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = match self.0 {
            Error::InvalidKey(_) | Error::Validation(_) => StatusCode::BAD_REQUEST,
            Error::NotFound(_) => StatusCode::NOT_FOUND,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };
        (status, Json(json!({ "error": self.0.to_string() }))).into_response()
    }
}

impl From<Error> for ApiError {
    fn from(error: Error) -> Self {
        Self(error)
    }
}

async fn upsert(
    State(engine): State<Arc<Engine>>,
    Path(namespace): Path<String>,
    Json(request): Json<UpsertRequest>,
) -> std::result::Result<Json<UpsertResponse>, ApiError> {
    let upserts = request
        .upserts
        .into_iter()
        .map(api_doc)
        .collect::<Result<Vec<_>>>()?;
    let schema = request
        .schema
        .into_iter()
        .map(|(field, hint)| (field, hint.full_text_search))
        .collect();
    let summary = engine
        .upsert(&namespace, upserts, request.deletes, schema)
        .await?;
    Ok(Json(UpsertResponse {
        upserted: summary.upserted,
        deleted: summary.deleted,
    }))
}

async fn query(
    State(engine): State<Arc<Engine>>,
    Path(namespace): Path<String>,
    Json(request): Json<QueryRequest>,
) -> std::result::Result<Json<QueryResponse>, ApiError> {
    let filter = request.filters.map(parse_filter).transpose()?;
    let top_k = request
        .top_k
        .ok_or_else(|| Error::Validation("top_k is required".to_owned()))?;
    if top_k > MAX_EF_SEARCH {
        return Err(Error::Validation(format!("top_k must not exceed {MAX_EF_SEARCH}")).into());
    }
    let ef_search = request.ef_search.unwrap_or(DEFAULT_EF_SEARCH);
    let query = Query {
        vector: request.vector,
        text: request.text,
        filter,
        top_k,
        include_attributes: request.include_attributes,
    };
    let results = engine
        .query_with_ef_search(&namespace, &query, ef_search)
        .await?;
    Ok(Json(QueryResponse {
        results: results.into_iter().map(api_result).collect(),
    }))
}

async fn list_namespaces(
    State(engine): State<Arc<Engine>>,
) -> std::result::Result<Json<NamespaceListResponse>, ApiError> {
    Ok(Json(NamespaceListResponse {
        namespaces: engine.list_namespaces()?,
    }))
}

async fn delete_namespace(
    State(engine): State<Arc<Engine>>,
    Path(namespace): Path<String>,
) -> std::result::Result<StatusCode, ApiError> {
    engine.delete_namespace(&namespace).await?;
    Ok(StatusCode::NO_CONTENT)
}

fn api_doc(doc: ApiDoc) -> Result<Doc> {
    Ok(Doc {
        id: doc.id,
        vector: doc.vector,
        attributes: doc
            .attributes
            .into_iter()
            .map(|(field, value)| Ok((field, parse_attr_value(value)?)))
            .collect::<Result<BTreeMap<_, _>>>()?,
    })
}

fn parse_attr_value(value: Value) -> Result<AttrValue> {
    if value.is_object() {
        if let Ok(value) = serde_json::from_value::<AttrValue>(value.clone()) {
            return Ok(value);
        }
        return Err(Error::Validation(
            "attributes must be strings, numbers, booleans, or string arrays".to_owned(),
        ));
    }
    match value {
        Value::String(value) => Ok(AttrValue::String(value)),
        Value::Bool(value) => Ok(AttrValue::Bool(value)),
        Value::Number(value) => value
            .as_i64()
            .map(AttrValue::Int)
            .or_else(|| value.as_f64().map(AttrValue::Float))
            .ok_or_else(|| Error::Validation("invalid numeric attribute".to_owned())),
        Value::Array(values) => values
            .into_iter()
            .map(|value| match value {
                Value::String(value) => Ok(value),
                _ => Err(Error::Validation(
                    "string lists may only contain strings".to_owned(),
                )),
            })
            .collect::<Result<Vec<_>>>()
            .map(AttrValue::StringList),
        Value::Null => Err(Error::Validation(
            "null is not a supported attribute value".to_owned(),
        )),
        Value::Object(_) => Err(Error::Validation(
            "object is not a supported attribute value".to_owned(),
        )),
    }
}

fn parse_filter(value: Value) -> Result<Filter> {
    if let Value::Array(filters) = value {
        return Ok(Filter::And {
            filters: filters
                .into_iter()
                .map(parse_filter)
                .collect::<Result<Vec<_>>>()?,
        });
    }
    let object = value
        .as_object()
        .ok_or_else(|| Error::Validation("filters must be an object".to_owned()))?;
    let op = object
        .get("op")
        .and_then(Value::as_str)
        .ok_or_else(|| Error::Validation("filter op is required".to_owned()))?;
    let field = || {
        object
            .get("field")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .ok_or_else(|| Error::Validation("filter field is required".to_owned()))
    };
    let attr = || {
        object
            .get("value")
            .cloned()
            .ok_or_else(|| Error::Validation("filter value is required".to_owned()))
            .and_then(parse_attr_value)
    };
    match op {
        "eq" => Ok(Filter::Eq {
            field: field()?,
            value: attr()?,
        }),
        "ne" => Ok(Filter::Ne {
            field: field()?,
            value: attr()?,
        }),
        "in" => Ok(Filter::In {
            field: field()?,
            values: object
                .get("values")
                .and_then(Value::as_array)
                .ok_or_else(|| Error::Validation("filter values are required".to_owned()))?
                .iter()
                .cloned()
                .map(parse_attr_value)
                .collect::<Result<Vec<_>>>()?,
        }),
        "lt" => Ok(Filter::Lt {
            field: field()?,
            value: attr()?,
        }),
        "lte" => Ok(Filter::Lte {
            field: field()?,
            value: attr()?,
        }),
        "gt" => Ok(Filter::Gt {
            field: field()?,
            value: attr()?,
        }),
        "gte" => Ok(Filter::Gte {
            field: field()?,
            value: attr()?,
        }),
        "and" | "or" => {
            let filters = object
                .get("filters")
                .and_then(Value::as_array)
                .ok_or_else(|| Error::Validation("nested filters are required".to_owned()))?
                .iter()
                .cloned()
                .map(parse_filter)
                .collect::<Result<Vec<_>>>()?;
            if op == "and" {
                Ok(Filter::And { filters })
            } else {
                Ok(Filter::Or { filters })
            }
        }
        _ => Err(Error::Validation(format!("unknown filter op: {op}"))),
    }
}

fn api_result(result: QueryResult) -> ApiResult {
    ApiResult {
        id: result.id,
        score: result.score,
        attributes: result.attributes.map(|attributes| {
            attributes
                .into_iter()
                .map(|(field, value)| (field, attr_to_json(value)))
                .collect()
        }),
    }
}

fn attr_to_json(value: AttrValue) -> Value {
    match value {
        AttrValue::String(value) => Value::String(value),
        AttrValue::Int(value) => json!(value),
        AttrValue::Float(value) => json!(value),
        AttrValue::Bool(value) => json!(value),
        AttrValue::StringList(value) => json!(value),
    }
}

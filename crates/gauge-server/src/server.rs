use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use axum::extract::State;
use axum::http::header;
use axum::response::IntoResponse;
use axum::routing::get;
use axum::{Json, Router};
use gauge_store::GaugeStore;
use serde::Serialize;

#[derive(Clone)]
pub struct AppState {
    pub store: GaugeStore,
    pub targets: Arc<std::sync::RwLock<BTreeMap<String, TargetStatus>>>,
    pub metrics: Arc<ServerMetrics>,
}

#[derive(Clone, Debug, Serialize)]
pub struct TargetStatus {
    pub target: String,
    pub url: String,
    pub up: bool,
    pub last_scrape_time: Option<i64>,
    pub last_error: Option<String>,
}

#[derive(Default)]
pub struct ServerMetrics {
    pub scrapes_total: AtomicU64,
    pub scrape_failures: AtomicU64,
    pub malformed_lines: AtomicU64,
    pub samples_written: AtomicU64,
    pub write_errors: AtomicU64,
}

impl AppState {
    pub fn new(
        store: GaugeStore,
        target_names: impl IntoIterator<Item = (String, String)>,
    ) -> Self {
        let targets = target_names
            .into_iter()
            .map(|(target, url)| {
                let status = TargetStatus {
                    target: target.clone(),
                    url,
                    up: false,
                    last_scrape_time: None,
                    last_error: None,
                };
                (target, status)
            })
            .collect();
        Self {
            store,
            targets: Arc::new(std::sync::RwLock::new(targets)),
            metrics: Arc::new(ServerMetrics::default()),
        }
    }

    pub fn update_target(
        &self,
        target: &crate::config::TargetConfig,
        up: bool,
        timestamp: i64,
        error: Option<String>,
    ) {
        let targets = Arc::clone(&self.targets);
        let name = target.name.clone();
        let url = target.url.clone();
        let mut targets = targets
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let status = targets.entry(name.clone()).or_insert_with(|| TargetStatus {
            target: name,
            url,
            up: false,
            last_scrape_time: None,
            last_error: None,
        });
        status.up = up;
        status.last_scrape_time = Some(timestamp);
        status.last_error = error;
    }
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/metrics", get(metrics))
        .route("/api/targets", get(targets))
        .with_state(state)
}

async fn targets(State(state): State<AppState>) -> Json<Vec<TargetStatus>> {
    let targets = state
        .targets
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    Json(targets.values().cloned().collect())
}

async fn metrics(State(state): State<AppState>) -> impl IntoResponse {
    let malformed = state.metrics.malformed_lines.load(Ordering::Relaxed);
    let scrapes = state.metrics.scrapes_total.load(Ordering::Relaxed);
    let failures = state.metrics.scrape_failures.load(Ordering::Relaxed);
    let written = state.metrics.samples_written.load(Ordering::Relaxed);
    let errors = state.metrics.write_errors.load(Ordering::Relaxed);
    let store_stats = state.store.stats().unwrap_or_default();
    let body = format!(
        "# HELP gauge_scrapes_total Completed target scrapes.\n# TYPE gauge_scrapes_total counter\ngauge_scrapes_total {scrapes}\n# HELP gauge_scrape_failures_total Target scrapes that did not return HTTP 200.\n# TYPE gauge_scrape_failures_total counter\ngauge_scrape_failures_total {failures}\n# HELP gauge_scrape_malformed_lines Malformed exposition lines skipped.\n# TYPE gauge_scrape_malformed_lines counter\ngauge_scrape_malformed_lines {malformed}\n# HELP gauge_store_samples_written_total Samples handed to gauge-store.\n# TYPE gauge_store_samples_written_total counter\ngauge_store_samples_written_total {written}\n# HELP gauge_store_write_errors_total Store write failures.\n# TYPE gauge_store_write_errors_total counter\ngauge_store_write_errors_total {errors}\n# HELP gauge_store_flushed_partitions_total Partitions flushed by gauge-store.\n# TYPE gauge_store_flushed_partitions_total counter\ngauge_store_flushed_partitions_total {}\n# HELP gauge_store_deleted_partitions_total Partitions deleted by gauge-store retention.\n# TYPE gauge_store_deleted_partitions_total counter\ngauge_store_deleted_partitions_total {}\n",
        store_stats.flushed_partitions, store_stats.deleted_partitions
    );
    (
        [(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        body,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    #[tokio::test]
    async fn targets_endpoint_is_json() {
        let path = std::env::temp_dir().join(format!("gauge-server-api-{}", std::process::id()));
        let store = GaugeStore::open(&path).unwrap();
        let state = AppState::new(store, [("one".to_owned(), "http://127.0.0.1:1".to_owned())]);
        let response = router(state)
            .oneshot(
                Request::builder()
                    .uri("/api/targets")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
    }
}

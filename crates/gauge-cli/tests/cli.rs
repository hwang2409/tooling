use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use assert_cmd::prelude::*;
use gauge_server::server::{AppState, router};
use gauge_store::{GaugeStore, Sample, Series};

static NEXT_PATH: AtomicU64 = AtomicU64::new(0);

async fn start_server() -> (String, tokio::task::JoinHandle<()>, std::path::PathBuf) {
    let path = std::env::temp_dir().join(format!(
        "gauge-cli-fixture-{}-{}",
        std::process::id(),
        NEXT_PATH.fetch_add(1, Ordering::Relaxed)
    ));
    let store = GaugeStore::open(&path).unwrap();
    store
        .append(
            Series::from_labels("cpu", [("job", "fixture")]),
            Sample::new(1_000, 2.5),
        )
        .unwrap();
    store
        .append(
            Series::from_labels("cpu", [("job", "fixture")]),
            Sample::new(2_000, f64::NAN),
        )
        .unwrap();
    store
        .append(
            Series::from_labels("cpu", [("job", "fixture")]),
            Sample::new(3_000, 4.5),
        )
        .unwrap();
    let state = AppState::new(store, std::iter::empty());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let task = tokio::spawn(async move {
        axum::serve(listener, router(state)).await.unwrap();
    });
    (format!("http://{address}"), task, path)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cli_queries_a_real_wired_server_and_renders_graph() {
    let (url, task, path) = start_server().await;
    let query = Command::cargo_bin("gauge")
        .unwrap()
        .args(["--url", &url, "--json", "query", "cpu", "--time", "1000"])
        .output()
        .unwrap();
    assert!(query.status.success());
    let query_body = String::from_utf8(query.stdout).unwrap();
    assert!(query_body.contains("\"status\":\"success\""));
    assert!(query_body.contains("\"value\":2.5"));

    let graph = Command::cargo_bin("gauge")
        .unwrap()
        .args([
            "--url", &url, "graph", "cpu", "--start", "1000", "--end", "3000", "--width", "28",
        ])
        .output()
        .unwrap();
    assert!(graph.status.success());
    let graph_body = String::from_utf8(graph.stdout).unwrap();
    assert!(graph_body.contains("cpu{job=\"fixture\"}"));
    assert!(graph_body.contains('█'));
    assert!(!graph_body.contains("NaN"));

    task.abort();
    let _ = std::fs::remove_dir_all(path);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cli_surfaces_positioned_server_errors_and_connection_url() {
    let (url, task, path) = start_server().await;
    let error = Command::cargo_bin("gauge")
        .unwrap()
        .args(["--url", &url, "query", "cpu{", "--time", "1000"])
        .output()
        .unwrap();
    assert!(!error.status.success());
    let stderr = String::from_utf8(error.stderr).unwrap();
    assert!(stderr.contains("position"));

    task.abort();
    let _ = std::fs::remove_dir_all(path);
    let refused = Command::cargo_bin("gauge")
        .unwrap()
        .args(["--url", "http://127.0.0.1:1", "targets"])
        .output()
        .unwrap();
    let stderr = String::from_utf8(refused.stderr).unwrap();
    assert!(stderr.contains("http://127.0.0.1:1"));
}

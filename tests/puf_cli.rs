use std::io::Write;
use std::process::{Command, Output, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;

use axum::http::{Method, StatusCode};
use axum::{extract::Request, middleware, response::IntoResponse, serve};
use pufferclone::api::router;
use pufferclone::Engine;
use tempfile::TempDir;
use tokio::sync::oneshot;

struct TestServer {
    url: String,
    shutdown: Option<oneshot::Sender<()>>,
    thread: Option<thread::JoinHandle<()>>,
    _data_dir: TempDir,
    upsert_requests: Arc<AtomicUsize>,
}

impl TestServer {
    fn start() -> Self {
        Self::start_with_rejection(None)
    }

    fn start_rejecting_after_first() -> Self {
        Self::start_with_rejection(Some(1))
    }

    fn start_with_rejection(reject_after: Option<usize>) -> Self {
        let data_dir = tempfile::tempdir().expect("data directory");
        let engine = std::sync::Arc::new(Engine::new(data_dir.path()).expect("engine"));
        let upsert_requests = Arc::new(AtomicUsize::new(0));
        let thread_upsert_requests = Arc::clone(&upsert_requests);
        let (address_tx, address_rx) = mpsc::channel();
        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        let thread = thread::spawn(move || {
            let runtime = tokio::runtime::Runtime::new().expect("runtime");
            runtime.block_on(async move {
                let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
                    .await
                    .expect("listener");
                address_tx
                    .send(listener.local_addr().expect("address"))
                    .expect("address receiver");
                let app = router(engine).layer(middleware::from_fn(
                    move |request: Request, next: middleware::Next| {
                        let upsert_path = request.method() == Method::POST
                            && request.uri().path().starts_with("/v1/namespaces/")
                            && !request.uri().path().ends_with("/query");
                        let request_number = upsert_path
                            .then(|| thread_upsert_requests.fetch_add(1, Ordering::SeqCst));
                        async move {
                            if request_number.is_some_and(|number| {
                                reject_after.is_some_and(|limit| number >= limit)
                            }) {
                                return (
                                    StatusCode::INTERNAL_SERVER_ERROR,
                                    "forced test rejection",
                                )
                                    .into_response();
                            }
                            next.run(request).await
                        }
                    },
                ));
                serve(listener, app)
                    .with_graceful_shutdown(async {
                        let _ = shutdown_rx.await;
                    })
                    .await
                    .expect("server");
            });
        });
        let address = address_rx.recv().expect("server address");
        Self {
            url: format!("http://{address}"),
            shutdown: Some(shutdown_tx),
            thread: Some(thread),
            _data_dir: data_dir,
            upsert_requests,
        }
    }

    fn upsert_requests(&self) -> usize {
        self.upsert_requests.load(Ordering::SeqCst)
    }
}

impl Drop for TestServer {
    fn drop(&mut self) {
        self.shutdown
            .take()
            .expect("shutdown sender")
            .send(())
            .expect("server shutdown");
        self.thread
            .take()
            .expect("server thread")
            .join()
            .expect("server join");
    }
}

fn puf(server: &TestServer, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_puf"))
        .arg("--url")
        .arg(&server.url)
        .args(args)
        .output()
        .expect("run puf")
}

fn puf_url(url: &str, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_puf"))
        .arg("--url")
        .arg(url)
        .args(args)
        .output()
        .expect("run puf")
}

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

#[test]
fn puf_drives_real_server_for_writes_queries_and_namespace_lifecycle() {
    let server = TestServer::start();
    let input = tempfile::NamedTempFile::new().expect("JSONL file");
    std::fs::write(
        input.path(),
        concat!(
            "{\"id\":\"rust\",\"vector\":[1,0],\"text\":\"rust systems language\",\"attrs\":{\"category\":\"code\",\"kind\":\"a\"}}\n",
            "{\"id\":\"python\",\"vector\":[0,1],\"text\":\"python scripting language\",\"attrs\":{\"category\":\"code\",\"kind\":\"b\"}}\n",
            "{\"id\":\"rust-tool\",\"vector\":[0.9,0.1],\"text\":\"rust search engine\",\"attrs\":{\"category\":\"search\",\"kind\":\"a\"}}\n",
        ),
    )
    .expect("write JSONL");

    let output = puf(
        &server,
        &[
            "--json",
            "upsert",
            "demo",
            "--file",
            input.path().to_str().expect("path"),
        ],
    );
    assert!(output.status.success(), "{}", stderr(&output));
    let summary: serde_json::Value = serde_json::from_str(&stdout(&output)).expect("summary JSON");
    assert_eq!(summary["upserted"], 3);
    assert_eq!(summary["deleted"], 0);

    let output = puf(&server, &["--json", "ns", "ls"]);
    assert!(output.status.success(), "{}", stderr(&output));
    let namespaces: serde_json::Value =
        serde_json::from_str(&stdout(&output)).expect("namespace JSON");
    assert_eq!(namespaces["namespaces"], serde_json::json!(["demo"]));

    for args in [
        vec!["--json", "query", "demo", "--text", "rust"],
        vec!["--json", "query", "demo", "--vector", "1,0"],
        vec![
            "--json", "query", "demo", "--text", "rust", "--vector", "1,0", "--top-k", "2",
        ],
    ] {
        let output = puf(&server, &args);
        assert!(output.status.success(), "{}", stderr(&output));
        let response: serde_json::Value =
            serde_json::from_str(&stdout(&output)).expect("query JSON");
        assert!(!response["results"].as_array().expect("results").is_empty());
    }

    let vector_file = tempfile::NamedTempFile::new().expect("vector file");
    std::fs::write(vector_file.path(), "[1, 0]").expect("write vector");
    let output = puf(
        &server,
        &[
            "--json",
            "query",
            "demo",
            "--vector-file",
            vector_file.path().to_str().expect("path"),
        ],
    );
    assert!(output.status.success(), "{}", stderr(&output));

    for args in [
        vec!["query", "demo", "--vector", "NaN"],
        vec!["query", "demo", "--vector", "inf"],
    ] {
        let output = puf(&server, &args);
        assert!(!output.status.success());
        assert!(
            stderr(&output).contains("finite values"),
            "{}",
            stderr(&output)
        );
    }
    let empty_vector_file = tempfile::NamedTempFile::new().expect("empty vector file");
    std::fs::write(empty_vector_file.path(), "[]").expect("write empty vector");
    let output = puf(
        &server,
        &[
            "query",
            "demo",
            "--vector-file",
            empty_vector_file.path().to_str().expect("path"),
        ],
    );
    assert!(!output.status.success());
    assert!(
        stderr(&output).contains("must not be empty"),
        "{}",
        stderr(&output)
    );

    let output = puf(
        &server,
        &[
            "--json",
            "query",
            "demo",
            "--text",
            "rust",
            "--filter",
            "category=search",
            "--filter-in",
            "kind=a,b",
            "--top-k",
            "1",
        ],
    );
    assert!(output.status.success(), "{}", stderr(&output));
    let response: serde_json::Value =
        serde_json::from_str(&stdout(&output)).expect("filtered query JSON");
    assert_eq!(response["results"][0]["id"], "rust-tool");
    assert_eq!(response["results"][0]["attributes"]["category"], "search");

    let output = puf(&server, &["ns", "rm", "demo", "--yes"]);
    assert!(output.status.success(), "{}", stderr(&output));
    let output = puf(&server, &["--json", "ns", "ls"]);
    assert!(output.status.success(), "{}", stderr(&output));
    let namespaces: serde_json::Value =
        serde_json::from_str(&stdout(&output)).expect("namespace JSON");
    assert_eq!(namespaces["namespaces"], serde_json::json!([]));

    let output = puf(&server, &["query", "missing", "--text", "rust"]);
    assert!(!output.status.success());
    assert!(stderr(&output).contains("server returned HTTP 404"));
    assert!(stderr(&output).contains("namespace not found"));
}

#[test]
fn puf_reports_partial_jsonl_failures_and_rejects_invalid_inputs() {
    let server = TestServer::start();
    let input = tempfile::NamedTempFile::new().expect("JSONL file");
    let mut contents = String::new();
    for id in 0..100 {
        contents.push_str(&format!(
            "{{\"id\":\"doc-{id}\",\"vector\":[1,0],\"text\":\"indexed\",\"attrs\":{{\"kind\":\"a\"}}}}\n"
        ));
    }
    contents.push_str("not json\n");
    contents.push_str("{\"id\":\"bad-schema\",\"attrs\":{\"text\":42}}\n");
    for id in 0..99 {
        contents.push_str(&format!(
            "{{\"id\":\"tail-{id}\",\"vector\":[1,0],\"attrs\":{{\"kind\":\"tail\"}}}}\n"
        ));
    }
    contents.push_str("{\"id\":\"not-attempted-1\",\"vector\":[1,0],\"attrs\":{}}\n");
    contents.push_str("{\"id\":\"not-attempted-2\",\"vector\":[1,0],\"attrs\":{}}\n");
    std::fs::write(input.path(), contents).expect("write JSONL");

    let output = puf(
        &server,
        &[
            "--json",
            "upsert",
            "bulk",
            "--file",
            input.path().to_str().expect("path"),
        ],
    );
    assert!(!output.status.success());
    let summary: serde_json::Value = serde_json::from_str(&stdout(&output)).expect("summary JSON");
    assert_eq!(summary["sent"], 100);
    assert_eq!(summary["total_lines"], 203);
    assert_eq!(summary["failed"], 101);
    assert_eq!(summary["unknown"], 0);
    assert_eq!(summary["not_attempted"], 2);
    let failed_lines = summary["failed_lines"].as_array().expect("failed lines");
    assert_eq!(failed_lines.len(), 101);
    assert_eq!(failed_lines[0], 101);
    assert_eq!(failed_lines[1], 102);
    assert_eq!(failed_lines[100], 201);
    assert_eq!(
        summary["not_attempted_lines"],
        serde_json::json!([202, 203])
    );
    assert!(stderr(&output).contains("line 101"), "{}", stderr(&output));
    assert!(
        stderr(&output).contains("server returned HTTP 400"),
        "{}",
        stderr(&output)
    );
    assert!(stderr(&output).contains("full-text field 'text' must be a string"));

    let invalid_vector = tempfile::NamedTempFile::new().expect("invalid vector JSONL");
    std::fs::write(
        invalid_vector.path(),
        "{\"id\":\"empty\",\"vector\":[],\"attrs\":{}}\n",
    )
    .expect("write invalid vector JSONL");
    let output = puf(
        &server,
        &[
            "--json",
            "upsert",
            "bulk",
            "--file",
            invalid_vector.path().to_str().expect("path"),
        ],
    );
    assert!(!output.status.success());
    let summary: serde_json::Value = serde_json::from_str(&stdout(&output)).expect("summary JSON");
    assert_eq!(summary["failed"], 1);
    assert!(stderr(&output).contains("upsert vector must not be empty"));

    for args in [
        vec!["query", "bulk", "--vector", "1,0", "--filter", "missing"],
        vec!["query", "bulk", "--vector", "1,0", "--filter", "=value"],
        vec!["query", "bulk", "--vector", "1,0", "--filter-in", "kind="],
        vec![
            "query",
            "bulk",
            "--vector",
            "1,0",
            "--filter-in",
            "kind=a,,b",
        ],
    ] {
        let output = puf(&server, &args);
        assert!(!output.status.success());
        assert!(
            stderr(&output).contains("must use") || stderr(&output).contains("must not be empty"),
            "{}",
            stderr(&output)
        );
    }
}

#[test]
fn puf_compiled_cli_accounts_invalid_utf8_read_error_and_trailing_records() {
    let server = TestServer::start();
    let input = tempfile::NamedTempFile::new().expect("JSONL file");
    let mut contents = Vec::new();
    for id in 0..100 {
        contents
            .extend_from_slice(format!("{{\"id\":\"utf-{id}\",\"vector\":[1,0]}}\n").as_bytes());
    }
    contents.extend_from_slice(b"\xff\xfe\n");
    contents.extend_from_slice(b"{\"id\":\"trailing-1\",\"vector\":[1,0]}\n");
    contents.extend_from_slice(b"{\"id\":\"trailing-2\",\"vector\":[1,0]}\n");
    std::fs::write(input.path(), contents).expect("write invalid UTF-8 JSONL");

    let output = puf(
        &server,
        &[
            "--json",
            "upsert",
            "invalid-utf8",
            "--file",
            input.path().to_str().expect("path"),
        ],
    );
    assert!(!output.status.success());
    let summary: serde_json::Value = serde_json::from_slice(&output.stdout).expect("summary JSON");
    assert_eq!(summary["total_lines"], 103);
    assert_eq!(summary["ok"], 100);
    assert_eq!(summary["failed"], 1);
    assert_eq!(summary["failed_lines"], serde_json::json!([101]));
    assert_eq!(summary["unknown"], 0);
    assert_eq!(summary["unknown_lines"], serde_json::json!([]));
    assert_eq!(summary["not_attempted"], 2);
    assert_eq!(
        summary["not_attempted_lines"],
        serde_json::json!([102, 103])
    );
    assert_eq!(summary["input_truncated_by_abort"], false);
    assert_eq!(server.upsert_requests(), 1, "no retry after read error");
    assert!(stderr(&output).contains("line 101"), "{}", stderr(&output));
}

#[test]
fn puf_send_abort_accounts_trailing_file_lines_without_retrying() {
    let server = TestServer::start_rejecting_after_first();
    let input = tempfile::NamedTempFile::new().expect("JSONL file");
    let mut contents = String::new();
    for id in 0..100 {
        contents.push_str(&format!(
            "{{\"id\":\"ok-{id}\",\"vector\":[1,0],\"text\":\"indexed\"}}\n"
        ));
    }
    contents.push_str("{\"id\":\"reject\",\"attrs\":{\"text\":42}}\n");
    for id in 0..99 {
        contents.push_str(&format!("{{\"id\":\"rejected-{id}\",\"vector\":[1,0]}}\n"));
    }
    contents.push_str("{\"id\":\"trailing-1\",\"vector\":[1,0]}\n");
    contents.push_str("{\"id\":\"trailing-2\",\"vector\":[1,0]}\n");
    std::fs::write(input.path(), contents).expect("write JSONL");

    let output = puf(
        &server,
        &[
            "--json",
            "upsert",
            "send-abort",
            "--file",
            input.path().to_str().expect("path"),
        ],
    );
    assert!(!output.status.success());
    let summary: serde_json::Value = serde_json::from_str(&stdout(&output)).expect("summary JSON");
    assert_eq!(summary["total_lines"], 202);
    assert_eq!(summary["ok"], 100);
    assert_eq!(summary["failed"], 100);
    assert_eq!(summary["unknown"], 0);
    assert_eq!(summary["not_attempted"], 2);
    let failed_lines = summary["failed_lines"].as_array().expect("failed lines");
    assert_eq!(failed_lines.len(), 100);
    assert_eq!(failed_lines[0], 101);
    assert_eq!(failed_lines[99], 200);
    assert_eq!(
        summary["not_attempted_lines"],
        serde_json::json!([201, 202])
    );
    assert_eq!(server.upsert_requests(), 2, "no retry after rejection");
    assert!(stderr(&output).contains("server returned HTTP 500"));
    assert!(stderr(&output).contains("forced test rejection"));
}

#[test]
fn puf_stdin_abort_does_not_wait_for_eof() {
    let server = TestServer::start_rejecting_after_first();
    let mut child = Command::new(env!("CARGO_BIN_EXE_puf"))
        .args(["--url", &server.url, "--json", "upsert", "held-open"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn puf");
    let mut input = child.stdin.take().expect("puf stdin");
    for id in 0..100 {
        writeln!(
            input,
            "{{\"id\":\"ok-{id}\",\"vector\":[1,0],\"text\":\"indexed\"}}"
        )
        .expect("write first batch");
    }
    writeln!(input, "{{\"id\":\"reject\",\"attrs\":{{\"text\":42}}}}")
        .expect("write rejecting document");
    for id in 0..99 {
        writeln!(input, "{{\"id\":\"rejected-{id}\",\"vector\":[1,0]}}")
            .expect("write second batch");
    }
    input.flush().expect("flush held-open input");

    let mut exited = false;
    for _ in 0..300 {
        if child.try_wait().expect("poll puf").is_some() {
            exited = true;
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    if !exited {
        child.kill().expect("kill hung puf");
    }
    drop(input);
    let output = child.wait_with_output().expect("collect puf output");
    assert!(exited, "puf waited for EOF after stdin abort");
    assert!(!output.status.success());
    let summary: serde_json::Value = serde_json::from_slice(&output.stdout).expect("summary JSON");
    assert_eq!(summary["ok"], 100);
    assert_eq!(summary["failed"], 100);
    assert_eq!(summary["unknown"], 0);
    assert_eq!(summary["not_attempted"], 0);
    assert_eq!(summary["input_truncated_by_abort"], true);
    assert_eq!(server.upsert_requests(), 2, "no retry after rejection");
}

#[test]
fn puf_names_connection_url_when_server_is_down() {
    let url = "http://127.0.0.1:0";
    let output = puf_url(url, &["ns", "ls"]);
    assert!(!output.status.success());
    assert!(stderr(&output).contains(url), "{}", stderr(&output));

    let input = tempfile::NamedTempFile::new().expect("down-server JSONL");
    std::fs::write(
        input.path(),
        "{\"id\":\"unknown\",\"vector\":[1,0],\"attrs\":{}}\n",
    )
    .expect("write down-server JSONL");
    let output = puf_url(
        url,
        &[
            "--json",
            "upsert",
            "down",
            "--file",
            input.path().to_str().expect("path"),
        ],
    );
    assert!(!output.status.success());
    let summary: serde_json::Value = serde_json::from_str(&stdout(&output)).expect("summary JSON");
    assert_eq!(summary["ok"], 0);
    assert_eq!(summary["failed"], 0);
    assert_eq!(summary["unknown"], 1);
    assert_eq!(summary["unknown_lines"], serde_json::json!([1]));
    assert!(
        stderr(&output).contains("request to"),
        "{}",
        stderr(&output)
    );
}

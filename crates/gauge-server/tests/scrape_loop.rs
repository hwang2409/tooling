use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::body::to_bytes;
use axum::http::Request;
use gauge_server::config::{Config, DurationValue, TargetConfig};
use gauge_server::scrape::{MAX_SCRAPE_BODY_BYTES, spawn_scrapers};
use gauge_server::server::{AppState, TargetStatus, router};
use gauge_store::{GaugeStore, Matcher, Sample};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tower::ServiceExt;

static NEXT_PATH: AtomicU64 = AtomicU64::new(0);

struct Fixture {
    address: std::net::SocketAddr,
    phases: Arc<Mutex<BTreeMap<String, u8>>>,
    request_times: Arc<Mutex<BTreeMap<String, Vec<Instant>>>>,
    task: tokio::task::JoinHandle<()>,
}

impl Fixture {
    async fn start() -> Self {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let phases = Arc::new(Mutex::new(BTreeMap::new()));
        let request_times = Arc::new(Mutex::new(BTreeMap::new()));
        let server_phases = Arc::clone(&phases);
        let server_times = Arc::clone(&request_times);
        let task = tokio::spawn(async move {
            loop {
                let Ok((stream, _)) = listener.accept().await else {
                    break;
                };
                let phases = Arc::clone(&server_phases);
                let request_times = Arc::clone(&server_times);
                tokio::spawn(async move {
                    let _ = serve_fixture_connection(stream, phases, request_times).await;
                });
            }
        });
        Self {
            address,
            phases,
            request_times,
            task,
        }
    }

    fn url(&self, key: &str) -> String {
        format!("http://{}/metrics?target={key}", self.address)
    }

    fn set_phase(&self, key: &str, phase: u8) {
        self.phases.lock().unwrap().insert(key.to_owned(), phase);
    }

    fn times(&self, key: &str) -> Vec<Instant> {
        self.request_times
            .lock()
            .unwrap()
            .get(key)
            .cloned()
            .unwrap_or_default()
    }

    fn stop(self) {
        self.task.abort();
    }
}

async fn serve_fixture_connection(
    mut stream: tokio::net::TcpStream,
    phases: Arc<Mutex<BTreeMap<String, u8>>>,
    request_times: Arc<Mutex<BTreeMap<String, Vec<Instant>>>>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut request = Vec::new();
    let mut buffer = [0_u8; 256];
    while !request.windows(4).any(|window| window == b"\r\n\r\n") {
        let read = stream.read(&mut buffer).await?;
        if read == 0 {
            return Ok(());
        }
        request.extend_from_slice(&buffer[..read]);
    }
    let request = String::from_utf8_lossy(&request);
    let path = request
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .unwrap_or("/metrics");
    let key = path
        .split_once("target=")
        .map(|(_, value)| value.split('&').next().unwrap_or(value))
        .unwrap_or("default")
        .to_owned();
    request_times
        .lock()
        .unwrap()
        .entry(key.clone())
        .or_default()
        .push(Instant::now());
    let phase = phases.lock().unwrap().get(&key).copied().unwrap_or(0);
    if phase == 5 {
        tokio::time::sleep(Duration::from_millis(500)).await;
        return Ok(());
    }
    if phase == 4 {
        tokio::time::sleep(Duration::from_millis(80)).await;
    }
    if phase == 2 {
        stream
            .write_all(b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 4\r\nConnection: close\r\n\r\ndown")
            .await?;
        return Ok(());
    }
    if phase == 8 {
        stream
            .write_all(
                b"HTTP/1.1 302 Found\r\nLocation: http://127.0.0.1:1/elsewhere\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
            )
            .await?;
        return Ok(());
    }
    let body = if phase == 7 {
        "x".repeat(MAX_SCRAPE_BODY_BYTES + 1)
    } else {
        match phase {
            1 => String::new(),
            3 => "shared{target=\"wrong\",city=\"Zürich\"} 2\n".to_owned(),
            6 => "shared{target=\"wrong\",city=\"Zürich\"} 1\n".to_owned(),
            _ => "shared{target=\"wrong\",city=\"Zürich\"} 1\n".to_owned(),
        }
    };
    if phase == 6 {
        stream
            .write_all(
                b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n",
            )
            .await?;
        for chunk in body.as_bytes().chunks(5) {
            stream
                .write_all(format!("{:x}\r\n", chunk.len()).as_bytes())
                .await?;
            stream.write_all(chunk).await?;
            stream.write_all(b"\r\n").await?;
            tokio::task::yield_now().await;
        }
        stream.write_all(b"0\r\n\r\n").await?;
    } else {
        stream
            .write_all(
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                )
                .as_bytes(),
            )
            .await?;
    }
    Ok(())
}

fn make_state(
    fixture: &Fixture,
    targets: &[&str],
    interval: Duration,
    timeout: Duration,
) -> AppState {
    let path = std::env::temp_dir().join(format!(
        "gauge-server-loop-{}-{}",
        std::process::id(),
        NEXT_PATH.fetch_add(1, Ordering::Relaxed)
    ));
    let store = GaugeStore::open(path).unwrap();
    let config = Config {
        targets: targets
            .iter()
            .map(|name| TargetConfig {
                name: (*name).to_owned(),
                url: fixture.url(name),
                interval: DurationValue(interval),
            })
            .collect(),
        scrape_timeout: DurationValue(timeout),
        storage: Default::default(),
    };
    let state = AppState::new(
        store,
        config
            .targets
            .iter()
            .map(|target| (target.name.clone(), target.url.clone())),
    );
    spawn_scrapers(&config, state.clone());
    state
}

async fn wait_for_status<F>(state: &AppState, target: &str, predicate: F)
where
    F: Fn(&TargetStatus) -> bool,
{
    for _ in 0..300 {
        let status = state.targets.read().unwrap().get(target).cloned();
        if status.as_ref().is_some_and(&predicate) {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("timed out waiting for target {target}");
}

fn samples(state: &AppState, name: &str, target: &str) -> Vec<Sample> {
    state
        .store
        .select(
            &[Matcher::metric_name(name), Matcher::exact("target", target)],
            0,
            i64::MAX,
        )
        .unwrap()
        .into_iter()
        .flat_map(|series| series.samples)
        .collect()
}

async fn wait_for_samples<F>(state: &AppState, name: &str, target: &str, predicate: F)
where
    F: Fn(&[Sample]) -> bool,
{
    for _ in 0..300 {
        let values = samples(state, name, target);
        if predicate(&values) {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("timed out waiting for samples {name} target {target}");
}

#[tokio::test]
async fn real_loop_handles_chunked_split_response_and_utf8_labels() {
    let fixture = Fixture::start().await;
    fixture.set_phase("chunked", 6);
    let state = make_state(
        &fixture,
        &["chunked"],
        Duration::from_millis(50),
        Duration::from_secs(1),
    );
    wait_for_samples(&state, "shared", "chunked", |values| !values.is_empty()).await;
    let stored = state
        .store
        .select(
            &[
                Matcher::metric_name("shared"),
                Matcher::exact("target", "chunked"),
            ],
            0,
            i64::MAX,
        )
        .unwrap();
    assert_eq!(stored[0].series.labels["city"], "Zürich");
    assert_eq!(stored[0].series.labels["target"], "chunked");
    fixture.stop();
}

#[tokio::test]
async fn real_loop_records_up_transition() {
    let fixture = Fixture::start().await;
    fixture.set_phase("transition", 0);
    let state = make_state(
        &fixture,
        &["transition"],
        Duration::from_millis(50),
        Duration::from_secs(1),
    );
    wait_for_status(&state, "transition", |status| status.up).await;
    fixture.set_phase("transition", 2);
    wait_for_status(&state, "transition", |status| !status.up).await;
    fixture.set_phase("transition", 3);
    wait_for_status(&state, "transition", |status| status.up).await;
    wait_for_samples(&state, "up", "transition", |values| {
        let values: Vec<_> = values.iter().map(|sample| sample.value as i64).collect();
        values.windows(3).any(|window| window == [1, 0, 1])
    })
    .await;
    fixture.stop();
}

#[tokio::test]
async fn real_loop_marks_only_seen_series_stale_per_target() {
    let fixture = Fixture::start().await;
    fixture.set_phase("a", 0);
    fixture.set_phase("b", 0);
    let state = make_state(
        &fixture,
        &["a", "b"],
        Duration::from_millis(50),
        Duration::from_secs(1),
    );
    wait_for_status(&state, "a", |status| status.up).await;
    wait_for_status(&state, "b", |status| status.up).await;
    fixture.set_phase("a", 1);
    wait_for_samples(&state, "shared", "a", |values| {
        values.iter().any(|sample| sample.value.is_nan())
    })
    .await;
    let b_values = samples(&state, "shared", "b");
    assert!(!b_values.is_empty());
    assert!(b_values.iter().all(|sample| !sample.value.is_nan()));
    let a_values = samples(&state, "shared", "a");
    assert!(a_values.iter().any(|sample| sample.value.is_nan()));
    fixture.stop();
}

#[tokio::test]
async fn real_loop_uses_fixed_deadlines_with_slow_responses() {
    let fixture = Fixture::start().await;
    fixture.set_phase("cadence", 4);
    let state = make_state(
        &fixture,
        &["cadence"],
        Duration::from_millis(100),
        Duration::from_secs(1),
    );
    for _ in 0..300 {
        if fixture.times("cadence").len() >= 5 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let times = fixture.times("cadence");
    assert!(
        times.len() >= 5,
        "fixture saw only {} requests",
        times.len()
    );
    for pair in times.windows(2) {
        let gap = pair[1].duration_since(pair[0]);
        assert!(
            gap >= Duration::from_millis(70),
            "cadence too fast: {gap:?}"
        );
        assert!(
            gap <= Duration::from_millis(160),
            "scrape duration leaked into cadence: {gap:?}"
        );
    }
    drop(state);
    fixture.stop();
}

#[tokio::test]
async fn real_loop_enforces_timeout() {
    let fixture = Fixture::start().await;
    fixture.set_phase("timeout", 5);
    let state = make_state(
        &fixture,
        &["timeout"],
        Duration::from_millis(50),
        Duration::from_millis(40),
    );
    wait_for_status(&state, "timeout", |status| {
        !status.up
            && status
                .last_error
                .as_deref()
                .is_some_and(|error| error.contains("timed out"))
    })
    .await;
    fixture.stop();
}

#[tokio::test]
async fn real_loop_rejects_redirects_without_following_them() {
    let fixture = Fixture::start().await;
    fixture.set_phase("redirect", 8);
    let state = make_state(
        &fixture,
        &["redirect"],
        Duration::from_millis(50),
        Duration::from_secs(1),
    );
    wait_for_status(&state, "redirect", |status| {
        !status.up
            && status
                .last_error
                .as_deref()
                .is_some_and(|error| error.contains("target returned 302"))
    })
    .await;
    fixture.stop();
}

#[tokio::test]
async fn real_loop_rejects_oversized_body_with_bounded_streaming_read() {
    let fixture = Fixture::start().await;
    fixture.set_phase("oversized", 7);
    let state = make_state(
        &fixture,
        &["oversized"],
        Duration::from_millis(50),
        Duration::from_secs(2),
    );
    wait_for_status(&state, "oversized", |status| {
        !status.up
            && status
                .last_error
                .as_deref()
                .is_some_and(|error| error.contains("scrape body exceeds"))
    })
    .await;
    fixture.stop();
}

#[tokio::test]
async fn wired_query_api_returns_scraped_data() {
    let fixture = Fixture::start().await;
    let state = make_state(
        &fixture,
        &["wired"],
        Duration::from_millis(25),
        Duration::from_secs(1),
    );
    wait_for_samples(&state, "shared", "wired", |samples| !samples.is_empty()).await;
    let timestamp = samples(&state, "shared", "wired")[0].timestamp;
    let response = router(state)
        .oneshot(
            Request::get(format!(
                "/api/query?expr=shared%7Btarget%3D%22wired%22%7D&time={timestamp}"
            ))
            .body(Body::empty())
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let body = String::from_utf8(body.to_vec()).unwrap();
    assert!(body.contains("\"status\":\"success\""));
    assert!(body.contains("\"value\":1.0"));
    fixture.stop();
}

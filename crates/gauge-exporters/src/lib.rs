//! Standalone Prometheus text exposition exporters.
//!
//! This crate intentionally has no dependency on gauge's storage engine. The
//! small HTTP server is enough for scrape-only, localhost exporters.

pub mod node;
pub mod proc;

use std::io::{self, BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Mutex, RwLock, TryLockError};
use std::thread;
use std::time::{Duration, Instant};

/// The default loopback address used by both exporters.
pub const DEFAULT_BIND: &str = "127.0.0.1";

/// A command-line listening configuration shared by both binaries.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ListenOptions {
    pub bind: String,
    pub port: u16,
}

impl ListenOptions {
    pub fn address(&self) -> String {
        if self.bind.contains(':') && !self.bind.starts_with('[') {
            format!("[{}]:{}", self.bind, self.port)
        } else {
            format!("{}:{}", self.bind, self.port)
        }
    }
}

/// The HTTP response returned by an exporter.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MetricsResponse {
    status: &'static str,
    content_type: &'static str,
    body: Arc<str>,
}

impl MetricsResponse {
    pub fn status_code(&self) -> &'static str {
        self.status
    }

    pub fn body(&self) -> &str {
        &self.body
    }

    fn ok(body: String) -> Self {
        Self::ok_arc(Arc::from(body))
    }

    fn ok_arc(body: Arc<str>) -> Self {
        Self {
            status: "200 OK",
            content_type: "text/plain; version=0.0.4; charset=utf-8",
            body,
        }
    }

    fn internal_server_error() -> Self {
        Self {
            status: "500 Internal Server Error",
            content_type: "text/plain; charset=utf-8",
            body: Arc::from("metrics collection failed\n"),
        }
    }
}

/// Convert an exporter result into an HTTP response.
pub trait IntoMetricsResponse {
    fn into_metrics_response(self) -> MetricsResponse;
}

impl IntoMetricsResponse for String {
    fn into_metrics_response(self) -> MetricsResponse {
        MetricsResponse::ok(self)
    }
}

impl IntoMetricsResponse for MetricsResponse {
    fn into_metrics_response(self) -> MetricsResponse {
        self
    }
}

/// Caches an exporter response while coalescing concurrent refreshes.
///
/// Readers take only the short snapshot read lock. A separate refresh mutex
/// protects the collector, so a refresh never blocks readers serving a
/// last-good snapshot. Collection panics are contained; a recent last-good
/// body is served during the grace period, otherwise that scrape receives HTTP
/// 500 and the next scrape retries collection.
pub struct CachedExporter<F> {
    snapshot: RwLock<Snapshot>,
    refresher: Mutex<F>,
    ttl: Duration,
    grace: Duration,
}

#[derive(Clone, Default)]
struct Snapshot {
    body: Option<Arc<str>>,
    collected_at: Option<Instant>,
    last_good_at: Option<Instant>,
}

impl<F> CachedExporter<F>
where
    F: FnMut() -> String,
{
    pub fn new(exporter: F, ttl: Duration) -> Self {
        Self::with_grace(exporter, ttl, Duration::from_secs(5))
    }

    pub fn with_grace(exporter: F, ttl: Duration, grace: Duration) -> Self {
        Self {
            snapshot: RwLock::new(Snapshot::default()),
            refresher: Mutex::new(exporter),
            ttl,
            grace,
        }
    }

    pub fn scrape(&self) -> MetricsResponse {
        let snapshot = self.read_snapshot();
        if self.is_fresh(&snapshot) {
            return Self::response_from_snapshot(snapshot, self.grace);
        }

        let mut refresher = match self.refresher.try_lock() {
            Ok(refresher) => refresher,
            Err(TryLockError::WouldBlock) => {
                return Self::response_from_snapshot(snapshot, self.grace);
            }
            Err(TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
        };

        // Another refresher may have completed between the first read and
        // acquiring the guard. Avoid an unnecessary second collection.
        let snapshot = self.read_snapshot();
        if self.is_fresh(&snapshot) {
            return Self::response_from_snapshot(snapshot, self.grace);
        }

        match catch_unwind(AssertUnwindSafe(&mut *refresher)) {
            Ok(body) => {
                let body: Arc<str> = Arc::from(body);
                let now = Instant::now();
                let mut snapshot = self
                    .snapshot
                    .write()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                snapshot.body = Some(Arc::clone(&body));
                snapshot.collected_at = Some(now);
                snapshot.last_good_at = Some(now);
                MetricsResponse::ok_arc(body)
            }
            Err(_) => Self::response_from_snapshot(self.read_snapshot(), self.grace),
        }
    }

    fn read_snapshot(&self) -> Snapshot {
        self.snapshot
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    fn is_fresh(&self, snapshot: &Snapshot) -> bool {
        snapshot
            .collected_at
            .is_some_and(|collected_at| collected_at.elapsed() < self.ttl)
    }

    fn response_from_snapshot(snapshot: Snapshot, grace: Duration) -> MetricsResponse {
        if let (Some(body), Some(last_good_at)) = (snapshot.body, snapshot.last_good_at)
            && last_good_at.elapsed() <= grace
        {
            MetricsResponse::ok_arc(body)
        } else {
            MetricsResponse::internal_server_error()
        }
    }
}

/// Parse `--bind` and `--port` flags, returning the remaining arguments.
pub fn parse_listen_options(
    args: impl IntoIterator<Item = String>,
    default_port: u16,
) -> Result<(ListenOptions, Vec<String>), String> {
    let mut bind = DEFAULT_BIND.to_owned();
    let mut port = default_port;
    let mut rest = Vec::new();
    let mut args = args.into_iter();

    while let Some(arg) = args.next() {
        if arg == "--bind" {
            bind = args
                .next()
                .ok_or_else(|| "--bind requires an address".to_owned())?;
        } else if let Some(value) = arg.strip_prefix("--bind=") {
            bind = value.to_owned();
        } else if arg == "--port" {
            port = args
                .next()
                .ok_or_else(|| "--port requires a number".to_owned())?
                .parse()
                .map_err(|_| "--port must be a number from 0 to 65535".to_owned())?;
        } else if let Some(value) = arg.strip_prefix("--port=") {
            port = value
                .parse()
                .map_err(|_| "--port must be a number from 0 to 65535".to_owned())?;
        } else {
            rest.push(arg);
        }
    }

    Ok((ListenOptions { bind, port }, rest))
}

/// Serve GET `/metrics` requests. Each request is handled on its own thread.
pub fn serve<F, R>(listener: TcpListener, exporter: F) -> io::Result<()>
where
    F: Fn() -> R + Send + Sync + 'static,
    R: IntoMetricsResponse,
{
    let exporter = Arc::new(exporter);
    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let exporter = Arc::clone(&exporter);
                thread::spawn(move || {
                    if let Err(error) = handle_connection(stream, exporter) {
                        eprintln!("metrics connection failed: {error}");
                    }
                });
            }
            Err(error) => eprintln!("metrics listener failed: {error}"),
        }
    }
    Ok(())
}

fn handle_connection<R>(
    mut stream: TcpStream,
    exporter: Arc<dyn Fn() -> R + Send + Sync>,
) -> io::Result<()>
where
    R: IntoMetricsResponse,
{
    stream.set_read_timeout(Some(std::time::Duration::from_secs(5)))?;
    let mut request_line = String::new();
    BufReader::new(stream.try_clone()?).read_line(&mut request_line)?;
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or_default();
    let path = parts
        .next()
        .unwrap_or_default()
        .split('?')
        .next()
        .unwrap_or_default();

    let response = if method == "GET" && path == "/metrics" {
        exporter().into_metrics_response()
    } else if method != "GET" {
        MetricsResponse {
            status: "405 Method Not Allowed",
            content_type: "text/plain; charset=utf-8",
            body: Arc::from("method not allowed\n"),
        }
    } else {
        MetricsResponse {
            status: "404 Not Found",
            content_type: "text/plain; charset=utf-8",
            body: Arc::from("not found\n"),
        }
    };

    write!(
        stream,
        "HTTP/1.1 {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        response.status,
        response.content_type,
        response.body.len()
    )?;
    stream.write_all(response.body.as_bytes())
}

/// A small helper for writing a Prometheus metric with a fixed set of labels.
pub fn write_metric(
    output: &mut String,
    name: &str,
    labels: &[(&str, &str)],
    value: impl std::fmt::Display,
) {
    output.push_str(name);
    if !labels.is_empty() {
        output.push('{');
        for (index, (label, value)) in labels.iter().enumerate() {
            if index > 0 {
                output.push(',');
            }
            output.push_str(label);
            output.push_str("=\"");
            append_escaped_label(output, value);
            output.push('"');
        }
        output.push('}');
    }
    output.push(' ');
    output.push_str(&format_prometheus_value(value));
    output.push('\n');
}

fn format_prometheus_value(value: impl std::fmt::Display) -> String {
    match value.to_string().as_str() {
        "inf" => "+Inf".to_owned(),
        "-inf" => "-Inf".to_owned(),
        "nan" => "NaN".to_owned(),
        value => value.to_owned(),
    }
}

/// Write a Prometheus `# TYPE` declaration.
pub fn write_type(output: &mut String, name: &str, kind: &str) {
    output.push_str("# TYPE ");
    output.push_str(name);
    output.push(' ');
    output.push_str(kind);
    output.push('\n');
}

fn append_escaped_label(output: &mut String, value: &str) {
    for character in value.chars() {
        match character {
            '\\' => output.push_str("\\\\"),
            '\n' => output.push_str("\\n"),
            '"' => output.push_str("\\\""),
            character => output.push(character),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::NodeCollector;
    use crate::proc::ProcCollector;
    use std::collections::BTreeMap;
    use std::io::{Read, Write};
    use std::net::TcpStream;
    use std::sync::Arc;

    #[derive(Debug)]
    struct Sample {
        name: String,
        labels: BTreeMap<String, String>,
        value: f64,
    }

    fn strict_parse(input: &str) -> (BTreeMap<String, String>, Vec<Sample>) {
        let mut types = BTreeMap::new();
        let mut samples = Vec::new();
        for line in input.lines() {
            if line.is_empty() {
                continue;
            }
            if let Some(declaration) = line.strip_prefix("# TYPE ") {
                let fields: Vec<_> = declaration.split(' ').collect();
                assert_eq!(fields.len(), 2, "invalid TYPE line: {line}");
                assert!(valid_name(fields[0]));
                assert!(matches!(
                    fields[1],
                    "counter" | "gauge" | "histogram" | "summary"
                ));
                assert!(
                    types
                        .insert(fields[0].to_owned(), fields[1].to_owned())
                        .is_none()
                );
                continue;
            }
            assert!(!line.starts_with('#'), "unexpected comment: {line}");
            let (metric, value) = line.rsplit_once(' ').expect("metric needs a value");
            let value = match value {
                "+Inf" => f64::INFINITY,
                "-Inf" => f64::NEG_INFINITY,
                "NaN" => f64::NAN,
                _ => {
                    let value = value.parse::<f64>().expect("metric value must be a float");
                    assert!(value.is_finite(), "non-canonical special float: {value}");
                    value
                }
            };
            let (name, labels) = if let Some(open) = metric.find('{') {
                assert!(metric.ends_with('}'));
                let name = &metric[..open];
                (name, parse_labels(&metric[open + 1..metric.len() - 1]))
            } else {
                (metric, BTreeMap::new())
            };
            assert!(valid_name(name));
            samples.push(Sample {
                name: name.to_owned(),
                labels,
                value,
            });
        }
        (types, samples)
    }

    fn valid_name(name: &str) -> bool {
        let mut characters = name.chars();
        matches!(characters.next(), Some('_' | ':' | 'a'..='z' | 'A'..='Z'))
            && characters.all(|character| {
                character == '_' || character == ':' || character.is_ascii_alphanumeric()
            })
    }

    fn parse_labels(input: &str) -> BTreeMap<String, String> {
        let mut labels = BTreeMap::new();
        let mut rest = input;
        while !rest.is_empty() {
            let equals = rest.find('=').expect("label needs an equals sign");
            let name = &rest[..equals];
            assert!(valid_name(name));
            let quoted = &rest[equals + 1..];
            assert!(quoted.starts_with('"'));
            let mut value = String::new();
            let mut escaped = false;
            let mut end = None;
            for (index, character) in quoted[1..].char_indices() {
                if escaped {
                    value.push(match character {
                        'n' => '\n',
                        '\\' => '\\',
                        '"' => '"',
                        _ => panic!("invalid label escape"),
                    });
                    escaped = false;
                } else if character == '\\' {
                    escaped = true;
                } else if character == '"' {
                    end = Some(index + 2);
                    break;
                } else {
                    value.push(character);
                }
            }
            let end = end.expect("unterminated label");
            assert!(labels.insert(name.to_owned(), value).is_none());
            rest = &quoted[end..];
            if !rest.is_empty() {
                assert!(rest.starts_with(','));
                rest = &rest[1..];
            }
        }
        labels
    }

    #[test]
    fn labels_are_escaped() {
        let mut output = String::new();
        write_metric(&mut output, "example", &[("label", "a\\b\"c\nd")], 1.0);
        assert_eq!(output, "example{label=\"a\\\\b\\\"c\\nd\"} 1\n");
        let exposition = format!("# TYPE example gauge\n{output}");
        let (_, samples) = strict_parse(&exposition);
        assert_eq!(samples[0].labels["label"], "a\\b\"c\nd");
    }

    #[test]
    fn special_floats_use_prometheus_spelling() {
        let mut output = String::new();
        write_metric(&mut output, "positive", &[], f64::INFINITY);
        write_metric(&mut output, "negative", &[], f64::NEG_INFINITY);
        write_metric(&mut output, "not_a_number", &[], f64::NAN);
        assert_eq!(output, "positive +Inf\nnegative -Inf\nnot_a_number NaN\n");
        assert!(std::panic::catch_unwind(|| strict_parse("lower inf\n")).is_err());
        let (_, samples) = strict_parse(&output);
        assert!(samples[0].value.is_infinite());
        assert!(samples[2].value.is_nan());
    }

    #[test]
    fn node_exposition_is_strictly_parseable_and_sane() {
        let mut collector = NodeCollector::new();
        let output = collector.scrape();
        let (types, samples) = strict_parse(&output);
        assert!(!samples.is_empty());
        assert_eq!(types["node_cpu_percent"], "gauge");
        let cpu_values: Vec<_> = samples
            .iter()
            .filter(|sample| sample.name == "node_cpu_percent")
            .map(|sample| sample.value)
            .collect();
        assert!(!cpu_values.is_empty());
        assert!(cpu_values.iter().all(|value| (0.0..=100.0).contains(value)));
        let total_memory = samples
            .iter()
            .find(|sample| sample.name == "node_memory_total_bytes")
            .unwrap();
        assert!(total_memory.value > 0.0);
    }

    #[test]
    fn process_exposition_is_strictly_parseable() {
        let executable = std::env::current_exe().unwrap();
        let pattern = executable
            .file_name()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        let output = ProcCollector::new(vec![pattern]).scrape();
        let (types, samples) = strict_parse(&output);
        assert_eq!(types["proc_rss_bytes"], "gauge");
        assert!(
            samples
                .iter()
                .any(|sample| sample.name == "proc_rss_bytes" && sample.value > 0.0)
        );
    }

    #[test]
    fn process_cache_coalesces_real_concurrent_scrapes() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let address = listener.local_addr().unwrap();
        let pattern = std::env::current_exe()
            .unwrap()
            .file_name()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        let collector = ProcCollector::new(vec![pattern]);
        let cache = Arc::new(CachedExporter::with_grace(
            move || collector.scrape(),
            Duration::ZERO,
            Duration::from_secs(1),
        ));
        assert_eq!(cache.scrape().status_code(), "200 OK");
        let server = thread::spawn(move || {
            serve(listener, move || cache.scrape()).unwrap();
        });

        let started = Instant::now();
        let mut requests = Vec::new();
        for _ in 0..20 {
            requests.push(thread::spawn(move || {
                let mut stream = TcpStream::connect(address).unwrap();
                stream
                    .write_all(b"GET /metrics HTTP/1.1\r\nHost: localhost\r\n\r\n")
                    .unwrap();
                let mut response = String::new();
                stream.read_to_string(&mut response).unwrap();
                assert!(response.starts_with("HTTP/1.1 200 OK"));
                assert!(response.contains("proc_rss_bytes"));
            }));
        }
        for request in requests {
            request.join().unwrap();
        }
        assert!(
            started.elapsed() < Duration::from_secs(3),
            "real proc concurrent scrape took too long: {:?}",
            started.elapsed()
        );
        drop(server);
    }

    #[test]
    fn cache_recovers_after_panic_and_retries() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let calls = Arc::new(AtomicUsize::new(0));
        let closure_calls = Arc::clone(&calls);
        let cache = CachedExporter::with_grace(
            move || match closure_calls.fetch_add(1, Ordering::SeqCst) {
                0 => "initial\n".to_owned(),
                1 => panic!("injected collector failure"),
                _ => "recovered\n".to_owned(),
            },
            Duration::ZERO,
            Duration::from_secs(1),
        );

        assert_eq!(cache.scrape().status_code(), "200 OK");
        let fallback = cache.scrape();
        assert_eq!(fallback.status_code(), "200 OK");
        assert_eq!(fallback.body(), "initial\n");
        let recovered = cache.scrape();
        assert_eq!(recovered.status_code(), "200 OK");
        assert_eq!(recovered.body(), "recovered\n");

        let calls = Arc::new(AtomicUsize::new(0));
        let closure_calls = Arc::clone(&calls);
        let cache = CachedExporter::with_grace(
            move || match closure_calls.fetch_add(1, Ordering::SeqCst) {
                0 => panic!("injected initial collector failure"),
                _ => "retry succeeded\n".to_owned(),
            },
            Duration::ZERO,
            Duration::ZERO,
        );
        assert_eq!(cache.scrape().status_code(), "500 Internal Server Error");
        assert_eq!(cache.scrape().body(), "retry succeeded\n");
    }

    #[test]
    fn stale_snapshot_is_served_during_slow_refresh() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::{Arc, Barrier};

        let started = Arc::new(Barrier::new(2));
        let release = Arc::new(Barrier::new(2));
        let calls = Arc::new(AtomicUsize::new(0));
        let closure_started = Arc::clone(&started);
        let closure_release = Arc::clone(&release);
        let closure_calls = Arc::clone(&calls);
        let cache = Arc::new(CachedExporter::with_grace(
            move || {
                if closure_calls.fetch_add(1, Ordering::SeqCst) == 0 {
                    return "initial\n".to_owned();
                }
                closure_started.wait();
                closure_release.wait();
                "updated\n".to_owned()
            },
            Duration::ZERO,
            Duration::from_secs(1),
        ));
        assert_eq!(cache.scrape().body(), "initial\n");

        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let address = listener.local_addr().unwrap();
        let server_cache = Arc::clone(&cache);
        let server = thread::spawn(move || {
            serve(listener, move || server_cache.scrape()).unwrap();
        });
        let refresh_cache = Arc::clone(&cache);
        let refresh = thread::spawn(move || refresh_cache.scrape());
        started.wait();

        let request_started = Instant::now();
        let mut stream = TcpStream::connect(address).unwrap();
        stream
            .write_all(b"GET /metrics HTTP/1.1\r\nHost: localhost\r\n\r\n")
            .unwrap();
        let mut response = String::new();
        stream.read_to_string(&mut response).unwrap();
        assert!(request_started.elapsed() < Duration::from_secs(1));
        assert!(response.starts_with("HTTP/1.1 200 OK"));
        assert!(response.ends_with("initial\n"));

        release.wait();
        assert_eq!(refresh.join().unwrap().body(), "updated\n");
        drop(server);
    }

    #[test]
    fn proc_state_panic_resets_and_next_http_scrape_recovers() {
        let pattern = std::env::current_exe()
            .unwrap()
            .file_name()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        let collector = Arc::new(ProcCollector::new(vec![pattern]));
        let cache_collector = Arc::clone(&collector);
        let cache = Arc::new(CachedExporter::with_grace(
            move || cache_collector.scrape(),
            Duration::ZERO,
            Duration::ZERO,
        ));
        assert_eq!(cache.scrape().status_code(), "200 OK");
        collector.panic_inside_state_once();

        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let address = listener.local_addr().unwrap();
        let server_cache = Arc::clone(&cache);
        let server = thread::spawn(move || {
            serve(listener, move || server_cache.scrape()).unwrap();
        });

        let request = |address| {
            let mut stream = TcpStream::connect(address).unwrap();
            stream
                .write_all(b"GET /metrics HTTP/1.1\r\nHost: localhost\r\n\r\n")
                .unwrap();
            let mut response = String::new();
            stream.read_to_string(&mut response).unwrap();
            response
        };
        assert!(request(address).starts_with("HTTP/1.1 500 Internal Server Error"));
        let recovered = request(address);
        assert!(recovered.starts_with("HTTP/1.1 200 OK"));
        assert!(recovered.contains("proc_rss_bytes"));
        assert_eq!(collector.rebuild_count(), 1);
        let recovered_again = request(address);
        assert!(recovered_again.starts_with("HTTP/1.1 200 OK"));
        assert_eq!(collector.rebuild_count(), 1);
        drop(server);
    }

    #[test]
    fn concurrent_metrics_requests_are_served() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let address = listener.local_addr().unwrap();
        let mut collector = NodeCollector::new();
        let cache = Arc::new(CachedExporter::new(
            move || collector.scrape(),
            Duration::ZERO,
        ));
        assert_eq!(cache.scrape().status_code(), "200 OK");
        let server = thread::spawn(move || {
            serve(listener, move || cache.scrape()).unwrap();
        });

        let mut requests = Vec::new();
        for _ in 0..20 {
            requests.push(thread::spawn(move || {
                let mut stream = TcpStream::connect(address).unwrap();
                stream
                    .write_all(b"GET /metrics HTTP/1.1\r\nHost: localhost\r\n\r\n")
                    .unwrap();
                let mut response = String::new();
                stream.read_to_string(&mut response).unwrap();
                assert!(response.starts_with("HTTP/1.1 200 OK"));
                assert!(response.contains("node_memory_total_bytes "));
            }));
        }
        for request in requests {
            request.join().unwrap();
        }
        drop(server);
    }
}

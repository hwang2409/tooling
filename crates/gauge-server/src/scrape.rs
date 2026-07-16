use std::collections::{BTreeMap, BTreeSet};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::config::{Config, TargetConfig};
use crate::parser::{ParseResult, parse_exposition};
use crate::server::AppState;
use gauge_store::{GaugeStore, Sample, Series};

pub const JITTER_FRACTION: f64 = 0.10;
pub const MAX_SCRAPE_BODY_BYTES: usize = 4 * 1024 * 1024;

pub fn jittered_interval(interval: Duration, entropy: u64) -> Duration {
    let span = interval.mul_f64(JITTER_FRACTION);
    let lower = interval.saturating_sub(span);
    let upper = interval.saturating_add(span);
    if upper <= lower {
        return interval;
    }
    let range = upper.as_nanos().saturating_sub(lower.as_nanos());
    let offset = u128::from(splitmix64(entropy)) % (range + 1);
    Duration::from_nanos(
        lower
            .as_nanos()
            .saturating_add(offset)
            .min(u128::from(u64::MAX)) as u64,
    )
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e3779b97f4a7c15);
    let mut result = value;
    result = (result ^ (result >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    result = (result ^ (result >> 27)).wrapping_mul(0x94d049bb133111eb);
    result ^ (result >> 31)
}

pub fn spawn_scrapers(config: &Config, state: AppState) -> Vec<tokio::task::JoinHandle<()>> {
    config
        .targets
        .iter()
        .cloned()
        .map(|target| {
            let state = state.clone();
            let timeout = config.scrape_timeout.as_duration();
            tokio::spawn(async move { target_loop(target, timeout, state).await })
        })
        .collect()
}

async fn target_loop(target: TargetConfig, timeout: Duration, state: AppState) {
    let client = match reqwest::Client::builder()
        .use_rustls_tls()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(timeout)
        .build()
    {
        Ok(client) => client,
        Err(error) => {
            state.update_target(
                &target,
                false,
                now_millis(),
                Some(format!("HTTP client setup failed: {error}")),
            );
            return;
        }
    };
    let mut previous = BTreeSet::new();
    let mut absent = BTreeMap::<Series, u8>::new();
    let mut sequence = 0_u64;
    let interval = target.interval.as_duration();
    // Deadlines are advanced from the previous deadline, not from completion
    // of the scrape. If a slow scrape falls behind, skip missed deadlines and
    // schedule one interval from now; scrape time is never added to cadence.
    let mut next = Instant::now();
    loop {
        let entropy = sequence ^ hash_name(&target.name);
        sequence = sequence.wrapping_add(1);
        let delay = jittered_interval(target.interval.as_duration(), entropy);
        tokio::time::sleep_until(next.into()).await;
        let timestamp = now_millis();
        let outcome = scrape_once(&target, &client, timeout).await;
        let (mut writes, up, error) = match outcome {
            Ok(body) => {
                let parsed = parse_exposition(&body);
                state
                    .metrics
                    .malformed_lines
                    .fetch_add(parsed.malformed_lines, std::sync::atomic::Ordering::Relaxed);
                let writes = samples_for_success(
                    &parsed,
                    &target.name,
                    timestamp,
                    &mut previous,
                    &mut absent,
                );
                (writes, true, None)
            }
            Err(error) => {
                let writes = samples_for_failure(timestamp, &mut previous, &mut absent);
                (writes, false, Some(error))
            }
        };
        state
            .metrics
            .scrapes_total
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if !up {
            state
                .metrics
                .scrape_failures
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        let up_series = stored_series(Series::new("up", BTreeMap::new()), &target.name);
        writes.push((
            up_series,
            Sample::new(timestamp, if up { 1.0 } else { 0.0 }),
        ));
        let write_count = writes.len() as u64;
        append_without_blocking(
            state.store.clone(),
            writes,
            state.metrics.clone(),
            write_count,
        );
        state.update_target(&target, up, timestamp, error);
        next += delay;
        if next <= Instant::now() {
            next = Instant::now() + interval;
        }
    }
}

fn samples_for_success(
    parsed: &ParseResult,
    target_name: &str,
    timestamp: i64,
    previous: &mut BTreeSet<Series>,
    absent: &mut BTreeMap<Series, u8>,
) -> Vec<(Series, Sample)> {
    let current: BTreeSet<_> = parsed
        .samples
        .iter()
        .map(|sample| stored_series(sample.series.clone(), target_name))
        .collect();
    let mut writes = parsed
        .samples
        .iter()
        .map(|sample| {
            (
                stored_series(sample.series.clone(), target_name),
                Sample::new(timestamp, sample.value),
            )
        })
        .collect::<Vec<_>>();
    previous.extend(current.iter().cloned());
    let mut stale = Vec::new();
    for series in previous.iter() {
        if current.contains(series) {
            absent.remove(series);
        } else {
            let count = absent.entry(series.clone()).or_insert(0);
            *count = count.saturating_add(1);
            if *count >= 2 {
                // NaN is gauge's explicit staleness marker. It is retained as
                // a sample so query layers can distinguish absence from zero.
                writes.push((series.clone(), Sample::new(timestamp, f64::NAN)));
                stale.push(series.clone());
            }
        }
    }
    for series in stale {
        previous.remove(&series);
        absent.remove(&series);
    }
    writes
}

/// Every scraped series gets the reserved target identity label. A target's
/// own `target` label is intentionally overridden so identity is unambiguous;
/// this is the server's simple `honor_labels = false` policy.
fn stored_series(mut series: Series, target_name: &str) -> Series {
    series
        .labels
        .insert("target".to_owned(), target_name.to_owned());
    series
}

fn samples_for_failure(
    timestamp: i64,
    previous: &mut BTreeSet<Series>,
    absent: &mut BTreeMap<Series, u8>,
) -> Vec<(Series, Sample)> {
    let mut writes = Vec::new();
    let mut stale = Vec::new();
    for series in previous.iter() {
        let count = absent.entry(series.clone()).or_insert(0);
        *count = count.saturating_add(1);
        if *count >= 2 {
            writes.push((series.clone(), Sample::new(timestamp, f64::NAN)));
            stale.push(series.clone());
        }
    }
    for series in stale {
        previous.remove(&series);
        absent.remove(&series);
    }
    writes
}

// This function exists to make the "no flush in the scrape task" invariant
// obvious at the call site. The store write itself is also blocking (WAL fsync)
// and therefore leaves the async executor through spawn_blocking.
fn append_without_blocking(
    store: GaugeStore,
    writes: Vec<(Series, Sample)>,
    metrics: std::sync::Arc<crate::server::ServerMetrics>,
    write_count: u64,
) {
    tokio::spawn(async move {
        let succeeded = matches!(
            tokio::task::spawn_blocking(move || store.writer().append_batch(&writes)).await,
            Ok(Ok(()))
        );
        if succeeded {
            metrics
                .samples_written
                .fetch_add(write_count, std::sync::atomic::Ordering::Relaxed);
        } else {
            metrics
                .write_errors
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    });
}

async fn scrape_once(
    target: &TargetConfig,
    client: &reqwest::Client,
    timeout: Duration,
) -> Result<String, String> {
    tokio::time::timeout(timeout, fetch_http(client, &target.url))
        .await
        .map_err(|_| format!("scrape timed out after {:?}", timeout))?
}

async fn fetch_http(client: &reqwest::Client, url: &str) -> Result<String, String> {
    let mut response = client
        .get(url)
        .send()
        .await
        .map_err(|error| format!("HTTP scrape failed: {error}"))?;
    if !response.status().is_success() {
        return Err(format!("target returned {}", response.status()));
    }
    let mut body = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|error| format!("response body failed: {error}"))?
    {
        if chunk.len() > MAX_SCRAPE_BODY_BYTES.saturating_sub(body.len()) {
            return Err(format!("scrape body exceeds {MAX_SCRAPE_BODY_BYTES} bytes"));
        }
        body.extend_from_slice(&chunk);
    }
    String::from_utf8(body).map_err(|_| "response was not UTF-8".to_owned())
}

fn now_millis() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

fn hash_name(name: &str) -> u64 {
    name.bytes().fold(0xcbf29ce484222325, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::AsyncReadExt;
    use tokio::io::AsyncWriteExt;

    #[test]
    fn jitter_stays_within_ten_percent() {
        let interval = Duration::from_secs(100);
        for entropy in 0..1000 {
            let actual = jittered_interval(interval, entropy);
            assert!(actual >= Duration::from_secs(90));
            assert!(actual <= Duration::from_secs(110));
        }
    }

    #[test]
    fn staleness_marker_is_written_after_two_absences() {
        let mut previous = BTreeSet::new();
        let mut absent = BTreeMap::new();
        let first = parse_exposition("temperature 3\n");
        let first_writes = samples_for_success(&first, "fixture", 1, &mut previous, &mut absent);
        assert_eq!(first_writes.len(), 1);
        assert!(
            samples_for_success(
                &ParseResult::default(),
                "fixture",
                2,
                &mut previous,
                &mut absent,
            )
            .is_empty()
        );
        let stale = samples_for_success(
            &ParseResult::default(),
            "fixture",
            3,
            &mut previous,
            &mut absent,
        );
        assert_eq!(stale.len(), 1);
        assert!(stale[0].1.value.is_nan());
    }

    #[tokio::test]
    async fn fixture_http_server_is_scrapable() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = [0_u8; 256];
            let _ = stream.read(&mut request).await.unwrap();
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 14\r\nConnection: close\r\n\r\ntemperature 3\n")
                .await
                .unwrap();
        });
        let target = TargetConfig {
            name: "fixture".to_owned(),
            url: format!("http://{address}/metrics"),
            interval: crate::config::DurationValue(Duration::from_millis(10)),
        };
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .timeout(Duration::from_secs(1))
            .build()
            .unwrap();
        assert_eq!(
            scrape_once(&target, &client, Duration::from_secs(1))
                .await
                .unwrap(),
            "temperature 3\n"
        );
        server.await.unwrap();
    }
}

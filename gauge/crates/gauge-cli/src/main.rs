use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::io::Write as _;
use std::time::{SystemTime, UNIX_EPOCH};

use clap::{Args, Parser, Subcommand};
use gauge_query::{
    ApiErrorResponse, InstantQueryResponse, QueryPoint, RangeQueryResponse, SeriesQueryResponse,
};
use reqwest::blocking::Client;
use serde::Deserialize;
use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;

const DEFAULT_URL: &str = "http://127.0.0.1:8428";
const DEFAULT_STEP_MS: i64 = 15_000;
const DEFAULT_CHART_WIDTH: usize = 80;
const MIN_CHART_WIDTH: usize = 20;
const MAX_CHART_WIDTH: usize = 1_000;
const CHART_HEIGHT: usize = 16;
const CHART_PREFIX_WIDTH: usize = 12;

#[derive(Debug, Parser)]
#[command(name = "gauge", version, about = "Query a gauge metrics server")]
struct Cli {
    /// Gauge server base URL.
    #[arg(long, global = true, env = "GAUGE_URL", default_value = DEFAULT_URL)]
    url: String,

    /// Print the raw JSON response from the server.
    #[arg(long, global = true)]
    json: bool,

    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Evaluate an expression at one instant.
    Query(QueryArgs),
    /// Evaluate an expression over a time range.
    Range(RangeArgs),
    /// Render an expression as a terminal chart.
    Graph(GraphArgs),
    /// List series matching a selector.
    Series(SeriesArgs),
    /// Show scrape target status.
    Targets,
}

#[derive(Debug, Args)]
struct QueryArgs {
    expr: String,
    /// Evaluation timestamp in milliseconds, or a duration before now.
    #[arg(long)]
    time: Option<String>,
}

#[derive(Debug, Args)]
struct RangeArgs {
    expr: String,
    /// Query the most recent duration, such as 5m, 1h, or 2d.
    #[arg(long, conflicts_with_all = ["start", "end"])]
    last: Option<String>,
    /// Start timestamp in milliseconds, or a duration before now.
    #[arg(long, requires = "end", conflicts_with = "last")]
    start: Option<String>,
    /// End timestamp in milliseconds, or a duration before now.
    #[arg(long, requires = "start", conflicts_with = "last")]
    end: Option<String>,
    /// Evaluation interval, such as 15s.
    #[arg(long, default_value = "15s")]
    step: String,
}

#[derive(Debug, Args)]
struct GraphArgs {
    expr: String,
    /// Query the most recent duration, such as 5m, 1h, or 2d.
    #[arg(long, conflicts_with_all = ["start", "end"])]
    last: Option<String>,
    /// Start timestamp in milliseconds, or a duration before now.
    #[arg(long, requires = "end", conflicts_with = "last")]
    start: Option<String>,
    /// End timestamp in milliseconds, or a duration before now.
    #[arg(long, requires = "start", conflicts_with = "last")]
    end: Option<String>,
    /// Override the detected terminal width for the chart.
    #[arg(long, value_parser = parse_width)]
    width: Option<usize>,
}

#[derive(Debug, Args)]
struct SeriesArgs {
    #[arg(name = "match")]
    matcher: String,
}

#[derive(Debug, Deserialize)]
struct TargetStatus {
    target: String,
    url: String,
    up: bool,
    last_scrape_time: Option<i64>,
    last_error: Option<String>,
}

#[derive(Debug)]
struct ApiClient {
    base_url: String,
    client: Client,
}

impl ApiClient {
    fn new(base_url: String) -> Result<Self, String> {
        let base_url = base_url.trim_end_matches('/').to_owned();
        reqwest::Url::parse(&base_url)
            .map_err(|error| format!("invalid gauge URL '{base_url}': {error}"))?;
        let client = Client::builder()
            .build()
            .map_err(|error| format!("could not create HTTP client: {error}"))?;
        Ok(Self { base_url, client })
    }

    fn get(&self, path: &str, params: &[(&str, String)]) -> Result<Vec<u8>, String> {
        let url = format!("{}{}", self.base_url, path);
        let response = self
            .client
            .get(&url)
            .query(params)
            .send()
            .map_err(|error| format!("could not connect to {}: {error}", self.base_url))?;
        let status = response.status();
        let body = response
            .bytes()
            .map_err(|error| format!("could not read response from {}: {error}", self.base_url))?
            .to_vec();
        if !status.is_success() {
            return Err(format_server_error(status.as_u16(), &body));
        }
        Ok(body)
    }
}

fn main() {
    let cli = Cli::parse();
    if let Err(error) = run(cli) {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run(cli: Cli) -> Result<(), String> {
    let client = ApiClient::new(cli.url)?;
    match cli.command {
        Command::Query(args) => {
            let now = now_millis();
            let time = parse_instant(args.time.as_deref(), now)?;
            let body = client.get(
                "/api/query",
                &[("expr", args.expr), ("time", time.to_string())],
            )?;
            if cli.json {
                print_body(&body)
            } else {
                let response: InstantQueryResponse = decode_json(&body)?;
                print_instant(&response);
            }
        }
        Command::Range(args) => {
            let (start, end) = parse_range(&args, now_millis())?;
            let step = parse_duration(&args.step)?;
            let body = client.get(
                "/api/query_range",
                &[
                    ("expr", args.expr),
                    ("start", start.to_string()),
                    ("end", end.to_string()),
                    ("step", step.to_string()),
                ],
            )?;
            if cli.json {
                print_body(&body)
            } else {
                let response: RangeQueryResponse = decode_json(&body)?;
                print_range(&response);
            }
        }
        Command::Graph(args) => {
            let (start, end) = parse_graph_range(&args, now_millis())?;
            let width = args.width.unwrap_or_else(detect_terminal_width);
            let span = end.saturating_sub(start).max(1);
            let step = (span / (width as i64 * 2)).max(DEFAULT_STEP_MS);
            let body = client.get(
                "/api/query_range",
                &[
                    ("expr", args.expr),
                    ("start", start.to_string()),
                    ("end", end.to_string()),
                    ("step", step.to_string()),
                ],
            )?;
            if cli.json {
                print_body(&body)
            } else {
                let response: RangeQueryResponse = decode_json(&body)?;
                print!("{}", render_chart(&response, start, end, width));
            }
        }
        Command::Series(args) => {
            let body = client.get("/api/series", &[("match", args.matcher)])?;
            if cli.json {
                print_body(&body)
            } else {
                let response: SeriesQueryResponse = decode_json(&body)?;
                print_series(&response);
            }
        }
        Command::Targets => {
            let body = client.get("/api/targets", &[])?;
            if cli.json {
                print_body(&body)
            } else {
                let targets: Vec<TargetStatus> = decode_json(&body)?;
                print_targets(&targets);
            }
        }
    }
    Ok(())
}

fn decode_json<T: serde::de::DeserializeOwned>(body: &[u8]) -> Result<T, String> {
    serde_json::from_slice(body).map_err(|error| format!("invalid JSON from gauge server: {error}"))
}

fn print_body(body: &[u8]) {
    std::io::stdout()
        .write_all(body)
        .expect("stdout write failed");
}

fn print_instant(response: &InstantQueryResponse) {
    for result in &response.data.result {
        println!(
            "{}\t{}\t{}",
            format_metric(&result.metric),
            result.value.timestamp,
            format_value(result.value.value)
        );
    }
}

fn print_range(response: &RangeQueryResponse) {
    for result in &response.data.result {
        println!("# {}", format_metric(&result.metric));
        for point in &result.values {
            println!("{}\t{}", point.timestamp, format_value(point.value));
        }
    }
}

fn print_series(response: &SeriesQueryResponse) {
    for metric in &response.data {
        println!("{}", format_metric(metric));
    }
}

fn print_targets(targets: &[TargetStatus]) {
    for target in targets {
        let state = if target.up { "up" } else { "down" };
        let timestamp = target
            .last_scrape_time
            .map_or_else(|| "-".to_owned(), |value| value.to_string());
        let error = target.last_error.as_deref().unwrap_or("");
        println!(
            "{}\t{}\t{}\t{}\t{}",
            target.target, state, timestamp, target.url, error
        );
    }
}

fn format_metric(metric: &BTreeMap<String, String>) -> String {
    let name = metric.get("__name__").cloned().unwrap_or_default();
    let labels: Vec<_> = metric
        .iter()
        .filter(|(key, _)| key.as_str() != "__name__")
        .map(|(key, value)| format!("{key}=\"{value}\""))
        .collect();
    if labels.is_empty() {
        name
    } else {
        format!("{name}{{{}}}", labels.join(","))
    }
}

fn format_value(value: f64) -> String {
    if value.is_nan() {
        "NaN".to_owned()
    } else if value == f64::INFINITY {
        "+Inf".to_owned()
    } else if value == f64::NEG_INFINITY {
        "-Inf".to_owned()
    } else {
        format!("{value}")
    }
}

fn now_millis() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(i64::MAX as u128) as i64
}

fn parse_width(value: &str) -> Result<usize, String> {
    let width = value
        .parse::<usize>()
        .map_err(|_| format!("width must be between {MIN_CHART_WIDTH} and {MAX_CHART_WIDTH}"))?;
    if !(MIN_CHART_WIDTH..=MAX_CHART_WIDTH).contains(&width) {
        return Err(format!(
            "width must be between {MIN_CHART_WIDTH} and {MAX_CHART_WIDTH}"
        ));
    }
    Ok(width)
}

fn parse_duration(value: &str) -> Result<i64, String> {
    let value = value.trim();
    if value.len() < 2 {
        return Err(format!(
            "invalid duration '{value}'; use a number followed by s, m, h, or d"
        ));
    }
    let (number, suffix) = value.split_at(value.len() - 1);
    let amount: i64 = number.parse().map_err(|_| {
        format!("invalid duration '{value}'; use a whole number followed by s, m, h, or d")
    })?;
    if amount <= 0 {
        return Err(format!("duration '{value}' must be positive"));
    }
    let multiplier = match suffix {
        "s" => 1_000_i64,
        "m" => 60_000,
        "h" => 3_600_000,
        "d" => 86_400_000,
        _ => {
            return Err(format!(
                "invalid duration suffix in '{value}'; use s, m, h, or d"
            ));
        }
    };
    amount
        .checked_mul(multiplier)
        .ok_or_else(|| format!("duration '{value}' is too large"))
}

fn parse_instant(value: Option<&str>, now: i64) -> Result<i64, String> {
    let Some(value) = value else { return Ok(now) };
    if let Ok(timestamp) = value.parse::<i64>() {
        return Ok(timestamp);
    }
    let duration = parse_duration(value)?;
    now.checked_sub(duration)
        .ok_or_else(|| format!("time '{value}' is too far in the past"))
}

fn parse_range(args: &RangeArgs, now: i64) -> Result<(i64, i64), String> {
    if let Some(last) = &args.last {
        let duration = parse_duration(last)?;
        let start = now
            .checked_sub(duration)
            .ok_or_else(|| format!("duration '{last}' is too large"))?;
        return Ok((start, now));
    }
    let start = parse_instant(args.start.as_deref(), now)?;
    let end = parse_instant(args.end.as_deref(), now)?;
    Ok((start, end))
}

fn parse_graph_range(args: &GraphArgs, now: i64) -> Result<(i64, i64), String> {
    let last = args.last.as_deref().unwrap_or("1h");
    if args.start.is_none() && args.end.is_none() {
        let duration = parse_duration(last)?;
        let start = now
            .checked_sub(duration)
            .ok_or_else(|| format!("duration '{last}' is too large"))?;
        return Ok((start, now));
    }
    let start = parse_instant(args.start.as_deref(), now)?;
    let end = parse_instant(args.end.as_deref(), now)?;
    Ok((start, end))
}

fn detect_terminal_width() -> usize {
    terminal_size::terminal_size()
        .map(|(terminal_size::Width(width), _)| width as usize)
        .unwrap_or(DEFAULT_CHART_WIDTH)
        .clamp(MIN_CHART_WIDTH, MAX_CHART_WIDTH)
}

fn render_chart(
    response: &RangeQueryResponse,
    requested_start: i64,
    requested_end: i64,
    width: usize,
) -> String {
    let mut output = String::new();
    if response.data.result.is_empty() {
        return "(no data)\n".to_owned();
    }
    let plot_width = width.saturating_sub(CHART_PREFIX_WIDTH).max(1);
    for (index, series) in response.data.result.iter().enumerate() {
        let marker = chart_marker(index);
        let prefix = format!("{:>2} {marker} ", index + 1);
        let label = truncate_label(
            &format_metric(&series.metric),
            width.saturating_sub(UnicodeWidthStr::width(prefix.as_str())),
        );
        let _ = writeln!(output, "{prefix}{label}");
    }

    let points: Vec<&QueryPoint> = response
        .data
        .result
        .iter()
        .flat_map(|series| series.values.iter())
        .collect();
    let finite: Vec<f64> = points
        .iter()
        .map(|point| point.value)
        .filter(|value| value.is_finite())
        .collect();
    let (mut min, mut max): (f64, f64) = finite
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), value| {
            (min.min(*value), max.max(*value))
        });
    if finite.is_empty() {
        min = 0.0;
        max = 1.0;
    } else if (max - min).abs() < f64::EPSILON {
        let padding = if min.abs() < 1.0 {
            1.0
        } else {
            min.abs() * 0.1
        };
        min -= padding;
        max += padding;
    }

    // The requested query window, rather than the first/last present sample,
    // is the domain. This preserves leading and trailing staleness gaps.
    let start = requested_start;
    let end = requested_end.max(start.saturating_add(1));
    for row in 0..CHART_HEIGHT {
        let value = max - (max - min) * row as f64 / (CHART_HEIGHT - 1) as f64;
        let mut line = format!("{value:>9.2} | ");
        let mut cells = vec![' '; plot_width];
        for (series_index, series) in response.data.result.iter().enumerate() {
            for point in &series.values {
                if !point.value.is_finite() {
                    continue;
                }
                let x = (((point.timestamp - start) as f64 / (end - start) as f64)
                    * (plot_width - 1) as f64)
                    .round() as usize;
                let y = (((max - point.value) / (max - min)) * (CHART_HEIGHT - 1) as f64).round()
                    as usize;
                if y == row && x < cells.len() {
                    cells[x] = chart_marker(series_index);
                }
            }
        }
        line.extend(cells);
        let _ = writeln!(output, "{}", line.trim_end_matches(' '));
    }
    let start_label = format_axis_timestamp(start, plot_width);
    let end_label = format_axis_timestamp(end, plot_width);
    let mut axis = " ".repeat(CHART_PREFIX_WIDTH);
    axis.push_str(&start_label);
    axis.push_str(&" ".repeat(plot_width.saturating_sub(start_label.len() + end_label.len())));
    axis.push_str(&end_label);
    let _ = writeln!(output, "{axis}");
    output
}

/// Truncate a legend label at grapheme boundaries using terminal display
/// cells, not UTF-8 bytes or Unicode scalar values. `UnicodeWidthStr` is a
/// portable approximation; terminals can disagree about emoji presentation.
fn truncate_label(label: &str, max_cells: usize) -> String {
    if UnicodeWidthStr::width(label) <= max_cells {
        return label.to_owned();
    }
    let ellipsis = "…";
    let ellipsis_width = UnicodeWidthStr::width(ellipsis);
    if max_cells < ellipsis_width {
        return String::new();
    }
    let mut truncated = String::new();
    for grapheme in label.graphemes(true) {
        let mut candidate = truncated.clone();
        candidate.push_str(grapheme);
        if UnicodeWidthStr::width(candidate.as_str()) + ellipsis_width > max_cells {
            break;
        }
        truncated = candidate;
    }
    truncated.push_str(ellipsis);
    truncated
}

fn chart_marker(index: usize) -> char {
    const MARKERS: &[char] = &[
        '█', '▓', '▒', '░', '●', '◆', '▲', '■', '+', 'x', 'o', '1', '2', '3', '4', '5', '6', '7',
        '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
        'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    ];
    MARKERS[index % MARKERS.len()]
}

fn format_timestamp(timestamp: i64) -> String {
    let seconds = timestamp.div_euclid(1_000);
    let day_seconds = seconds.rem_euclid(86_400);
    format!(
        "{:02}:{:02}:{:02}",
        day_seconds / 3_600,
        day_seconds / 60 % 60,
        day_seconds % 60
    )
}

fn format_axis_timestamp(timestamp: i64, plot_width: usize) -> String {
    let seconds = timestamp.div_euclid(1_000).rem_euclid(86_400);
    if plot_width >= 17 {
        format_timestamp(timestamp)
    } else if plot_width >= 11 {
        format!("{:02}:{:02}", seconds / 60 % 60, seconds % 60)
    } else if plot_width >= 5 {
        format!("{:02}", seconds % 100)
    } else {
        String::new()
    }
}

fn format_server_error(status: u16, body: &[u8]) -> String {
    match serde_json::from_slice::<ApiErrorResponse>(body) {
        Ok(error) => match error.position {
            Some(position) => format!(
                "server returned HTTP {status}: {} (position {position})",
                error.error
            ),
            None => format!("server returned HTTP {status}: {}", error.error),
        },
        Err(_) => format!(
            "server returned HTTP {status}: {}",
            String::from_utf8_lossy(body).trim()
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gauge_query::{RangeQueryData, RangeResult};

    fn fixture(series: Vec<(&str, Vec<(i64, f64)>)>) -> RangeQueryResponse {
        RangeQueryResponse {
            status: "success".to_owned(),
            data: RangeQueryData {
                result_type: "matrix".to_owned(),
                result: series
                    .into_iter()
                    .map(|(name, values)| RangeResult {
                        metric: BTreeMap::from([("__name__".to_owned(), name.to_owned())]),
                        values: values
                            .into_iter()
                            .map(|(timestamp, value)| QueryPoint { timestamp, value })
                            .collect(),
                    })
                    .collect(),
            },
        }
    }

    #[test]
    fn parses_documented_durations() {
        assert_eq!(parse_duration("5m").unwrap(), 300_000);
        assert_eq!(parse_duration("1h").unwrap(), 3_600_000);
        assert_eq!(parse_duration("2d").unwrap(), 172_800_000);
        assert!(parse_duration("0s").is_err());
        assert!(parse_duration("1w").is_err());
        assert_eq!(parse_width("20").unwrap(), 20);
        assert_eq!(parse_width("1000").unwrap(), 1000);
        assert!(parse_width("19").is_err());
        assert!(parse_width("1001").is_err());
        assert!(parse_width("9223372036854775808").is_err());
    }

    #[test]
    fn chart_keeps_non_finite_values_as_gaps() {
        let response = RangeQueryResponse {
            status: "success".to_owned(),
            data: RangeQueryData {
                result_type: "matrix".to_owned(),
                result: vec![RangeResult {
                    metric: BTreeMap::from([(String::from("__name__"), String::from("cpu"))]),
                    values: vec![
                        QueryPoint {
                            timestamp: 0,
                            value: 1.0,
                        },
                        QueryPoint {
                            timestamp: 1_000,
                            value: f64::NAN,
                        },
                        QueryPoint {
                            timestamp: 2_000,
                            value: 2.0,
                        },
                    ],
                }],
            },
        };
        let chart = render_chart(&response, 0, 2_000, 32);
        assert!(chart.contains("cpu"));
        assert!(!chart.contains("NaN"));
        assert_eq!(chart, render_chart(&response, 0, 2_000, 32));
    }

    #[test]
    fn typed_query_points_accept_special_wire_values() {
        let point: QueryPoint = serde_json::from_str(r#"{"timestamp":1,"value":"+Inf"}"#).unwrap();
        assert_eq!(point.value, f64::INFINITY);
    }

    #[test]
    fn narrow_legend_is_width_bounded_and_truncated() {
        let response = RangeQueryResponse {
            status: "success".to_owned(),
            data: RangeQueryData {
                result_type: "matrix".to_owned(),
                result: vec![RangeResult {
                    metric: BTreeMap::from([
                        ("__name__".to_owned(), "node_cpu_percent".to_owned()),
                        ("cpu".to_owned(), "very-long-core-name".to_owned()),
                        ("instance".to_owned(), "host-with-a-long-name".to_owned()),
                    ]),
                    values: vec![QueryPoint {
                        timestamp: 500,
                        value: 7.0,
                    }],
                }],
            },
        };
        let chart = render_chart(&response, 0, 1_000, 20);
        assert!(chart.lines().all(|line| UnicodeWidthStr::width(line) <= 20));
        assert!(chart.lines().next().is_some_and(|line| line.ends_with('…')));
        assert_eq!(
            chart,
            r#" 1 █ node_cpu_perce…
     7.70 |
     7.61 |
     7.51 |
     7.42 |
     7.33 |
     7.23 |
     7.14 |
     7.05 |
     6.95 |     █
     6.86 |
     6.77 |
     6.67 |
     6.58 |
     6.49 |
     6.39 |
     6.30 |
            00    01
"#
        );
    }

    #[test]
    fn wide_unicode_legend_golden_uses_display_cells() {
        let response = RangeQueryResponse {
            status: "success".to_owned(),
            data: RangeQueryData {
                result_type: "matrix".to_owned(),
                result: vec![RangeResult {
                    metric: BTreeMap::from([
                        ("__name__".to_owned(), "温度メトリック".to_owned()),
                        ("地域".to_owned(), "東京".to_owned()),
                        ("instance".to_owned(), "ホスト長い名前".to_owned()),
                    ]),
                    values: vec![QueryPoint {
                        timestamp: 500,
                        value: 7.0,
                    }],
                }],
            },
        };
        let chart = render_chart(&response, 0, 1_000, 20);
        assert!(chart.lines().all(|line| UnicodeWidthStr::width(line) <= 20));
        assert_eq!(
            chart,
            r#" 1 █ 温度メトリック…
     7.70 |
     7.61 |
     7.51 |
     7.42 |
     7.33 |
     7.23 |
     7.14 |
     7.05 |
     6.95 |     █
     6.86 |
     6.77 |
     6.67 |
     6.58 |
     6.49 |
     6.39 |
     6.30 |
            00    01
"#
        );
    }

    #[test]
    fn emoji_sequence_legend_golden_uses_grapheme_boundaries() {
        let response = RangeQueryResponse {
            status: "success".to_owned(),
            data: RangeQueryData {
                result_type: "matrix".to_owned(),
                result: vec![RangeResult {
                    metric: BTreeMap::from([
                        ("__name__".to_owned(), "x".to_owned()),
                        ("family".to_owned(), "👩‍👩‍👧‍👦".to_owned()),
                        ("instance".to_owned(), "host-long-name".to_owned()),
                    ]),
                    values: vec![QueryPoint {
                        timestamp: 500,
                        value: 7.0,
                    }],
                }],
            },
        };
        let chart = render_chart(&response, 0, 1_000, 20);
        assert!(chart.lines().all(|line| UnicodeWidthStr::width(line) <= 20));
        assert_eq!(
            chart,
            r#" 1 █ x{family="👩‍👩‍👧‍👦",…
     7.70 |
     7.61 |
     7.51 |
     7.42 |
     7.33 |
     7.23 |
     7.14 |
     7.05 |
     6.95 |     █
     6.86 |
     6.77 |
     6.67 |
     6.58 |
     6.49 |
     6.39 |
     6.30 |
            00    01
"#
        );
    }

    fn assert_golden(response: RangeQueryResponse, start: i64, end: i64, expected: &str) {
        assert_eq!(render_chart(&response, start, end, 28), expected);
    }

    #[test]
    fn chart_goldens_cover_gaps_series_specials_and_empty_results() {
        let fixtures = [
            (
                "interior",
                fixture(vec![(
                    "cpu",
                    vec![(0, 1.0), (5_000, f64::NAN), (10_000, 3.0)],
                )]),
                0,
                10_000,
            ),
            (
                "leading_trailing",
                fixture(vec![("cpu", vec![(4_000, 2.0), (6_000, 2.0)])]),
                0,
                10_000,
            ),
            (
                "multi",
                fixture(vec![
                    ("cpu0", vec![(0, 0.0), (10_000, 1.0)]),
                    ("cpu1", vec![(0, 1.0), (10_000, 2.0)]),
                    ("cpu2", vec![(0, 2.0), (10_000, 3.0)]),
                    ("cpu3", vec![(0, 3.0), (10_000, 4.0)]),
                    ("cpu4", vec![(0, 4.0), (10_000, 5.0)]),
                ]),
                0,
                10_000,
            ),
            (
                "special",
                fixture(vec![(
                    "cpu",
                    vec![
                        (0, f64::INFINITY),
                        (5_000, 1.0),
                        (10_000, f64::NEG_INFINITY),
                    ],
                )]),
                0,
                10_000,
            ),
            ("empty", fixture(vec![]), 0, 10_000),
        ];
        assert_golden(
            fixtures[0].1.clone(),
            fixtures[0].2,
            fixtures[0].3,
            r#" 1 █ cpu
     3.00 |                █
     2.87 |
     2.73 |
     2.60 |
     2.47 |
     2.33 |
     2.20 |
     2.07 |
     1.93 |
     1.80 |
     1.67 |
     1.53 |
     1.40 |
     1.27 |
     1.13 |
     1.00 | █
            00:00      00:10
"#,
        );
        assert_golden(
            fixtures[1].1.clone(),
            fixtures[1].2,
            fixtures[1].3,
            r#" 1 █ cpu
     2.20 |
     2.17 |
     2.15 |
     2.12 |
     2.09 |
     2.07 |
     2.04 |
     2.01 |
     1.99 |       █  █
     1.96 |
     1.93 |
     1.91 |
     1.88 |
     1.85 |
     1.83 |
     1.80 |
            00:00      00:10
"#,
        );
        assert_golden(
            fixtures[2].1.clone(),
            fixtures[2].2,
            fixtures[2].3,
            r#" 1 █ cpu0
 2 ▓ cpu1
 3 ▒ cpu2
 4 ░ cpu3
 5 ● cpu4
     5.00 |                ●
     4.67 |
     4.33 |
     4.00 | ●              ░
     3.67 |
     3.33 |
     3.00 | ░              ▒
     2.67 |
     2.33 |
     2.00 | ▒              ▓
     1.67 |
     1.33 |
     1.00 | ▓              █
     0.67 |
     0.33 |
     0.00 | █
            00:00      00:10
"#,
        );
        assert_golden(
            fixtures[3].1.clone(),
            fixtures[3].2,
            fixtures[3].3,
            r#" 1 █ cpu
     1.10 |
     1.09 |
     1.07 |
     1.06 |
     1.05 |
     1.03 |
     1.02 |
     1.01 |
     0.99 |         █
     0.98 |
     0.97 |
     0.95 |
     0.94 |
     0.93 |
     0.91 |
     0.90 |
            00:00      00:10
"#,
        );
        assert_golden(fixtures[4].1.clone(), 0, 10_000, "(no data)\n");
    }
}

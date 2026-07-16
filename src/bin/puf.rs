use std::collections::BTreeMap;
use std::env;
use std::fmt::{self, Display, Formatter};
use std::fs::File;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

const BATCH_SIZE: usize = 100;
const DEFAULT_SERVER_URL: &str = "http://127.0.0.1:8666";

#[derive(Debug, Deserialize, Serialize)]
struct ApiDoc {
    id: String,
    vector: Option<Vec<f32>>,
    #[serde(default)]
    attributes: BTreeMap<String, Value>,
}

#[derive(Debug, Deserialize, Serialize)]
struct SchemaHint {
    #[serde(default)]
    full_text_search: bool,
}

#[derive(Debug, Deserialize, Serialize)]
struct UpsertRequest {
    #[serde(default)]
    upserts: Vec<ApiDoc>,
    #[serde(default)]
    deletes: Vec<String>,
    #[serde(default)]
    schema: BTreeMap<String, SchemaHint>,
}

#[derive(Debug, Deserialize, Serialize)]
struct UpsertResponse {
    upserted: usize,
    deleted: usize,
}

#[derive(Debug, Deserialize, Serialize)]
struct NamespaceListResponse {
    namespaces: Vec<String>,
}

#[derive(Debug, Deserialize, Serialize)]
struct QueryRequest {
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

#[derive(Debug, Deserialize, Serialize)]
struct QueryResponse {
    results: Vec<ApiResult>,
}

#[derive(Debug, Deserialize, Serialize)]
struct ApiResult {
    id: String,
    score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    attributes: Option<BTreeMap<String, Value>>,
}

#[derive(Debug, Parser)]
#[command(name = "puf", about = "CLI client for pufferclone")]
struct Cli {
    /// Pufferclone HTTP URL.
    #[arg(long, global = true)]
    url: Option<String>,
    /// Emit machine-readable JSON.
    #[arg(long, global = true)]
    json: bool,
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Manage namespaces.
    Ns {
        #[command(subcommand)]
        command: NamespaceCommand,
    },
    /// Upsert documents from JSONL.
    Upsert(UpsertArgs),
    /// Query a namespace.
    Query(QueryArgs),
}

#[derive(Debug, Subcommand)]
enum NamespaceCommand {
    /// List namespaces.
    Ls,
    /// Delete a namespace.
    Rm {
        namespace: String,
        /// Skip the interactive confirmation prompt.
        #[arg(long)]
        yes: bool,
    },
}

#[derive(Debug, Args)]
struct UpsertArgs {
    namespace: String,
    /// JSONL input file, or `-` for stdin.
    #[arg(short = 'f', long = "file", default_value = "-")]
    file: PathBuf,
}

#[derive(Debug, Args)]
struct QueryArgs {
    namespace: String,
    /// Full-text query.
    #[arg(long)]
    text: Option<String>,
    /// File containing a comma-separated or JSON vector.
    #[arg(long = "vector-file", conflicts_with = "vector")]
    vector_file: Option<PathBuf>,
    /// Comma-separated query vector.
    #[arg(long, conflicts_with = "vector_file")]
    vector: Option<String>,
    /// Number of results to return.
    #[arg(long, default_value_t = 10)]
    top_k: usize,
    /// HNSW search breadth.
    #[arg(long)]
    ef_search: Option<usize>,
    /// Equality filter, in `field=value` form.
    #[arg(long = "filter")]
    filters: Vec<String>,
    /// Membership filter, in `field=value1,value2` form.
    #[arg(long = "filter-in")]
    filter_in: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct InputDoc {
    id: String,
    #[serde(default)]
    vector: Option<Vec<f32>>,
    #[serde(default)]
    text: Option<String>,
    #[serde(default, alias = "attributes")]
    attrs: BTreeMap<String, Value>,
}

#[derive(Debug, Serialize)]
struct UpsertSummary {
    total_lines: usize,
    upserted: usize,
    deleted: usize,
    batches: usize,
    sent: usize,
    ok: usize,
    failed: usize,
    failed_lines: Vec<usize>,
    unknown: usize,
    unknown_lines: Vec<usize>,
    not_attempted: usize,
    not_attempted_lines: Vec<usize>,
    input_truncated_by_abort: bool,
}

struct UpsertOutcome {
    summary: UpsertSummary,
    failures: Vec<(usize, String)>,
    operation_error: Option<RequestError>,
    accounting_error: Option<String>,
}

#[derive(Debug)]
struct CliError(String);

#[derive(Debug)]
enum RequestError {
    Confirmed(String),
    Unknown(String),
    Local(String),
}

impl CliError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl Display for CliError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

impl std::error::Error for CliError {}

impl Display for RequestError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Confirmed(message) | Self::Unknown(message) | Self::Local(message) => {
                message.fmt(formatter)
            }
        }
    }
}

impl From<RequestError> for CliError {
    fn from(error: RequestError) -> Self {
        Self(error.to_string())
    }
}

type Result<T> = std::result::Result<T, CliError>;

struct Client {
    base_url: String,
    agent: ureq::Agent,
}

impl Client {
    fn new(base_url: String) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_owned(),
            agent: ureq::AgentBuilder::new().build(),
        }
    }

    fn request(
        &self,
        method: &str,
        path: &str,
        body: Option<&str>,
    ) -> std::result::Result<String, RequestError> {
        let url = format!("{}{}", self.base_url, path);
        let result = match (method, body) {
            ("GET", None) => self.agent.get(&url).call(),
            ("DELETE", None) => self.agent.delete(&url).call(),
            ("POST", Some(body)) => self
                .agent
                .post(&url)
                .set("content-type", "application/json")
                .send_string(body),
            _ => return Err(RequestError::Local("unsupported HTTP request".to_owned())),
        };

        match result {
            Ok(response) => response.into_string().map_err(|error| {
                RequestError::Unknown(format!("reading response from {url}: {error}"))
            }),
            Err(ureq::Error::Status(status, response)) => {
                let body = response.into_string().unwrap_or_default();
                let detail = if body.trim().is_empty() {
                    "no response body".to_owned()
                } else {
                    body
                };
                Err(RequestError::Confirmed(format!(
                    "server returned HTTP {status} for {url}: {detail}"
                )))
            }
            Err(error) => Err(RequestError::Unknown(format!(
                "request to {url} failed: {error}"
            ))),
        }
    }

    fn get<T: for<'de> Deserialize<'de>>(&self, path: &str) -> Result<T> {
        decode_response(self.request("GET", path, None)).map_err(CliError::from)
    }

    fn post<T: for<'de> Deserialize<'de>>(
        &self,
        path: &str,
        request: &impl Serialize,
    ) -> Result<T> {
        let body = serde_json::to_string(request)
            .map_err(|error| CliError::new(format!("encoding request: {error}")))?;
        decode_response(self.request("POST", path, Some(&body))).map_err(CliError::from)
    }

    fn post_with_failure<T: for<'de> Deserialize<'de>>(
        &self,
        path: &str,
        request: &impl Serialize,
    ) -> std::result::Result<T, RequestError> {
        let body = serde_json::to_string(request)
            .map_err(|error| RequestError::Local(format!("encoding request: {error}")))?;
        decode_response(self.request("POST", path, Some(&body)))
    }

    fn delete(&self, path: &str) -> Result<()> {
        self.request("DELETE", path, None)
            .map(|_| ())
            .map_err(CliError::from)
    }
}

fn decode_response<T: for<'de> Deserialize<'de>>(
    body: std::result::Result<String, RequestError>,
) -> std::result::Result<T, RequestError> {
    let body = body?;
    serde_json::from_str(&body)
        .map_err(|error| RequestError::Unknown(format!("invalid JSON response: {error}: {body}")))
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let url = cli
        .url
        .or_else(|| env::var("PUFFERCLONE_URL").ok())
        .unwrap_or_else(|| DEFAULT_SERVER_URL.to_owned());
    let client = Client::new(url);

    match cli.command {
        Command::Ns { command } => run_namespace(&client, command, cli.json),
        Command::Upsert(args) => run_upsert(&client, args, cli.json),
        Command::Query(args) => run_query(&client, args, cli.json),
    }
}

fn run_namespace(client: &Client, command: NamespaceCommand, json_output: bool) -> Result<()> {
    match command {
        NamespaceCommand::Ls => {
            let response: NamespaceListResponse = client.get("/v1/namespaces")?;
            if json_output {
                print_json(&response)?;
            } else {
                println!("NAMESPACE");
                println!("---------");
                for namespace in response.namespaces {
                    println!("{namespace}");
                }
            }
        }
        NamespaceCommand::Rm { namespace, yes } => {
            if !yes && !confirm_delete(&namespace)? {
                return Err(CliError::new("aborted"));
            }
            client.delete(&namespace_path(&namespace))?;
            if json_output {
                print_json(&json!({"namespace": namespace, "deleted": true}))?;
            } else {
                println!("Deleted namespace '{namespace}'.");
            }
        }
    }
    Ok(())
}

fn run_upsert(client: &Client, args: UpsertArgs, json_output: bool) -> Result<()> {
    let stdin_input = args.file.as_os_str() == "-";
    let reader: Box<dyn BufRead> = if stdin_input {
        Box::new(BufReader::new(io::stdin()))
    } else {
        let file = File::open(&args.file)
            .map_err(|error| CliError::new(format!("opening {}: {error}", args.file.display())))?;
        Box::new(BufReader::new(file))
    };

    let UpsertOutcome {
        summary,
        failures,
        operation_error,
        accounting_error,
    } = process_upsert_lines(reader.lines(), stdin_input, |batch| {
        send_batch(client, &args.namespace, batch)
    });
    if json_output {
        print_json(&summary)?;
    } else {
        let failed_lines = summary
            .failed_lines
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        let unknown_lines = summary
            .unknown_lines
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        let not_attempted_lines = summary
            .not_attempted_lines
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        println!(
            "ok={} failed={} unknown={} not_attempted={} batches={}",
            summary.ok, summary.failed, summary.unknown, summary.not_attempted, summary.batches
        );
        if !failed_lines.is_empty() {
            println!("failed lines: {failed_lines}");
        }
        if !unknown_lines.is_empty() {
            println!("unknown lines: {unknown_lines}");
        }
        if !not_attempted_lines.is_empty() {
            println!("not attempted lines: {not_attempted_lines}");
        }
        if summary.input_truncated_by_abort {
            println!("input truncated by abort: remaining lines unread");
        }
    }
    let mut errors = Vec::new();
    if let Some(error) = accounting_error {
        errors.push(error);
    }
    if !failures.is_empty() {
        let details = failures
            .iter()
            .map(|(line, error)| format!("line {line}: {error}"))
            .collect::<Vec<_>>()
            .join("; ");
        errors.push(format!("invalid JSONL input: {details}"));
    }
    if let Some(error) = operation_error {
        errors.push(error.to_string());
    }
    if !errors.is_empty() {
        return Err(CliError::new(errors.join("; ")));
    }
    Ok(())
}

fn process_upsert_lines<I, F>(input: I, stdin_input: bool, mut send: F) -> UpsertOutcome
where
    I: Iterator<Item = std::io::Result<String>>,
    F: FnMut(&mut Vec<(usize, ApiDoc)>) -> std::result::Result<UpsertResponse, RequestError>,
{
    let mut batch = Vec::with_capacity(BATCH_SIZE);
    let mut upserted = 0;
    let mut deleted = 0;
    let mut batches = 0;
    let mut failures = Vec::new();
    let mut failed_lines = Vec::new();
    let mut unknown_lines = Vec::new();
    let mut not_attempted_lines = Vec::new();
    let mut input_truncated_by_abort = false;
    let mut operation_error: Option<RequestError> = None;
    let mut lines = input;
    let mut line_number = 0;
    let mut aborted = false;
    while let Some(line_result) = lines.next() {
        line_number += 1;
        let line = match line_result {
            Ok(line) => line,
            Err(error) => {
                operation_error = Some(RequestError::Local(format!(
                    "reading JSONL line {line_number}: {error}"
                )));
                failures.push((line_number, format!("read error: {error}")));
                not_attempted_lines.extend(batch.iter().map(|(line, _)| *line));
                batch.clear();
                aborted = true;
                if stdin_input {
                    input_truncated_by_abort = true;
                } else {
                    drain_remaining(&mut lines, &mut line_number, &mut not_attempted_lines);
                }
                break;
            }
        };
        if line.trim().is_empty() {
            failures.push((line_number, "blank JSONL line".to_owned()));
            continue;
        }
        let input: InputDoc = match serde_json::from_str(&line) {
            Ok(input) => input,
            Err(error) => {
                failures.push((line_number, format!("invalid JSONL document: {error}")));
                continue;
            }
        };
        let doc = match input_doc(input) {
            Ok(doc) => doc,
            Err(error) => {
                failures.push((line_number, error.to_string()));
                continue;
            }
        };
        batch.push((line_number, doc));
        if batch.len() == BATCH_SIZE {
            let line_numbers = batch.iter().map(|(line, _)| *line).collect::<Vec<_>>();
            match send(&mut batch) {
                Ok(summary) => {
                    upserted += summary.upserted;
                    deleted += summary.deleted;
                    batches += 1;
                }
                Err(error) => {
                    match &error {
                        RequestError::Confirmed(_) => failed_lines.extend(line_numbers),
                        RequestError::Unknown(_) => unknown_lines.extend(line_numbers),
                        RequestError::Local(_) => {}
                    }
                    operation_error = Some(error);
                    aborted = true;
                    if stdin_input {
                        input_truncated_by_abort = true;
                    } else {
                        drain_remaining(&mut lines, &mut line_number, &mut not_attempted_lines);
                    }
                    break;
                }
            }
        }
    }
    if !aborted && !batch.is_empty() {
        let line_numbers = batch.iter().map(|(line, _)| *line).collect::<Vec<_>>();
        match send(&mut batch) {
            Ok(summary) => {
                upserted += summary.upserted;
                deleted += summary.deleted;
                batches += 1;
            }
            Err(error) => {
                match &error {
                    RequestError::Confirmed(_) => failed_lines.extend(line_numbers),
                    RequestError::Unknown(_) => unknown_lines.extend(line_numbers),
                    RequestError::Local(_) => {}
                }
                operation_error = Some(error);
            }
        }
    }

    failed_lines.extend(failures.iter().map(|(line, _)| *line));
    failed_lines.sort_unstable();
    failed_lines.dedup();
    unknown_lines.sort_unstable();
    unknown_lines.dedup();
    not_attempted_lines.sort_unstable();
    not_attempted_lines.dedup();

    let summary = UpsertSummary {
        total_lines: line_number,
        upserted,
        deleted,
        batches,
        sent: upserted,
        ok: upserted,
        failed: failed_lines.len(),
        failed_lines,
        unknown: unknown_lines.len(),
        unknown_lines,
        not_attempted: not_attempted_lines.len(),
        not_attempted_lines,
        input_truncated_by_abort,
    };
    let accounted = summary.ok + summary.failed + summary.unknown + summary.not_attempted;
    let accounting_error = (accounted != summary.total_lines).then(|| {
        format!(
            "upsert accounting invariant failed: {accounted} buckets for {} input lines",
            summary.total_lines
        )
    });
    UpsertOutcome {
        summary,
        failures,
        operation_error,
        accounting_error,
    }
}

fn drain_remaining<I: Iterator<Item = std::io::Result<String>>>(
    lines: &mut I,
    line_number: &mut usize,
    not_attempted_lines: &mut Vec<usize>,
) {
    for _line in lines {
        *line_number += 1;
        not_attempted_lines.push(*line_number);
    }
}

fn input_doc(input: InputDoc) -> Result<ApiDoc> {
    let mut attributes = input.attrs;
    if let Some(text) = input.text {
        attributes.insert("text".to_owned(), Value::String(text));
    }
    let vector = input
        .vector
        .map(|vector| validate_vector(vector, "upsert vector"))
        .transpose()?;
    Ok(ApiDoc {
        id: input.id,
        vector,
        attributes,
    })
}

fn send_batch(
    client: &Client,
    namespace: &str,
    batch: &mut Vec<(usize, ApiDoc)>,
) -> std::result::Result<UpsertResponse, RequestError> {
    let schema = if batch
        .iter()
        .any(|(_, doc)| matches!(doc.attributes.get("text"), Some(Value::String(_))))
    {
        BTreeMap::from([(
            "text".to_owned(),
            SchemaHint {
                full_text_search: true,
            },
        )])
    } else {
        BTreeMap::new()
    };
    let request = UpsertRequest {
        upserts: std::mem::take(batch)
            .into_iter()
            .map(|(_, doc)| doc)
            .collect(),
        deletes: Vec::new(),
        schema,
    };
    client.post_with_failure(&namespace_path(namespace), &request)
}

fn run_query(client: &Client, args: QueryArgs, json_output: bool) -> Result<()> {
    let vector = match (args.vector, args.vector_file) {
        (Some(vector), None) => Some(parse_vector(&vector, "--vector")?),
        (None, Some(path)) => {
            let mut input = String::new();
            File::open(&path)
                .and_then(|mut file| file.read_to_string(&mut input))
                .map_err(|error| CliError::new(format!("reading {}: {error}", path.display())))?;
            Some(parse_vector(&input, "--vector-file")?)
        }
        (None, None) => None,
        (Some(_), Some(_)) => unreachable!("clap prevents conflicting vector options"),
    };
    if args.text.is_none() && vector.is_none() {
        return Err(CliError::new(
            "query requires --text, --vector, or --vector-file",
        ));
    }

    let filters = build_filters(&args.filters, &args.filter_in)?;
    let request = QueryRequest {
        vector,
        text: args.text,
        filters,
        top_k: Some(args.top_k),
        include_attributes: true,
        ef_search: args.ef_search,
    };
    let response: QueryResponse = client.post(
        &format!("{}/query", namespace_path(&args.namespace)),
        &request,
    )?;
    if json_output {
        print_json(&response)?;
    } else {
        println!("ID\tSCORE\tATTRIBUTES");
        for result in response.results {
            let attributes = result
                .attributes
                .map(|attributes| serde_json::to_string(&attributes).unwrap_or_default())
                .unwrap_or_default();
            println!("{}\t{:.6}\t{}", result.id, result.score, attributes);
        }
    }
    Ok(())
}

fn parse_vector(value: &str, option: &str) -> Result<Vec<f32>> {
    let value = value.trim();
    if value.is_empty() {
        return Err(CliError::new(format!("{option} must not be empty")));
    }
    if value.starts_with('[') {
        let vector = serde_json::from_str(value)
            .map_err(|error| CliError::new(format!("invalid {option} JSON vector: {error}")))?;
        return validate_vector(vector, option);
    }
    let vector = value
        .split(',')
        .enumerate()
        .map(|(index, component)| {
            component.trim().parse::<f32>().map_err(|error| {
                CliError::new(format!("invalid {option} component {}: {error}", index + 1))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    validate_vector(vector, option)
}

fn validate_vector(vector: Vec<f32>, option: &str) -> Result<Vec<f32>> {
    if vector.is_empty() {
        return Err(CliError::new(format!("{option} must not be empty")));
    }
    if vector.iter().any(|value| !value.is_finite()) {
        return Err(CliError::new(format!(
            "{option} must contain only finite values"
        )));
    }
    Ok(vector)
}

fn build_filters(filters: &[String], filter_in: &[String]) -> Result<Option<Value>> {
    let mut values = Vec::with_capacity(filters.len() + filter_in.len());
    for filter in filters {
        let (field, value) = split_filter(filter, "--filter")?;
        values.push(json!({
            "op": "eq",
            "field": field,
            "value": parse_filter_value(value),
        }));
    }
    for filter in filter_in {
        let (field, values_text) = split_filter(filter, "--filter-in")?;
        if values_text.split(',').any(|value| value.trim().is_empty()) {
            return Err(CliError::new("--filter-in members must not be empty"));
        }
        let filter_values = values_text
            .split(',')
            .map(|value| parse_filter_value(value.trim()))
            .collect::<Vec<_>>();
        values.push(json!({
            "op": "in",
            "field": field,
            "values": filter_values,
        }));
    }
    Ok(match values.len() {
        0 => None,
        1 => Some(values.remove(0)),
        _ => Some(Value::Array(values)),
    })
}

fn split_filter<'a>(filter: &'a str, option: &str) -> Result<(&'a str, &'a str)> {
    let (field, value) = filter
        .split_once('=')
        .ok_or_else(|| CliError::new(format!("{option} must use field=value")))?;
    let field = field.trim();
    let value = value.trim();
    if field.is_empty() || value.is_empty() {
        return Err(CliError::new(format!(
            "{option} must use non-empty field and value"
        )));
    }
    Ok((field, value))
}

fn parse_filter_value(value: &str) -> Value {
    if value.eq_ignore_ascii_case("true") {
        Value::Bool(true)
    } else if value.eq_ignore_ascii_case("false") {
        Value::Bool(false)
    } else if let Ok(value) = value.parse::<i64>() {
        json!(value)
    } else if let Ok(value) = value.parse::<f64>() {
        json!(value)
    } else {
        Value::String(value.to_owned())
    }
}

fn namespace_path(namespace: &str) -> String {
    format!("/v1/namespaces/{}", percent_encode(namespace))
}

fn percent_encode(value: &str) -> String {
    value
        .bytes()
        .map(|byte| match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                (byte as char).to_string()
            }
            byte => format!("%{byte:02X}"),
        })
        .collect()
}

fn confirm_delete(namespace: &str) -> Result<bool> {
    eprint!("Delete namespace '{namespace}'? [y/N] ");
    io::stderr()
        .flush()
        .map_err(|error| CliError::new(format!("flushing confirmation prompt: {error}")))?;
    let mut answer = String::new();
    io::stdin()
        .read_line(&mut answer)
        .map_err(|error| CliError::new(format!("reading confirmation: {error}")))?;
    Ok(matches!(
        answer.trim().to_ascii_lowercase().as_str(),
        "y" | "yes"
    ))
}

fn print_json(value: &impl Serialize) -> Result<()> {
    println!(
        "{}",
        serde_json::to_string(value)
            .map_err(|error| CliError::new(format!("encoding JSON output: {error}")))?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::{process_upsert_lines, UpsertResponse};

    #[test]
    fn read_error_mid_file_drains_and_accounts_trailing_records() {
        let mut input = (0..100)
            .map(|id| Ok(format!("{{\"id\":\"doc-{id}\",\"vector\":[1,0]}}")))
            .collect::<Vec<std::io::Result<String>>>();
        input.push(Err(std::io::Error::new(
            std::io::ErrorKind::Interrupted,
            "injected read error",
        )));
        input.push(Ok("trailing-1".to_owned()));
        input.push(Ok("trailing-2".to_owned()));
        let requests = Cell::new(0);

        let outcome = process_upsert_lines(input.into_iter(), false, |batch| {
            requests.set(requests.get() + 1);
            let upserted = batch.len();
            batch.clear();
            Ok(UpsertResponse {
                upserted,
                deleted: 0,
            })
        });

        assert_eq!(requests.get(), 1);
        assert_eq!(outcome.summary.total_lines, 103);
        assert_eq!(outcome.summary.ok, 100);
        assert_eq!(outcome.summary.failed, 1);
        assert_eq!(outcome.summary.failed_lines, vec![101]);
        assert_eq!(outcome.summary.unknown, 0);
        assert_eq!(outcome.summary.not_attempted, 2);
        assert_eq!(outcome.summary.not_attempted_lines, vec![102, 103]);
        assert!(!outcome.summary.input_truncated_by_abort);
        assert!(outcome.operation_error.is_some());
        let json = serde_json::to_value(&outcome.summary).expect("summary JSON");
        assert_eq!(json["failed_lines"], serde_json::json!([101]));
        assert_eq!(json["not_attempted_lines"], serde_json::json!([102, 103]));
    }
}

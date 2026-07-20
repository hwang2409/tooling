use std::time::Duration;

use gauge_exporters::proc::ProcCollector;
use gauge_exporters::{CachedExporter, parse_listen_options, serve};

fn main() {
    let (options, rest) = match parse_listen_options(std::env::args().skip(1), 9101) {
        Ok(value) => value,
        Err(error) => exit_with_usage(&error),
    };
    if rest.iter().any(|arg| arg == "--help" || arg == "-h") {
        print_usage();
        return;
    }

    let mut patterns = Vec::new();
    let mut args = rest.into_iter();
    while let Some(arg) = args.next() {
        if arg == "--pattern" {
            patterns.push(
                args.next()
                    .unwrap_or_else(|| exit_with_usage("--pattern requires a value")),
            );
        } else if let Some(pattern) = arg.strip_prefix("--pattern=") {
            patterns.push(pattern.to_owned());
        } else if arg == "--patterns" {
            let value = args
                .next()
                .unwrap_or_else(|| exit_with_usage("--patterns requires a comma-separated value"));
            patterns.extend(
                value
                    .split(',')
                    .filter(|pattern| !pattern.is_empty())
                    .map(str::to_owned),
            );
        } else if let Some(value) = arg.strip_prefix("--patterns=") {
            patterns.extend(
                value
                    .split(',')
                    .filter(|pattern| !pattern.is_empty())
                    .map(str::to_owned),
            );
        } else {
            exit_with_usage(&format!("unknown argument: {arg}"));
        }
    }
    if patterns.is_empty() {
        patterns.push("*".to_owned());
    }

    let listener = std::net::TcpListener::bind(options.address()).unwrap_or_else(|error| {
        exit_with_usage(&format!("could not bind {}: {error}", options.address()))
    });
    eprintln!(
        "gauge-proc listening on http://{}/metrics",
        listener.local_addr().unwrap()
    );
    let collector = ProcCollector::new(patterns);
    let cache = CachedExporter::new(move || collector.scrape(), Duration::from_secs(1));
    if let Err(error) = serve(listener, move || cache.scrape()) {
        exit_with_usage(&error.to_string());
    }
}

fn print_usage() {
    println!(
        "Usage: gauge-proc [--bind ADDRESS] [--port PORT] --pattern PATTERN [--pattern PATTERN ...]"
    );
    println!("Plain patterns match substrings; * and ? are supported as wildcards.");
}

fn exit_with_usage(error: &str) -> ! {
    eprintln!("error: {error}");
    print_usage();
    std::process::exit(2);
}

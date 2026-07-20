use std::time::Duration;

use gauge_exporters::node::NodeCollector;
use gauge_exporters::{CachedExporter, ListenOptions, parse_listen_options, serve};

fn main() {
    let (options, rest) = match parse_listen_options(std::env::args().skip(1), 9100) {
        Ok(value) => value,
        Err(error) => exit_with_usage(&error),
    };
    if rest.iter().any(|arg| arg == "--help" || arg == "-h") {
        print_usage();
        return;
    }
    if let Some(unknown) = rest.first() {
        exit_with_usage(&format!("unknown argument: {unknown}"));
    }

    let listener = bind(&options);
    eprintln!(
        "gauge-node listening on http://{}/metrics",
        listener.local_addr().unwrap()
    );
    let mut collector = NodeCollector::new();
    let cache = CachedExporter::new(move || collector.scrape(), Duration::from_secs(1));
    if let Err(error) = serve(listener, move || cache.scrape()) {
        exit_with_usage(&error.to_string());
    }
}

fn bind(options: &ListenOptions) -> std::net::TcpListener {
    std::net::TcpListener::bind(options.address()).unwrap_or_else(|error| {
        exit_with_usage(&format!("could not bind {}: {error}", options.address()))
    })
}

fn print_usage() {
    println!("Usage: gauge-node [--bind ADDRESS] [--port PORT]");
}

fn exit_with_usage(error: &str) -> ! {
    eprintln!("error: {error}");
    eprintln!("Usage: gauge-node [--bind ADDRESS] [--port PORT]");
    std::process::exit(2);
}

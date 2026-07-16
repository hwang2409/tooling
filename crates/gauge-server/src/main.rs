use std::path::PathBuf;

use gauge_server::config::Config;
use gauge_server::scrape::spawn_scrapers;
use gauge_server::server::{AppState, router};
use gauge_store::{GaugeStore, StoreConfig};

#[tokio::main]
async fn main() {
    let args = match Args::parse(std::env::args().skip(1)) {
        Ok(args) => args,
        Err(error) => exit_with_usage(&error),
    };
    if args.help {
        print_usage();
        return;
    }
    let config = match Config::load(&args.config) {
        Ok(config) => config,
        Err(error) => exit_with_usage(&error.to_string()),
    };
    let mut store_config = StoreConfig::default().with_retention_ms(
        config
            .storage
            .retention
            .as_duration()
            .as_millis()
            .min(i64::MAX as u128) as i64,
    );
    if let Some(duration) = &config.storage.partition_duration {
        store_config = store_config.with_partition_duration_ms(
            duration.as_duration().as_millis().min(i64::MAX as u128) as i64,
        );
    }
    if let Some(duration) = &config.storage.out_of_order_tolerance {
        store_config = store_config.with_out_of_order_tolerance_ms(
            duration.as_duration().as_millis().min(i64::MAX as u128) as i64,
        );
    }
    let store = match GaugeStore::open_with_config(&config.storage.data_dir, store_config) {
        Ok(store) => store,
        Err(error) => exit_with_usage(&format!("could not open store: {error}")),
    };
    let state = AppState::new(
        store.clone(),
        config
            .targets
            .iter()
            .map(|target| (target.name.clone(), target.url.clone())),
    );
    spawn_scrapers(&config, state.clone());
    spawn_flush_task(store.clone());
    let bind = format!("{}:{}", args.bind, args.port);
    let listener = match tokio::net::TcpListener::bind(&bind).await {
        Ok(listener) => listener,
        Err(error) => exit_with_usage(&format!("could not bind {bind}: {error}")),
    };
    eprintln!(
        "gauge-server listening on http://{}",
        listener.local_addr().unwrap()
    );
    if let Err(error) = axum::serve(listener, router(state)).await {
        exit_with_usage(&error.to_string());
    }
}

fn spawn_flush_task(store: GaugeStore) {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(30));
        loop {
            interval.tick().await;
            let store = store.clone();
            let _ = tokio::task::spawn_blocking(move || store.flush(now_millis())).await;
        }
    });
}

fn now_millis() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

struct Args {
    config: PathBuf,
    bind: String,
    port: u16,
    help: bool,
}

impl Args {
    fn parse(args: impl IntoIterator<Item = String>) -> Result<Self, String> {
        let mut config = None;
        let mut bind = "127.0.0.1".to_owned();
        let mut port = 8428;
        let mut help = false;
        let mut args = args.into_iter();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--help" | "-h" => help = true,
                "--config" => config = Some(PathBuf::from(next_arg(&mut args, "--config")?)),
                "--bind" => bind = next_arg(&mut args, "--bind")?,
                "--port" => port = parse_port(&next_arg(&mut args, "--port")?)?,
                _ if arg.starts_with("--config=") => config = Some(PathBuf::from(&arg[9..])),
                _ if arg.starts_with("--bind=") => bind = arg[7..].to_owned(),
                _ if arg.starts_with("--port=") => port = parse_port(&arg[7..])?,
                _ => return Err(format!("unknown argument: {arg}")),
            }
        }
        if help {
            return Ok(Self {
                config: PathBuf::new(),
                bind,
                port,
                help,
            });
        }
        Ok(Self {
            config: config.ok_or_else(|| "--config is required".to_owned())?,
            bind,
            port,
            help,
        })
    }
}

fn next_arg(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn parse_port(value: &str) -> Result<u16, String> {
    value
        .parse()
        .map_err(|_| "--port must be between 0 and 65535".to_owned())
}

fn print_usage() {
    println!("Usage: gauge-server --config FILE [--bind ADDRESS] [--port PORT]");
}

fn exit_with_usage(error: &str) -> ! {
    eprintln!("error: {error}");
    print_usage();
    std::process::exit(2);
}

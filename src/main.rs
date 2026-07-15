use std::env;
use std::net::SocketAddr;
use std::sync::Arc;

use pufferclone::api::router;
use pufferclone::engine::Engine;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data_dir = env::var("PUFFERCLONE_DATA").unwrap_or_else(|_| "./data".to_owned());
    let engine = Arc::new(Engine::new(data_dir)?);
    let app = router(engine);
    let address: SocketAddr = "127.0.0.1:8666".parse()?;
    let listener = tokio::net::TcpListener::bind(address).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

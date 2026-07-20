use std::env;
use std::net::SocketAddr;
use std::sync::Arc;

use pufferclone::api::router;
use pufferclone::engine::Engine;
use pufferclone::store_s3::S3Store;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data_dir = env::var("PUFFERCLONE_DATA").unwrap_or_else(|_| "./data".to_owned());
    let engine = match env::var("PUFFERCLONE_S3_URL") {
        Ok(url) => {
            let bucket = env::var("PUFFERCLONE_S3_BUCKET").map_err(|_| {
                "PUFFERCLONE_S3_BUCKET must be set when PUFFERCLONE_S3_URL is configured"
            })?;
            Arc::new(Engine::try_with_store(Arc::new(S3Store::new(
                url, bucket,
            )?))?)
        }
        Err(_) => Arc::new(Engine::new(data_dir)?),
    };
    let app = router(engine);
    let address: SocketAddr = "127.0.0.1:8666".parse()?;
    let listener = tokio::net::TcpListener::bind(address).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

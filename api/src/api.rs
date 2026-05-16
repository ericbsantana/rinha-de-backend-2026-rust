use api::dataset::Dataset;
use api::server;
use std::env;
use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;

fn main() -> std::io::Result<()> {
    let port: u16 = env::var("PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(9999);

    let dataset = Arc::new(Dataset::load(Path::new("out"))?);
    println!("loaded {} vectors", dataset.len());

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;

    rt.block_on(async move {
        let addr = SocketAddr::from(([0, 0, 0, 0], port));
        let listener = tokio::net::TcpListener::bind(addr).await?;
        println!("listening on {}", addr);
        server::run(listener, dataset).await
    })
}

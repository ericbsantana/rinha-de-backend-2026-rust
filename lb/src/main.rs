use std::env;
use tokio::{
    io::{self},
    net::{TcpListener, TcpStream},
};

#[tokio::main]
async fn main() -> io::Result<()> {
    let listen = env::var("LISTEN").unwrap_or_else(|_| "0.0.0.0:9999".to_string());
    let upstreams_env =
        env::var("UPSTREAMS").unwrap_or_else(|_| "127.0.0.1:8080,127.0.0.1:8081".to_string());
    let upstreams: Vec<String> = upstreams_env
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    if upstreams.is_empty() {
        eprintln!("UPSTREAMS env var produced empty list");
        std::process::exit(1);
    }

    println!("lb: listening on {listen}, upstreams = {upstreams:?}");

    let listener = TcpListener::bind(&listen).await?;
    let mut counter: usize = 0;
    loop {
        let (mut client, _socket_address) = listener.accept().await?;
        let upstream = upstreams[counter % upstreams.len()].clone();
        counter = counter.wrapping_add(1);
        tokio::spawn(async move {
            let mut api = TcpStream::connect(&upstream).await?;
            tokio::io::copy_bidirectional(&mut client, &mut api).await
        });
    }
}

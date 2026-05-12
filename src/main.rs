use tokio::{io, net::TcpListener};

#[tokio::main]
async fn main() -> io::Result<()> {
    let listener = TcpListener::bind("0.0.0.0:9999").await?;
    loop {
        let (_tcp_stream, socket_address) = listener.accept().await?;
        println!("connection from {}", socket_address.ip())
    }
}

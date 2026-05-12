use tokio::{
    io::{self},
    net::{TcpListener, TcpStream},
};

#[tokio::main]
async fn main() -> io::Result<()> {
    let upstreams = ["127.0.0.1:8080", "127.0.0.1:8081"];
    let mut counter: usize = 0;
    let listener = TcpListener::bind("0.0.0.0:9999").await?;
    loop {
        let (mut client, _socket_address) = listener.accept().await?;
        let idx = counter % upstreams.len();
        let upstream = upstreams[idx];
        counter += 1;
        tokio::spawn(async move {
            let mut api = TcpStream::connect(upstream).await?;
            tokio::io::copy_bidirectional(&mut client, &mut api).await
        });
    }
}

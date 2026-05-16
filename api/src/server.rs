use crate::dataset::Dataset;
use crate::distance::l2_sq;
use crate::knn::knn;
use crate::vectorize::{Payload, vectorize};
use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use serde::Serialize;
use std::convert::Infallible;
use std::sync::Arc;
use tokio::net::TcpListener;
const K_NEIGHBORS: usize = 5;
const FRAUD_THRESHOLD: f32 = 0.6;

#[derive(Serialize)]
struct FraudResponse {
    approved: bool,
    fraud_score: f32,
}

type ResBody = Full<Bytes>;

#[derive(Debug)]
enum AppError {
    BadJson,
    Internal,
}

impl AppError {
    fn to_response(&self) -> Response<ResBody> {
        let status = match self {
            AppError::BadJson => StatusCode::BAD_REQUEST,
            AppError::Internal => StatusCode::INTERNAL_SERVER_ERROR,
        };
        Response::builder()
            .status(status)
            .body(Full::new(Bytes::new()))
            .unwrap()
    }
}

impl From<serde_json::Error> for AppError {
    fn from(_: serde_json::Error) -> Self {
        AppError::BadJson
    }
}

impl From<hyper::Error> for AppError {
    fn from(_: hyper::Error) -> Self {
        AppError::Internal
    }
}

async fn handle_fraud_score(
    req: Request<Incoming>,
    dataset: Arc<Dataset>,
) -> Result<Response<ResBody>, AppError> {
    let bytes = req.into_body().collect().await?.to_bytes();
    let payload: Payload = serde_json::from_slice(&bytes)?;

    let query = vectorize(&payload);
    let neighbors = knn(&query, &dataset, K_NEIGHBORS, l2_sq);

    let frauds = neighbors.iter().filter(|(_, label)| *label == 1).count();
    let fraud_score = frauds as f32 / K_NEIGHBORS as f32;
    let approved = fraud_score < FRAUD_THRESHOLD;

    let body = serde_json::to_vec(&FraudResponse {
        approved,
        fraud_score,
    })
    .expect("FraudResponse serialization is infallible");

    Ok(Response::builder()
        .header("content-type", "application/json")
        .body(Full::new(Bytes::from(body)))
        .unwrap())
}

async fn handle(
    req: Request<Incoming>,
    dataset: Arc<Dataset>,
) -> Result<Response<ResBody>, Infallible> {
    match (req.method(), req.uri().path()) {
        (&Method::GET, "/ready") => Ok(Response::new(Full::new(Bytes::new()))),
        (&Method::POST, "/fraud-score") => Ok(handle_fraud_score(req, dataset)
            .await
            .unwrap_or_else(|e| e.to_response())),
        _ => Ok(Response::builder()
            .status(StatusCode::NOT_FOUND)
            .body(Full::new(Bytes::new()))
            .unwrap()),
    }
}

pub async fn run(listener: TcpListener, dataset: Arc<Dataset>) -> std::io::Result<()> {
    loop {
        let (stream, _) = listener.accept().await?;
        let io = TokioIo::new(stream);
        let dataset = Arc::clone(&dataset);

        tokio::spawn(async move {
            let service = service_fn(move |req| {
                let dataset = Arc::clone(&dataset);
                async move { handle(req, dataset).await }
            });

            if let Err(e) = hyper::server::conn::http1::Builder::new()
                .serve_connection(io, service)
                .await
            {
                eprintln!("connection error: {e}");
            }
        });
    }
}

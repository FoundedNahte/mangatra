use std::net::SocketAddr;
use std::sync::Arc;

use tonic::transport::{Server, server::Routes};
use axum::Router;
use axum::routing::{post, IntoMakeService};
use hyper::server::conn::AddrIncoming;

use crate::proto::mangatra_service_server::MangatraServiceServer;
use crate::web::http_routes::*;
use crate::web::hybrid_service::{hybrid, MangatraGrpcService, HybridMakeService};
use crate::web::state::HttpServiceState;

pub fn create_server(addr: &SocketAddr) -> hyper::Server<AddrIncoming, HybridMakeService<IntoMakeService<Router>, Routes>> {
    let model_path = String::from("test");
    let tessdata_path = String::from("test");

    let state = Arc::new(HttpServiceState {
        model_path: model_path.clone(),
        tessdata_path: tessdata_path.clone()
    });

    let grpc_service = Server::builder()
    .add_service(MangatraServiceServer::new(MangatraGrpcService {
        model_path,
        tessdata_path
    }))
    .into_service();

    let http_service = Router::new()
        .route("/clean", post(http_clean))
        .route("/extract", post(http_extract))
        .route("/replace", post(http_replace))
        .route("/detect", post(http_detect))
        .with_state(state)
        .into_make_service();

    let hybrid_make_service = hybrid(http_service, grpc_service);

    let server = hyper::Server::bind(&addr).serve(hybrid_make_service);

    server
}
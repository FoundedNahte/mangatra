use std::net::SocketAddr;

use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use mangatra::web::server::create_server;

const IP_ADDRESS: ([u8; 4], u16) = ([0, 0, 0, 0], 3000);

// TODO! Update Axum, Hyper, and Tonic once Tonic gets support for http 1.0.0
#[tokio::main]
async fn main() {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::filter::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                "mangatra=debug,tower_http=debug,axum::rejection=trace".into()
            }),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let addr = SocketAddr::from(IP_ADDRESS);

    let server = create_server(&addr);
    if let Err(e) = server.await {
        eprintln!("Server error: {}", e);
    }

    /*
    let http_server = async move {
        let listener = TcpListener::bind(http_addr).await?;

        loop {
            let (stream, _) = listener.accept().await?;
            let io = TokioIo::new(stream);

            let http_service = hyper::service::service_fn(|request: hyper::Request<hyper::body::Incoming>| {
                http_router.call(request)
            });

            tokio::task::spawn(async move {
                if let Err(err) = auto::Builder::new(TokioExecutor::new())
                    .serve_connection(io, http_service)
                    .await
                {
                    println!("FAILED TO SERVE HTTP CONNECTIOn");
                }
            });
        }
    };

    let grpc_server = async move {
        let listener = TcpListener::bind(grpc_addr).await?;

        loop {
            let (stream, _) = listener.accept().await?;
            let io = TokioIo::new(stream);

            let http_service = hyper::service::service_fn(|request: hyper::Request<hyper::body::Incoming>| {
                grpc_service.call(request)
            });

            tokio::task::spawn(async move {
                if let Err(err) = auto::Builder::new(TokioExecutor::new())
                    .serve_connection(io, grpc_service)
                    .await
                {
                    println!("FAILED TO SERVE HTTP CONNECTIOn");
                }
            });
        }
    }
    loop {
        let (tcp_stream, _) = listener.accept().await?;

        let io = hyper_util::rt::TokioIo::new(tcp_stream);

        let test = grpc_service.clone();

        //let service = TowerToHyperService::new(hybrid);
        let grpc_content_header = b"application/grpc";

        let test = hyper::service::service_fn(move |request: hyper::Request<HybridBody<hyper::body::Incoming, Box<dyn hyper::body::Body>>>| {
            hybrid_service.call(request)
        });

        let hyper_service = hyper::service::service_fn(move |request: hyper::Request<hyper::body::Body>| {
            match request.headers().get("content-type").map(|x| x.as_bytes()) {
                Some(content_type) => {
                    if &content_type[..grpc_content_header.len()] == grpc_content_header {
                        HybridFuture::Grpc(grpc_service.call(request))
                    } else {
                        HybridFuture::Web(http_service.call(request))
                    }
                }
                _ => ()
            }
        });

        tokio::task::spawn(async move {
            if let Err(err) = auto::Builder::new(TokioExecutor::new())
                .serve_connection(io, test)
                .await
            {
                println!("error serving connection: {:?}", err);
            }
        });
    }
    */
}
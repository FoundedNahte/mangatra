// Adapted from https://github.com/snoyberg/tonic-example/blob/master/src/bin/server-hybrid.rs#L199
use std::future::Future;
use std::pin::Pin;
use std::task::Poll;

use hyper::HeaderMap;
use hyper::{body::HttpBody, Request, Response};
use itertools::izip;
use opencv::prelude::MatTraitConst;
use pin_project::pin_project;
use tonic::async_trait;
use tower::Service;

use crate::handlers::*;
use crate::proto::mangatra_service_server::MangatraService;
use crate::proto::{
    Box as ProtoBox, CleanRequest, CleanResponse, DetectRequest, DetectResponse, Detection,
    ExtractRequest, ExtractResponse, ReplaceRequest, ReplaceResponse,
};

impl BoundingBox for ProtoBox {
    fn x(&self) -> i32 {
        self.x
    }

    fn y(&self) -> i32 {
        self.y
    }

    fn width(&self) -> anyhow::Result<u32> {
        Ok(self.width)
    }

    fn height(&self) -> anyhow::Result<u32> {
        Ok(self.height)
    }
}

impl MangatraDetection for Detection {
    type B = ProtoBox;

    fn text(&self) -> &String {
        &self.text
    }

    fn bounding_box(&self) -> Option<&Self::B> {
        self.r#box.as_ref()
    }
}

pub struct MangatraGrpcService {
    pub model_path: String,
    pub tessdata_path: String,
}

#[async_trait]
impl MangatraService for MangatraGrpcService {
    async fn clean(
        &self,
        request: tonic::Request<CleanRequest>,
    ) -> Result<tonic::Response<CleanResponse>, tonic::Status> {
        let (send, recv) = tokio::sync::oneshot::channel::<Result<CleanResponse, anyhow::Error>>();

        let model_path = self.model_path.clone();
        rayon::spawn(move || {
            let image_bytes = &request.get_ref().image;
            let padding: Option<u16> = match request.get_ref().padding {
                Some(padding) => match u16::try_from(padding) {
                    Ok(u16_padding) => Some(u16_padding),
                    Err(e) => {
                        let _ = send.send(Err(e.into()));
                        return;
                    }
                },
                // Default padding of 10px
                None => None,
            };

            match clean_image(image_bytes, padding, &model_path) {
                Ok(cleaned_image_bytes) => {
                    let response = CleanResponse {
                        image: cleaned_image_bytes,
                    };
                    let _ = send.send(Ok(response));
                }
                Err(e) => {
                    let _ = send.send(Err(e));
                }
            }
        });

        match recv.await {
            Ok(cleaned_image_result) => match cleaned_image_result {
                Ok(response) => Ok(tonic::Response::new(response)),
                Err(e) => Err(tonic::Status::from_error(e.into())),
            },
            Err(e) => Err(tonic::Status::from_error(Box::new(e))),
        }
    }

    async fn extract(
        &self,
        request: tonic::Request<ExtractRequest>,
    ) -> Result<tonic::Response<ExtractResponse>, tonic::Status> {
        let (send, recv) = tokio::sync::oneshot::channel::<
            Result<ExtractResponse, Box<dyn std::error::Error + Send + Sync + 'static>>,
        >();

        let model_path = self.model_path.clone();
        let tessdata_path = self.tessdata_path.clone();
        rayon::spawn(move || {
            let image_bytes = &request.get_ref().image;
            let padding: Option<u16> = match request.get_ref().padding {
                Some(padding) => match u16::try_from(padding) {
                    Ok(u16_padding) => Some(u16_padding),
                    Err(e) => {
                        let _ = send.send(Err(Box::new(e)));
                        return;
                    }
                },
                // Default padding of 10px
                None => None,
            };
            let lang = &request.get_ref().lang;

            match extract_text(image_bytes, padding, &model_path, &tessdata_path, lang) {
                Ok((extracted_text, text_regions, origins)) => {
                    let mut detections: Vec<Detection> = Vec::new();
                    for (text, image_region, origin) in izip!(extracted_text, text_regions, origins)
                    {
                        // Proprogate any errors from the try_into statements
                        let box_struct = || -> Result<ProtoBox, anyhow::Error> {
                            Ok(ProtoBox {
                                x: origin.0,
                                y: origin.1,
                                width: image_region.cols().try_into()?,
                                height: image_region.rows().try_into()?,
                            })
                        }();

                        // No errors, shadow the variable as the struct
                        let box_struct = match box_struct {
                            Ok(box_struct) => box_struct,
                            Err(e) => {
                                let _ = send.send(Err(e.into()));
                                return;
                            }
                        };

                        let detection = Detection {
                            text,
                            r#box: Some(box_struct),
                        };

                        detections.push(detection);
                    }

                    // Create the response and send it off
                    let response = ExtractResponse {
                        extractions: detections,
                    };
                    let _ = send.send(Ok(response));
                }
                Err(e) => {
                    let _ = send.send(Err(e.into()));
                }
            }
        });

        match recv.await {
            Ok(text_extraction_result) => match text_extraction_result {
                Ok(response) => Ok(tonic::Response::new(response)),
                Err(e) => Err(tonic::Status::from_error(e)),
            },
            Err(e) => Err(tonic::Status::from_error(Box::new(e))),
        }
    }

    async fn replace(
        &self,
        request: tonic::Request<ReplaceRequest>,
    ) -> Result<tonic::Response<ReplaceResponse>, tonic::Status> {
        let (send, recv) = tokio::sync::oneshot::channel::<
            Result<ReplaceResponse, Box<dyn std::error::Error + Send + Sync + 'static>>,
        >();

        rayon::spawn(move || {
            let image_bytes = &request.get_ref().image;
            let padding: Option<u16> = match request.get_ref().padding {
                Some(padding) => match u16::try_from(padding) {
                    Ok(u16_padding) => Some(u16_padding),
                    Err(e) => {
                        let _ = send.send(Err(Box::new(e)));
                        return;
                    }
                },
                // Default padding of 10px
                None => None,
            };
            let translations = &request.get_ref().translations;

            match replace_image(image_bytes, padding, translations) {
                Ok(replacement_image_bytes) => {
                    let response = ReplaceResponse {
                        image: replacement_image_bytes,
                    };
                    let _ = send.send(Ok(response));
                }
                Err(e) => {
                    let _ = send.send(Err(e.into()));
                }
            }
        });

        match recv.await {
            Ok(replacement_image_result) => match replacement_image_result {
                Ok(response) => Ok(tonic::Response::new(response)),
                Err(e) => Err(tonic::Status::from_error(e)),
            },
            Err(e) => Err(tonic::Status::from_error(Box::new(e))),
        }
    }

    async fn detect(
        &self,
        request: tonic::Request<DetectRequest>,
    ) -> Result<tonic::Response<DetectResponse>, tonic::Status> {
        let (send, recv) = tokio::sync::oneshot::channel::<
            Result<DetectResponse, Box<dyn std::error::Error + Send + Sync + 'static>>,
        >();

        let model_path = self.model_path.clone();
        rayon::spawn(move || {
            let image_bytes = &request.get_ref().image;
            let padding: Option<u16> = match request.get_ref().padding {
                Some(padding) => match u16::try_from(padding) {
                    Ok(u16_padding) => Some(u16_padding),
                    Err(e) => {
                        let _ = send.send(Err(Box::new(e)));
                        return;
                    }
                },
                // Default padding of 10px
                None => None,
            };

            match detect_boxes(image_bytes, padding, &model_path) {
                Ok((image_regions, origins)) => {
                    let mut boxes: Vec<ProtoBox> = Vec::new();

                    for (image_region, origin) in izip!(image_regions, origins) {
                        let box_struct = || -> Result<ProtoBox, anyhow::Error> {
                            Ok(ProtoBox {
                                x: origin.0,
                                y: origin.1,
                                width: image_region.cols().try_into()?,
                                height: image_region.rows().try_into()?,
                            })
                        }();

                        let box_struct = match box_struct {
                            Ok(box_struct) => box_struct,
                            Err(e) => {
                                let _ = send.send(Err(e.into()));
                                return;
                            }
                        };

                        boxes.push(box_struct);
                    }

                    let response = DetectResponse { boxes };
                    let _ = send.send(Ok(response));
                }
                Err(e) => {
                    let _ = send.send(Err(e.into()));
                }
            }
        });

        match recv.await {
            Ok(detection_result) => match detection_result {
                Ok(response) => Ok(tonic::Response::new(response)),
                Err(e) => Err(tonic::Status::from_error(e)),
            },
            Err(e) => Err(tonic::Status::from_error(Box::new(e))),
        }
    }
}

pub fn hybrid<MakeWeb, Grpc>(make_web: MakeWeb, grpc: Grpc) -> HybridMakeService<MakeWeb, Grpc> {
    HybridMakeService { make_web, grpc }
}

pub struct HybridMakeService<MakeWeb, Grpc> {
    make_web: MakeWeb,
    grpc: Grpc,
}

impl<ConnInfo, MakeWeb, Grpc> Service<ConnInfo> for HybridMakeService<MakeWeb, Grpc>
where
    MakeWeb: Service<ConnInfo>,
    Grpc: Clone,
{
    type Response = HybridService<MakeWeb::Response, Grpc>;
    type Error = MakeWeb::Error;
    type Future = HybridMakeServiceFuture<MakeWeb::Future, Grpc>;

    fn poll_ready(
        &mut self,
        cx: &mut std::task::Context,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        self.make_web.poll_ready(cx)
    }

    fn call(&mut self, conn_info: ConnInfo) -> Self::Future {
        HybridMakeServiceFuture {
            web_future: self.make_web.call(conn_info),
            grpc: Some(self.grpc.clone()),
        }
    }
}

#[pin_project]
pub struct HybridMakeServiceFuture<WebFuture, Grpc> {
    #[pin]
    web_future: WebFuture,
    grpc: Option<Grpc>,
}

impl<WebFuture, Web, WebError, Grpc> Future for HybridMakeServiceFuture<WebFuture, Grpc>
where
    WebFuture: Future<Output = Result<Web, WebError>>,
{
    type Output = Result<HybridService<Web, Grpc>, WebError>;

    fn poll(self: Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> Poll<Self::Output> {
        let this = self.project();
        match this.web_future.poll(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(Err(e)) => Poll::Ready(Err(e)),
            Poll::Ready(Ok(web)) => Poll::Ready(Ok(HybridService {
                web,
                grpc: this.grpc.take().expect("Cannot poll twice!"),
            })),
        }
    }
}

#[derive(Clone, Copy)]
pub struct HybridService<Web, Grpc> {
    pub web: Web,
    pub grpc: Grpc,
}

impl<Web, Grpc, WebBody, GrpcBody, RequestBody> Service<Request<RequestBody>>
    for HybridService<Web, Grpc>
where
    RequestBody: HttpBody,
    Web: Service<Request<RequestBody>, Response = Response<WebBody>>,
    Grpc: Service<Request<RequestBody>, Response = Response<GrpcBody>>,
    Web::Error: Into<Box<dyn std::error::Error + Send + Sync + 'static>>,
    Grpc::Error: Into<Box<dyn std::error::Error + Send + Sync + 'static>>,
{
    type Response = Response<HybridBody<WebBody, GrpcBody>>;
    type Error = Box<dyn std::error::Error + Send + Sync + 'static>;
    type Future = HybridFuture<Web::Future, Grpc::Future>;

    fn poll_ready(
        &mut self,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        match self.web.poll_ready(cx) {
            Poll::Ready(Ok(())) => match self.grpc.poll_ready(cx) {
                Poll::Ready(Ok(())) => Poll::Ready(Ok(())),
                Poll::Ready(Err(e)) => Poll::Ready(Err(e.into())),
                Poll::Pending => Poll::Pending,
            },
            Poll::Ready(Err(e)) => Poll::Ready(Err(e.into())),
            Poll::Pending => Poll::Pending,
        }
    }

    fn call(&mut self, req: Request<RequestBody>) -> Self::Future {
        if req.headers().get("content-type").map(|x| x.as_bytes()) == Some(b"application/grpc") {
            HybridFuture::Grpc(self.grpc.call(req))
        } else {
            HybridFuture::Web(self.web.call(req))
        }
    }
}

#[pin_project(project = HybridBodyProj)]
pub enum HybridBody<WebBody, GrpcBody> {
    Web(#[pin] WebBody),
    Grpc(#[pin] GrpcBody),
}

impl<WebBody, GrpcBody> HttpBody for HybridBody<WebBody, GrpcBody>
where
    WebBody: HttpBody + Send + Unpin,
    GrpcBody: HttpBody<Data = WebBody::Data> + Send + Unpin,
    WebBody::Error: std::error::Error + Send + Sync + 'static,
    GrpcBody::Error: std::error::Error + Send + Sync + 'static,
{
    type Data = WebBody::Data;
    type Error = Box<dyn std::error::Error + Send + Sync + 'static>;

    fn is_end_stream(&self) -> bool {
        match self {
            HybridBody::Web(b) => b.is_end_stream(),
            HybridBody::Grpc(b) => b.is_end_stream(),
        }
    }

    fn poll_data(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Option<Result<Self::Data, Self::Error>>> {
        match self.project() {
            HybridBodyProj::Web(b) => b.poll_data(cx).map_err(|e| e.into()),
            HybridBodyProj::Grpc(b) => b.poll_data(cx).map_err(|e| e.into()),
        }
    }

    fn poll_trailers(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context,
    ) -> Poll<Result<Option<HeaderMap>, Self::Error>> {
        match self.project() {
            HybridBodyProj::Web(b) => b.poll_trailers(cx).map_err(|e| e.into()),
            HybridBodyProj::Grpc(b) => b.poll_trailers(cx).map_err(|e| e.into()),
        }
    }
}

#[pin_project(project = HybridFutureProj)]
pub enum HybridFuture<WebFuture, GrpcFuture> {
    Web(#[pin] WebFuture),
    Grpc(#[pin] GrpcFuture),
}

impl<WebFuture, GrpcFuture, WebBody, GrpcBody, WebError, GrpcError> Future
    for HybridFuture<WebFuture, GrpcFuture>
where
    WebFuture: Future<Output = Result<Response<WebBody>, WebError>>,
    GrpcFuture: Future<Output = Result<Response<GrpcBody>, GrpcError>>,
    WebError: Into<Box<dyn std::error::Error + Send + Sync + 'static>>,
    GrpcError: Into<Box<dyn std::error::Error + Send + Sync + 'static>>,
{
    type Output = Result<
        Response<HybridBody<WebBody, GrpcBody>>,
        Box<dyn std::error::Error + Send + Sync + 'static>,
    >;

    fn poll(self: Pin<&mut Self>, cx: &mut std::task::Context) -> Poll<Self::Output> {
        match self.project() {
            HybridFutureProj::Web(a) => match a.poll(cx) {
                Poll::Ready(Ok(res)) => Poll::Ready(Ok(res.map(HybridBody::Web))),
                Poll::Ready(Err(e)) => Poll::Ready(Err(e.into())),
                Poll::Pending => Poll::Pending,
            },
            HybridFutureProj::Grpc(b) => match b.poll(cx) {
                Poll::Ready(Ok(res)) => Poll::Ready(Ok(res.map(HybridBody::Grpc))),
                Poll::Ready(Err(e)) => Poll::Ready(Err(e.into())),
                Poll::Pending => Poll::Pending,
            },
        }
    }
}

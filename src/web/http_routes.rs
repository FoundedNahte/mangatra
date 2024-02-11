use std::sync::Arc;

use axum::extract::{Json, State};
use itertools::izip;
use opencv::prelude::MatTraitConst;
use serde::{Deserialize, Serialize};

use crate::handlers::*;
use crate::web::error::MangatraError;
use crate::web::state::HttpServiceState;

#[derive(Serialize, Deserialize)]
pub struct HttpBox {
    x: i32,
    y: i32,
    width: u32,
    height: u32,
}

impl BoundingBox for HttpBox {
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

#[derive(Serialize, Deserialize)]
pub struct HttpDetection {
    text: String,
    bounding_box: HttpBox,
}

impl MangatraDetection for HttpDetection {
    type B = HttpBox;

    fn text(&self) -> &String {
        &self.text
    }

    fn bounding_box(&self) -> Option<&Self::B> {
        Some(&self.bounding_box)
    }
}

/// Request format for HTTP clean requests
#[derive(Deserialize)]
pub struct HttpCleanRequest {
    /// The image to be cleaned
    image: Vec<u8>,
    padding: Option<u16>,
}

#[derive(Serialize)]
pub struct HttpCleanResponse {
    image: Vec<u8>,
}

pub async fn http_clean(
    State(state): State<Arc<HttpServiceState>>,
    Json(request): Json<HttpCleanRequest>,
) -> Result<Json<HttpCleanResponse>, MangatraError> {
    let (send, recv) =
        tokio::sync::oneshot::channel::<Result<Json<HttpCleanResponse>, anyhow::Error>>();

    rayon::spawn(
        move || match clean_image(&request.image, request.padding, &state.model_path) {
            Ok(cleaned_image_bytes) => {
                let response = HttpCleanResponse {
                    image: cleaned_image_bytes,
                };
                let _ = send.send(Ok(Json(response)));
            }
            Err(e) => {
                let _ = send.send(Err(e));
            }
        },
    );

    match recv.await {
        Ok(cleaned_image_result) => cleaned_image_result.map_err(|error| error.into()),
        Err(e) => Err(e.into()),
    }
}

#[derive(Deserialize)]
pub struct HttpExtractRequest {
    image: Vec<u8>,
    padding: Option<u16>,
    lang: String,
}

#[derive(Serialize)]
pub struct HttpExtractResponse {
    detections: Vec<HttpDetection>,
}

pub async fn http_extract(
    State(state): State<Arc<HttpServiceState>>,
    Json(request): Json<HttpExtractRequest>,
) -> Result<Json<HttpExtractResponse>, MangatraError> {
    let (send, recv) =
        tokio::sync::oneshot::channel::<Result<Json<HttpExtractResponse>, anyhow::Error>>();

    rayon::spawn(move || {
        match extract_text(
            &request.image,
            request.padding,
            &state.model_path,
            &state.tessdata_path,
            &request.lang,
        ) {
            Ok((extracted_text, text_regions, origins)) => {
                let mut detections: Vec<HttpDetection> = Vec::new();
                for (text, image_region, origin) in izip!(extracted_text, text_regions, origins) {
                    // Proprogate any errors from the try_into statements
                    let box_struct = || -> Result<HttpBox, anyhow::Error> {
                        Ok(HttpBox {
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
                            let _ = send.send(Err(e));
                            return;
                        }
                    };

                    let detection = HttpDetection {
                        text,
                        bounding_box: box_struct,
                    };

                    detections.push(detection);
                }

                // Create the response and send it off
                let response = HttpExtractResponse { detections };
                let _ = send.send(Ok(Json(response)));
            }
            Err(e) => {
                let _ = send.send(Err(e));
            }
        }
    });

    match recv.await {
        Ok(text_extraction_result) => text_extraction_result.map_err(|error| error.into()),
        Err(e) => Err(e.into()),
    }
}

#[derive(Deserialize)]
pub struct HttpReplaceRequest {
    image: Vec<u8>,
    padding: Option<u16>,
    translations: Vec<HttpDetection>,
}

#[derive(Serialize)]
pub struct HttpReplaceResponse {
    image: Vec<u8>,
}

pub async fn http_replace(
    State(_state): State<Arc<HttpServiceState>>,
    Json(request): Json<HttpReplaceRequest>,
) -> Result<Json<HttpReplaceResponse>, MangatraError> {
    let (send, recv) =
        tokio::sync::oneshot::channel::<Result<Json<HttpReplaceResponse>, anyhow::Error>>();

    rayon::spawn(move || {
        match replace_image(&request.image, request.padding, &request.translations) {
            Ok(replacement_image_bytes) => {
                let response = HttpReplaceResponse {
                    image: replacement_image_bytes,
                };
                let _ = send.send(Ok(Json(response)));
            }
            Err(e) => {
                let _ = send.send(Err(e));
            }
        }
    });

    match recv.await {
        Ok(replacement_image_result) => replacement_image_result.map_err(|error| error.into()),
        Err(e) => Err(e.into()),
    }
}

#[derive(Deserialize)]
pub struct HttpDetectRequest {
    image: Vec<u8>,
    padding: Option<u16>,
}

#[derive(Serialize)]
pub struct HttpDetectResponse {
    boxes: Vec<HttpBox>,
}

pub async fn http_detect(
    State(state): State<Arc<HttpServiceState>>,
    Json(request): Json<HttpDetectRequest>,
) -> Result<Json<HttpDetectResponse>, MangatraError> {
    let (send, recv) =
        tokio::sync::oneshot::channel::<Result<Json<HttpDetectResponse>, anyhow::Error>>();

    let model_path = state.model_path.clone();
    rayon::spawn(
        move || match detect_boxes(&request.image, request.padding, &model_path) {
            Ok((image_regions, origins)) => {
                let mut boxes: Vec<HttpBox> = Vec::new();

                for (image_region, origin) in izip!(image_regions, origins) {
                    let box_struct = || -> Result<HttpBox, anyhow::Error> {
                        Ok(HttpBox {
                            x: origin.0,
                            y: origin.1,
                            width: image_region.cols().try_into()?,
                            height: image_region.rows().try_into()?,
                        })
                    }();

                    let box_struct = match box_struct {
                        Ok(box_struct) => box_struct,
                        Err(e) => {
                            let _ = send.send(Err(e));
                            return;
                        }
                    };

                    boxes.push(box_struct);
                }

                let response = HttpDetectResponse { boxes };
                let _ = send.send(Ok(Json(response)));
            }
            Err(e) => {
                let _ = send.send(Err(e));
            }
        },
    );

    match recv.await {
        Ok(detection_result) => detection_result.map_err(|error| error.into()),
        Err(e) => Err(e.into()),
    }
}

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
};
use thiserror::Error;
use tokio::sync::oneshot::error::RecvError;

#[derive(Error, Debug)]
pub enum MangatraError {
    #[error(transparent)]
    AnyhowError(#[from] anyhow::Error),
    #[error(transparent)]
    ChannelReceiveError(#[from] RecvError)
}

//pub struct MangatraError(anyhow::Error);

impl IntoResponse for MangatraError {
    fn into_response(self) -> Response {
        match self {
            MangatraError::AnyhowError(err) => (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()).into_response(),
            MangatraError::ChannelReceiveError(err) => (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()).into_response()
        }
    }
}
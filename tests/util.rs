use anyhow::Result;
use reqwest::Body;

use tonic::{transport::Channel, Response};

use mangatra::proto::{*, mangatra_service_client::MangatraServiceClient};

pub struct TestApp {
    pub address: String,
    pub port: u16,
    pub http_client: reqwest::Client,
    pub grpc_client: MangatraServiceClient<Channel>
}

impl TestApp {
    pub async fn http_clean(&self, body: Body) -> reqwest::Response {
        self.http_client
            .post(&format!("{}/clean", &self.address))
            .body(body)
            .send()
            .await
            .expect("Failed to execute http clean request")
    }

    pub async fn grpc_clean(&self) -> CleanResponse {
        let image_bytes: Vec<u8> = Vec::new();

        let request = tonic::Request::new(CleanRequest {
            image: image_bytes,
            padding: None
        });

        let response = &self.grpc_client.clean(request).await.unwrap();

        response.into_inner()
    }

    pub async fn http_extract(&self, body: Body) -> reqwest::Response {
        self.http_client
            .post(&format!("{}/extract", &self.address))
            .body(body)
            .send()
            .await
            .expect("Failed to execute http extract request")
    }

    pub async fn grpc_extract(&self, image_bytes: Vec<u8>, padding: u16, lang: String) {
        let request = tonic::Request::new(ExtractRequest {
            image: image_bytes,
            padding,
            lang
        });

        let response = &self.grpc_client.extract(request).await.unwrap();

        response.into_inner()
    }

    pub async fn http_replace(&self, body: Body) -> reqwest::Response {
        self.http_client
            .post(&format!("{}/replace", &self.address))
            .body(body)
            .send()
            .await
            .expect("Failed to execute http replace request")
    }

    pub async fn grpc_replace(&self, image: Vec<u8>, padding: u16, translations: Vec<Detection>) {
        let request = tonic::Request::new(ReplaceRequest {
            image,
            padding,
            translations
        });

        let response = &self.grpc_client.replace(request).await.unwrap();

        response.into_inner()
    }

    pub async fn http_detect(&self, body: Body) -> reqwest::Response {
        self.http_client
            .post(&format!("{}/detect", &self.address))
            .body(body)
            .send()
            .await
            .expect("Failed to execute http detect request")
    }

    pub async fn grpc_detect(&self, image: Vec<u8>, padding: u16) {
        let request = tonic::Request::new(DetectRequest {
            image,
            padding
        });

        let response = &self.grpc_client.detect(request).await.unwrap();

        response.into_inner()
    }
}

//spawn app
pub async fn spawn_app() -> TestApp {
    todo!()
} 
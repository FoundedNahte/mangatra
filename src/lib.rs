pub const DEFAULT_PADDING: u16 = 10;

pub mod proto {
    tonic::include_proto!("mangatra");
}
pub mod config;
pub mod detection;
pub mod handlers;
pub mod ocr;
pub mod replacer;
pub mod utils;
pub mod web;

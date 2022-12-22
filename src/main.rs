mod detection;
mod ocr;
mod replacer;
mod translation;

use detection::run_inference;
use ocr::extract_text;
use opencv::{core, imgcodecs};
use replacer::{mat_to_image_buffer, replace_text_regions, write_text};
use std::fs::File;
use std::io::prelude::*;
use std::io::LineWriter;
use translation::translate;
fn main() {
    println!("RUNNING");
    match run_inference(10) {
        Ok((text_regions, origins)) => {
            let extracted_text = extract_text(&text_regions).unwrap();
            let translated_text = translate(extracted_text.clone()).unwrap();

            let file = File::create("detected.txt").unwrap();
            let mut file = LineWriter::new(file);

            let translation_file = File::create("translation.txt").unwrap();
            let mut translation_file = LineWriter::new(translation_file);

            let translated_text = translate(extracted_text.clone()).unwrap();

            for text in extracted_text {
                file.write_all(text.as_bytes());
                file.write_all("\n".as_bytes());
            }

            for text in translated_text.clone() {
                translation_file.write_all(text.as_bytes());
                translation_file.write_all("\n".as_bytes());
            }

            let translated_regions = write_text(&text_regions, &translated_text).unwrap();

            let original_image = imgcodecs::imread("img.jpg", imgcodecs::IMREAD_COLOR).unwrap();

            let result =
                replace_text_regions(&original_image, &translated_regions, &origins).unwrap();

            imgcodecs::imwrite("test.png", &result, &core::Vector::new());
        }
        Err(e) => {
            eprintln!("{e}");
        }
    }
}

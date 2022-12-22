mod detection;
mod ocr;
mod translation;

use detection::run_inference;
use ocr::extract_text;
use std::fs::File;
use std::io::prelude::*;
use std::io::LineWriter;
use translation::translate;

fn main() {
    println!("RUNNING");
    match run_inference() {
        Ok(text_regions) => match extract_text(text_regions) {
            Ok(extracted_text) => {
                let file = File::create("detected.txt").unwrap();
                let mut file = LineWriter::new(file);

                let translation_file = File::create("translation.txt").unwrap();
                let mut translation_file = LineWriter::new(translation_file);

                let translated_text = translate(extracted_text.clone()).unwrap();

                for text in extracted_text {
                    println!("{text}");
                    file.write_all(text.as_bytes());
                }

                for text in translated_text {
                    translation_file.write_all(text.as_bytes());
                }
            }
            Err(e) => {
                eprintln!("{e}");
            }
        },
        Err(e) => {
            eprintln!("{e}");
        }
    }
}

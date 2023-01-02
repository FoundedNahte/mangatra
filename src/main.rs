mod config;
mod detection;
mod ocr;
mod replacer;
mod translation;

use anyhow::Result;
use config::{Config, InputMode};
use detection::Detector;
use ocr::OCR;
use opencv::{core, imgcodecs};
use replacer::Replacer;
use translation::translate;


fn run_single_translation(
    input: &str,
    output: &str,
    padding: u16,
    detector: &mut Detector,
    ocr: &mut OCR,
) -> Result<()> {
    let (text_regions, origins) = detector.run_inference(input)?;

    let extracted_text = ocr.extract_text(&text_regions)?;

    let translated_text = translate(extracted_text.clone())?;

    let original_image = imgcodecs::imread(input, imgcodecs::IMREAD_COLOR)?;

    let replacer = Replacer::new(text_regions, translated_text, origins, original_image)?;

    let final_image = replacer.replace_text_regions()?;

    imgcodecs::imwrite(output, &final_image, &core::Vector::new())?;

    Ok(())
}

fn main() -> Result<()> {
    let config = Config::parse()?;

    let mut detector = Detector::new(&config.model, config.padding)?;

    let mut ocr = OCR::new("C:/tools/tesseract/tessdata")?;

    match config.input_mode {
        InputMode::Directory => {}
        InputMode::Image => {
            run_single_translation(
                &config.input,
                &config.output,
                config.padding,
                &mut detector,
                &mut ocr,
            )?;
        }
    }

    Ok(())
}

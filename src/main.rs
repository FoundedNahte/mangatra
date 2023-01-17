mod config;
mod detection;
mod ocr;
mod replacer;
mod translation;

use anyhow::Result;
use config::{Config, InputMode};
use detection::Detector;
use ocr::Ocr;
use opencv::{core, imgcodecs};
use replacer::Replacer;
use serde::Deserialize;
use serde_json::{json, Value};
use translation::translate;

#[derive(Deserialize, Debug)]
struct Json {
    pub text: Vec<String>,
}

pub struct Runtime {
    config: Config,
    detector: Detector,
    ocr: Ocr,
}

impl Runtime {
    pub fn new() -> Result<Runtime> {
        let config = Config::parse()?;
        
        let detector = Detector::new(&config.model, config.padding)?;

        let ocr = Ocr::new("C:/tools/tesseract/tessdata")?;

        Ok(Runtime {
            config,
            detector,
            ocr,
        })
    }

    pub fn run(&mut self) -> Result<()> {
        if self.config.extract_mode {
            self.extract_mode()?;
        } else if self.config.replace_mode {
            self.replace_mode()?;
        } else {
            match self.config.input_mode {
                InputMode::Image => {
                    self.run_single_translation()?;
                }
                InputMode::Directory => {
                    self.run_multi_translation()?;
                }
            }
        }

        Ok(())
    }

    fn run_single_translation(&mut self) -> Result<()> {
        

        let (text_regions, origins) = self.detector.run_inference(&self.config.input)?;

        let extracted_text = self.ocr.extract_text(&text_regions)?;

        let translated_text = translate(extracted_text)?;

        let original_image = imgcodecs::imread(&self.config.input, imgcodecs::IMREAD_COLOR)?;

        let replacer = Replacer::new(
            text_regions,
            translated_text,
            origins,
            original_image,
            self.config.padding,
        )?;

        let final_image = replacer.replace_text_regions()?;

        imgcodecs::imwrite(&self.config.output, &final_image, &core::Vector::new())?;

        Ok(())
    }

    fn extract_mode(&mut self) -> Result<()> {
        let (text_regions, origins) = self.detector.run_inference(&self.config.input)?;

        let extracted_text = self.ocr.extract_text(&text_regions)?;

        let data = json!({ "text": extracted_text });

        std::fs::write(&self.config.output, serde_json::to_string_pretty(&data)?)?;

        Ok(())
    }

    fn replace_mode(&mut self) -> Result<()> {
        let data = std::fs::read_to_string(&self.config.text)?;

        let data = serde_json::from_str::<Json>(&data)?;

        let (text_regions, origins) = self.detector.run_inference(&self.config.input)?;

        let original_image = imgcodecs::imread(&self.config.input, imgcodecs::IMREAD_COLOR)?;

        let replacer = Replacer::new(
            text_regions,
            data.text,
            origins,
            original_image,
            self.config.padding,
        )?;

        let final_image = replacer.replace_text_regions()?;

        imgcodecs::imwrite(&self.config.output, &final_image, &core::Vector::new())?;

        Ok(())
    }

    fn run_multi_translation(&mut self) -> Result<()> {
        unimplemented!()
    }
}

fn main() -> Result<()> {
    let mut runtime = Runtime::new()?;

    runtime.run()?;

    Ok(())
}

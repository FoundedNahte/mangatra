use anyhow::{Error, Result};
use mangatra::config::{Config, InputMode};
use mangatra::detection::Detector;
use mangatra::ocr::Ocr;
use mangatra::replacer::Replacer;
use mangatra::translation::translate;
use mangatra::utils::validation;
use opencv::{core, imgcodecs};
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::{json, Value};
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

#[derive(Deserialize, Debug)]
struct Json {
    pub text: Vec<String>,
}

pub struct Runtime {
    config: Config,
    detector: Arc<Mutex<Detector>>,
    ocr: Arc<Mutex<Ocr>>,
}

impl Runtime {
    pub fn new() -> Result<Runtime> {
        let config = Config::parse()?;

        let detector = Arc::new(Mutex::new(Detector::new(&config.model, config.padding)?));

        let ocr = Arc::new(Mutex::new(Ocr::new("C:/tools/tesseract/tessdata")?));

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
            self.run_translation()?;
        }

        Ok(())
    }

    fn run_translation(&mut self) -> Result<()> {
        if self.config.input_mode == InputMode::Image {
            let final_image = Self::translate_image(
                &self.config.input,
                self.config.padding,
                &mut self.ocr,
                &mut self.detector,
            )?;

            imgcodecs::imwrite(&self.config.output, &final_image, &core::Vector::new())?;
        } else {
            let (input_images, output_paths) = self.walk_image_directory()?;

            let image_data: Vec<Result<core::Mat, Error>> = input_images
                .par_iter()
                .map(|input_path| {
                    Self::translate_image(
                        input_path,
                        self.config.padding,
                        &self.ocr,
                        &self.detector,
                    )
                })
                .collect();

            for (final_image, output_path) in image_data.iter().zip(output_paths.iter()) {
                match (final_image, output_path.to_str()) {
                    // Write to output path
                    (Ok(data), Some(path)) => {
                        imgcodecs::imwrite(path, &data, &core::Vector::new())?;
                    }
                    
                    // Catches errors in translating the image (OpenCV and libtesseract errors)
                    (Err(e), _) => eprintln!("{e}"),

                    // Catches errors with path not being in UTF-8
                    (_, None) => {
                        let file_name = output_path.display();
                        eprintln!("{file_name} must be UTF-8 compatible.");
                    }
                }
            }
        }

        Ok(())
    }

    // Main function for extraction mode. Depending on input mode, will extract text from a single image or multiple.
    fn extract_mode(&mut self) -> Result<()> {
        if self.config.input_mode == InputMode::Image {
            let data = Self::extract_text(&self.config.input, &mut self.ocr, &mut self.detector)?;

            std::fs::write(&self.config.output, serde_json::to_string_pretty(&data)?)?;
        } else {
            let (input_images, output_paths) = self.walk_image_directory()?;

            // Parallel processing of images
            let image_data: Vec<Result<Value, Error>> = input_images
                .par_iter()
                .map(|input_path| Self::extract_text(input_path, &self.ocr, &self.detector))
                .collect();

            // Write text to output location
            for (data_result, output_path) in image_data.iter().zip(output_paths.iter()) {
                match data_result {
                    Ok(data) => {
                        std::fs::write(output_path, serde_json::to_string_pretty(&data)?)?;
                    }
                    Err(e) => eprintln!("{e}"),
                }
            }
        }

        Ok(())
    }

    // Main function for replacement mode. Will replace a single image or multiple depending on input mode.
    fn replace_mode(&mut self) -> Result<()> {
        if self.config.input_mode == InputMode::Image {
        } else {
        }

        /*
        let data = std::fs::read_to_string(&self.config.text)?;

        let data = serde_json::from_str::<Json>(&data)?;

        let final_image = Self::replace_text(
            &self.config.input,
            data,
            self.config.padding,
            &mut self.detector,
        )?;

        imgcodecs::imwrite(&self.config.output, &final_image, &core::Vector::new())?;


        if self.config.input_mode == InputMode::Image {
            let data = std::fs::read_to_string(&self.config.text)?;

            let data = serde_json::from_str::<Json>(&data)?;

            let final_image = Self::replace_text(
                &self.config.input,
                data,
                self.config.padding,
                &mut self.detector,
            )?;
            imgcodecs::imwrite(&self.config.output, &final_image, &core::Vector::new())?;
        } else {
            let mut input_images: Vec<String> = Vec::new();
            let mut output_paths: Vec<PathBuf> = Vec::new();
        }
        Ok(())
        */
        unimplemented!()
    }

    // Translation helper function used in both single and multi image translation functions.
    fn translate_image(
        input: &str,
        padding: u16,
        ocr: &Arc<Mutex<Ocr>>,
        detector: &Arc<Mutex<Detector>>,
    ) -> Result<core::Mat> {
        let text_regions;
        let origins;

        {
            let detector = &mut *detector.lock().unwrap();
            (text_regions, origins) = detector.run_inference(input)?;
        }

        let extracted_text;

        {
            let ocr = &mut *ocr.lock().unwrap();
            extracted_text = ocr.extract_text(&text_regions)?;
        }

        let translated_text = translate(extracted_text)?;

        let original_image = imgcodecs::imread(input, imgcodecs::IMREAD_COLOR)?;

        let replacer = Replacer::new(
            text_regions,
            translated_text,
            origins,
            original_image,
            padding,
        )?;

        let final_image = replacer.replace_text_regions()?;

        Ok(final_image)
    }

    // Text extraction helper function to extract and return text from a single image
    fn extract_text(
        input: &str,
        ocr: &Arc<Mutex<Ocr>>,
        detector: &Arc<Mutex<Detector>>,
    ) -> Result<Value> {
        let text_regions;

        {
            let detector = &mut *detector.lock().unwrap();
            text_regions = detector.run_inference(input)?.0;
        }

        let extracted_text;

        {
            let ocr = &mut *ocr.lock().unwrap();
            extracted_text = ocr.extract_text(&text_regions)?;
        }

        let data = json!({ "text": extracted_text });

        Ok(data)
    }

    // Replacement helper function to replace text in single image and return a OpenCV Mat
    fn replace_text(
        input: &str,
        data: Json,
        padding: u16,
        detector: &Arc<Mutex<Detector>>,
    ) -> Result<core::Mat> {
        let text_regions;
        let origins;

        {
            let detector = &mut *detector.lock().unwrap();
            (text_regions, origins) = detector.run_inference(input)?;
        }

        let original_image = imgcodecs::imread(input, imgcodecs::IMREAD_COLOR)?;

        let replacer = Replacer::new(text_regions, data.text, origins, original_image, padding)?;

        let final_image = replacer.replace_text_regions()?;

        Ok(final_image)
    }

    // Get images from input directory for processing
    fn walk_image_directory(&self) -> Result<(Vec<String>, Vec<PathBuf>)> {
        let mut input_images: Vec<String> = Vec::new();
        let mut output_paths: Vec<PathBuf> = Vec::new();

        // Walking through the input directory appending image files to be processed
        for result in fs::read_dir(&self.config.input)? {
            match result {
                Ok(dir_entry) => {
                    let dir_entry_path = dir_entry.path();

                    if validation::validate_image(&dir_entry.path()) {
                        match dir_entry_path.to_str() {
                            Some(path_string) => match dir_entry_path.file_stem() {
                                Some(file_stem) => {
                                    let mut output_path = PathBuf::new();
                                    output_path.push(&self.config.output);
                                    output_path.push(file_stem);
                                    output_path.set_extension("json");

                                    input_images.push(path_string.to_string());
                                    output_paths.push(output_path);
                                }
                                None => {
                                    let bad_path = dir_entry_path.display();
                                    eprintln!("{bad_path} needs to have a UTF-8 compatible name.");
                                }
                            },
                            None => {
                                let bad_path = dir_entry_path.display();
                                eprintln!("{bad_path} needs to have a UTF-8 compatible name.");
                            }
                        }
                    }
                }
                Err(e) => {
                    eprintln!("{e}");
                }
            }
        }

        Ok((input_images, output_paths))
    }

    // Get text data from text directory for replacement
    fn walk_text_directory(&self) -> Result<()> {
        unimplemented!()
    }
}

fn main() -> Result<()> {
    let mut runtime = Runtime::new()?;

    runtime.run()?;

    Ok(())
}

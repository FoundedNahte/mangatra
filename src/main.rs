use anyhow::Result;
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressIterator};
use mangatra::config::{Config, InputMode};
use mangatra::detection::Detector;
use mangatra::ocr::Ocr;
use mangatra::replacer::Replacer;
use mangatra::translation::translate;
use mangatra::utils::{image_conversion, validation};
use opencv::core;
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::{json, Value};
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

#[derive(Deserialize, Debug, Clone)]
struct Json {
    pub text: Vec<String>,
}

// Runtime struct that holds configuration and other needed components for translation
pub struct Runtime {
    config: Arc<Config>,
}

impl Runtime {
    pub fn new() -> Result<Runtime> {
        let config = Arc::new(Config::parse()?);

        Ok(Runtime { config })
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
            let final_image = Self::translate_image(self.config.clone(), &self.config.input)?;

            image_conversion::mat_to_image_buffer(&final_image)?.save(&self.config.output)?;
        } else {
            let (input_images, output_paths, _) = self.walk_image_directory(false)?;

            // Multi-threading or single-threading based on configuration
            if self.config.single {
                input_images
                    .into_iter()
                    .zip(output_paths.into_iter())
                    .progress()
                    .for_each(|(input_path, output_path)| {
                        let final_image = Self::translate_image(self.config.clone(), &input_path);

                        match (final_image, output_path.to_str()) {
                            // Write to output path
                            (Ok(data), Some(path)) => {
                                match image_conversion::mat_to_image_buffer(&data) {
                                    Ok(buffer) => {
                                        if let Err(e) = buffer.save(path) {
                                            eprintln!("{e}")
                                        }
                                    }
                                    Err(e) => eprintln!("{e}"),
                                }
                            }

                            // Catches errors in translating the image (OpenCV and libtesseract errors)
                            (Err(e), _) => eprintln!("{e}"),

                            // Catches errors with path not being in UTF-8
                            (_, None) => {
                                let file_name = output_path.display();
                                eprintln!("{file_name} must be UTF-8 compatible.");
                            }
                        }
                    })
            } else {
                let total_length = input_images.len() as u64;

                input_images
                    .into_par_iter()
                    .zip(output_paths.into_par_iter())
                    .progress_count(total_length)
                    .for_each(|(input_path, output_path)| {
                        let final_image = Self::translate_image(self.config.clone(), &input_path);

                        match (final_image, output_path.to_str()) {
                            // Write to output path
                            (Ok(data), Some(path)) => {
                                match image_conversion::mat_to_image_buffer(&data) {
                                    Ok(buffer) => {
                                        if let Err(e) = buffer.save(path) {
                                            eprintln!("{e}")
                                        }
                                    }
                                    Err(e) => eprintln!("{e}"),
                                }
                            }

                            // Catches errors in translating the image (OpenCV and libtesseract errors)
                            (Err(e), _) => eprintln!("{e}"),

                            // Catches errors with path not being in UTF-8
                            (_, None) => {
                                let file_name = output_path.display();
                                eprintln!("{file_name} must be UTF-8 compatible.");
                            }
                        }
                    })
            }
        }

        Ok(())
    }

    // Main function for extraction mode. Depending on input mode, will extract text from a single image or multiple.
    fn extract_mode(&mut self) -> Result<()> {
        if self.config.input_mode == InputMode::Image {
            let data = Self::extract_text(self.config.clone(), &self.config.input)?;

            std::fs::write(&self.config.output, serde_json::to_string_pretty(&data)?)?;
        } else {
            let (input_images, output_paths, _) = self.walk_image_directory(true)?;

            if self.config.single {
                input_images
                    .into_iter()
                    .zip(output_paths.into_iter())
                    .progress()
                    .for_each(|(input_path, output_path)| {
                        let data_result = Self::extract_text(self.config.clone(), &input_path);

                        match data_result {
                            Ok(data) => match serde_json::to_string_pretty(&data) {
                                Ok(json_data) => {
                                    if let Err(e) = std::fs::write(output_path, json_data) {
                                        eprintln!("{e}")
                                    }
                                }
                                Err(e) => eprintln!("{e}"),
                            },
                            Err(e) => eprintln!("{e}"),
                        }
                    })
            } else {
                let total_length = input_images.len() as u64;

                input_images
                    .into_par_iter()
                    .zip(output_paths.into_par_iter())
                    .progress_count(total_length)
                    .for_each(|(input_path, output_path)| {
                        let data_result = Self::extract_text(self.config.clone(), &input_path);

                        match data_result {
                            Ok(data) => match serde_json::to_string_pretty(&data) {
                                Ok(json_data) => {
                                    if let Err(e) = std::fs::write(output_path, json_data) {
                                        eprintln!("{e}")
                                    }
                                }
                                Err(e) => eprintln!("{e}"),
                            },
                            Err(e) => eprintln!("{e}"),
                        }
                    })
            }
        }

        Ok(())
    }

    // Main function for replacement mode. Will replace a single image or multiple depending on input mode.
    fn replace_mode(&mut self) -> Result<()> {
        if self.config.input_mode == InputMode::Image {
            // Validation of single image paths is done during configuration
            let data = std::fs::read_to_string(&self.config.text)?;

            let data = serde_json::from_str::<Json>(&data)?;

            let final_image = Self::replace_text(self.config.clone(), &data, &self.config.input)?;

            image_conversion::mat_to_image_buffer(&final_image)?.save(&self.config.output)?;
        } else {
            let (input_images, output_paths, file_stems) = self.walk_image_directory(false)?;

            let text_data = self.walk_text_directory(file_stems)?;

            if self.config.single {
                input_images
                    .into_iter()
                    .zip(text_data.into_iter())
                    .zip(output_paths.into_iter())
                    .progress()
                    .for_each(|((input_path, data), output_path)| {
                        let image_data =
                            Self::replace_text(self.config.clone(), &data, &input_path);

                        match (image_data, output_path.to_str()) {
                            // Write to output path
                            (Ok(data), Some(path)) => {
                                match image_conversion::mat_to_image_buffer(&data) {
                                    Ok(buffer) => {
                                        if let Err(e) = buffer.save(path) {
                                            eprintln!("{e}")
                                        }
                                    }
                                    Err(e) => eprintln!("{e}"),
                                };
                            }

                            // Catches errors in translating the image (OpenCV and libtesseract errors)
                            (Err(e), _) => eprintln!("{e}"),

                            // Catches errors with path not being in UTF-8
                            (_, None) => {
                                let file_name = output_path.display();
                                eprintln!("{file_name} must be UTF-8 compatible.");
                            }
                        }
                    })
            } else {
                let total_length = input_images.len() as u64;

                input_images
                    .into_par_iter()
                    .zip(text_data.into_par_iter())
                    .zip(output_paths.into_par_iter())
                    .progress_count(total_length)
                    .for_each(|((input_path, data), output_path)| {
                        let image_data =
                            Self::replace_text(self.config.clone(), &data, &input_path);

                        match (image_data, output_path.to_str()) {
                            // Write to output path
                            (Ok(data), Some(path)) => {
                                match image_conversion::mat_to_image_buffer(&data) {
                                    Ok(buffer) => {
                                        if let Err(e) = buffer.save(path) {
                                            eprintln!("{e}")
                                        }
                                    }
                                    Err(e) => eprintln!("{e}"),
                                };
                            }

                            // Catches errors in translating the image (OpenCV and libtesseract errors)
                            (Err(e), _) => eprintln!("{e}"),

                            // Catches errors with path not being in UTF-8
                            (_, None) => {
                                let file_name = output_path.display();
                                eprintln!("{file_name} must be UTF-8 compatible.");
                            }
                        }
                    })
            }
        }

        Ok(())
    }

    // Translation helper function used in both single and multi image translation functions.
    fn translate_image(config: Arc<Config>, input: &str) -> Result<core::Mat> {
        let mut detector = Detector::new(&config.model, config.padding)?;
        let mut ocr = Ocr::new(&config.data)?;

        let (text_regions, origins) = detector.run_inference(input)?;

        let extracted_text = ocr.extract_text(&text_regions)?;

        let translated_text = translate(extracted_text)?;

        let original_image = image::open(input)?;
        let original_image = image_conversion::image_buffer_to_mat(original_image.to_rgb8())?;

        let replacer = Replacer::new(
            text_regions,
            translated_text,
            origins,
            original_image,
            config.padding,
        )?;

        let final_image = replacer.replace_text_regions()?;

        Ok(final_image)
    }

    // Text extraction helper function to extract and return text from a single image
    fn extract_text(config: Arc<Config>, input: &str) -> Result<Value> {
        let mut detector = Detector::new(&config.model, config.padding)?;
        let mut ocr = Ocr::new(&config.data)?;

        let (text_regions, _) = detector.run_inference(input)?;

        let extracted_text = ocr.extract_text(&text_regions)?;

        let data = json!({ "text": extracted_text });

        Ok(data)
    }

    // Replacement helper function to replace text in single image and return a OpenCV Mat
    fn replace_text(config: Arc<Config>, data: &Json, input: &str) -> Result<core::Mat> {
        let mut detector = Detector::new(&config.model, config.padding)?;

        let (text_regions, origins) = detector.run_inference(input)?;

        let original_image = image::open(input)?;
        let original_image = image_conversion::image_buffer_to_mat(original_image.to_rgb8())?;

        let replacer = Replacer::new(
            text_regions,
            data.text.clone(),
            origins,
            original_image,
            config.padding,
        )?;

        let final_image = replacer.replace_text_regions()?;

        Ok(final_image)
    }

    // Get images from input directory for processing
    fn walk_image_directory(
        &self,
        extract_mode: bool,
    ) -> Result<(Vec<String>, Vec<PathBuf>, Vec<String>)> {
        let mut input_images: Vec<String> = Vec::new();
        let mut output_paths: Vec<PathBuf> = Vec::new();
        let mut file_stems: Vec<String> = Vec::new();

        // Walking through the input directory appending image files to be processed
        for result in fs::read_dir(&self.config.input)? {
            match result {
                Ok(dir_entry) => {
                    let dir_entry_path = dir_entry.path();

                    if !dir_entry_path.is_dir() {
                        match validation::validate_image(&dir_entry.path()) {
                            Ok(()) => {
                                match dir_entry_path.to_str() {
                                    Some(path_string) => match dir_entry_path.file_stem() {
                                        Some(file_stem) if file_stem.to_str().is_some() => {
                                            // Pattern guard ensures that to_str gives "Some"
                                            let file_stem = file_stem.to_str().unwrap();

                                            // Create the output path for each text_file
                                            let mut output_path = PathBuf::new();

                                            file_stems.push(file_stem.to_string());

                                            if extract_mode {
                                                output_path.push(&self.config.output);
                                                output_path.push(file_stem);
                                                output_path.set_extension("json");
                                            } else {
                                                let output_filename = format!("{file_stem}_output");

                                                output_path.push(&self.config.output);
                                                output_path.push(output_filename);
                                                output_path.set_extension("png");
                                            }

                                            input_images.push(path_string.to_string());
                                            output_paths.push(output_path);
                                        }
                                        _ => {
                                            let bad_path = dir_entry_path.display();
                                            eprintln!(
                                                "{bad_path} needs to have a UTF-8 compatible name."
                                            );
                                        }
                                    },
                                    None => {
                                        let bad_path = dir_entry_path.display();
                                        eprintln!(
                                            "{bad_path} needs to have a UTF-8 compatible name."
                                        );
                                    }
                                }
                            }
                            Err(e) => {
                                eprintln!("{e}");
                            }
                        }
                    }
                }
                Err(e) => {
                    eprintln!("{e}");
                }
            }
        }

        Ok((input_images, output_paths, file_stems))
    }

    // Get text data from text directory for replacement
    fn walk_text_directory(&self, input_stems: Vec<String>) -> Result<Vec<Json>> {
        let mut text_paths: Vec<PathBuf> = Vec::new();
        let mut text_data: Vec<Json> = Vec::new();

        for result in fs::read_dir(&self.config.text)? {
            match result {
                Ok(dir_entry) => {
                    let dir_entry_path = dir_entry.path();

                    if !dir_entry_path.is_dir() {
                        if let Ok(()) = validation::validate_text(&dir_entry.path()) {
                            text_paths.push(dir_entry_path);
                        }
                    }
                }
                Err(e) => {
                    eprintln!("{e}");
                }
            }
        }

        println!("{text_paths:?}");

        validation::validate_replace_mode(&input_stems, &text_paths)?;

        for text_path in text_paths.into_iter() {
            match text_path.to_str() {
                Some(path_string) => {
                    let data =
                        serde_json::from_str::<Json>(&std::fs::read_to_string(path_string)?)?;

                    text_data.push(data);
                }
                None => {
                    let bad_path = text_path.display();
                    eprintln!("{bad_path} needs to have a UTF-8 compatible name.");
                }
            }
        }

        Ok(text_data)
    }
}

fn main() -> Result<()> {
    let before = Instant::now();

    let mut runtime = Runtime::new()?;
    runtime.run()?;

    println!("Finished in {:.2?} secs", before.elapsed());

    Ok(())
}

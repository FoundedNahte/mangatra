use anyhow::Result;
use globwalk::GlobWalkerBuilder;
use indexmap::IndexMap;
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressIterator};
use itertools::Itertools;
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
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

#[derive(Deserialize, Debug, Clone)]
struct Json {
    #[serde(with = "indexmap::serde_seq")]
    pub text: IndexMap<String, String>,
}

impl From<IndexMap<String, String>> for Json {
    fn from(text: IndexMap<String, String>) -> Json {
        Json { text }
    }
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
            let final_image = Self::translate_image(Arc::clone(&self.config), &self.config.input)?;

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
                        let final_image =
                            Self::translate_image(Arc::clone(&self.config), &input_path);

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
                        let final_image =
                            Self::translate_image(Arc::clone(&self.config), &input_path);

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
            let data = Self::extract_text(Arc::clone(&self.config), &self.config.input)?;

            std::fs::write(&self.config.output, serde_json::to_string_pretty(&data)?)?;
        } else {
            let (input_images, output_paths, _) = self.walk_image_directory(true)?;

            if self.config.single {
                input_images
                    .into_iter()
                    .zip(output_paths.into_iter())
                    .progress()
                    .for_each(|(input_path, output_path)| {
                        let data_result = Self::extract_text(Arc::clone(&self.config), &input_path);

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
                        let data_result = Self::extract_text(Arc::clone(&self.config), &input_path);

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

            let final_image =
                Self::replace_text(Arc::clone(&self.config), &data, &self.config.input)?;

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
                        let image_data = Self::replace_text(
                            Arc::clone(&self.config),
                            &Json::from(data),
                            &input_path,
                        );

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
                        let image_data = Self::replace_text(
                            Arc::clone(&self.config),
                            &Json::from(data),
                            &input_path,
                        );

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

        let translated_text = translate(&extracted_text)?;

        let text_pairs = extracted_text
            .iter()
            .zip(translated_text.iter())
            .map(|(original, translation)| (original.as_str(), translation.as_str()))
            .collect::<IndexMap<&str, &str>>();

        let original_image = image::open(input)?;
        let original_image = image_conversion::image_buffer_to_mat(original_image.to_rgb8())?;

        let replacer = Replacer::new(
            text_regions,
            &text_pairs,
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

        let text_pairs: IndexMap<&str, &str> =
            extracted_text
                .iter()
                .fold(IndexMap::new(), |mut acc, text| {
                    acc.insert(text.as_str(), "");
                    acc
                });

        let data = json!(text_pairs);

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
            &data.text,
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
        let image_walker = GlobWalkerBuilder::from_patterns(
            &self.config.input,
            &["*{jpg,JPG,jpeg,JPEG,png,PNG,webp,WEBP,tiff,TIFF}"],
        )
        .follow_links(false)
        .build()?;

        Ok(image_walker
            .into_iter()
            .filter_map(|image| match image {
                Ok(image) => {
                    if let Some(image_path) = image.path().to_str() {
                        match image.path().file_stem() {
                            Some(file_stem) if file_stem.to_str().is_some() => {
                                let file_stem = file_stem.to_str().unwrap();

                                let mut output_path = PathBuf::new();

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

                                Some((image_path.to_string(), output_path, file_stem.to_string()))
                            }
                            _ => {
                                eprintln!(
                                    "{} needs to have a UTF-8 compatible name.",
                                    image.path().display()
                                );
                                None
                            }
                        }
                    } else {
                        eprintln!(
                            "{} needs to have a UTF-8 compatible name.",
                            image.path().display()
                        );
                        None
                    }
                }
                Err(e) => {
                    eprintln!("{e}");
                    None
                }
            })
            .multiunzip::<(Vec<String>, Vec<PathBuf>, Vec<String>)>())
    }

    // Get text data from text directory for replacement
    fn walk_text_directory(
        &self,
        input_stems: Vec<String>,
    ) -> Result<Vec<IndexMap<String, String>>> {
        let text_walker = GlobWalkerBuilder::from_patterns(&self.config.text, &["*{json,JSON}"])
            .follow_links(false)
            .build()?;

        let text_paths = text_walker
            .into_iter()
            .filter_map(|text| match text {
                Ok(text_res) => {
                    if text_res.path().clone().to_str().is_none() {
                        eprintln!(
                            "{} needs to hve a UTF-8 compatible name.",
                            text_res.path().display()
                        );
                        return None;
                    }
                    Some(text_res.into_path())
                }
                Err(e) => {
                    eprintln!("{e}");
                    None
                }
            })
            .collect::<Vec<PathBuf>>();

        validation::validate_replace_mode(input_stems, &text_paths)?;

        let mut text_data: Vec<IndexMap<String, String>> = Vec::new();

        for text_path in text_paths.iter() {
            match text_path.to_str() {
                Some(path_string) => {
                    let data = serde_json::from_str::<IndexMap<String, String>>(
                        &std::fs::read_to_string(path_string)?,
                    )?;

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

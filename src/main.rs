use anyhow::Result;
use globwalk::GlobWalkerBuilder;
use indexmap::IndexMap;
use indicatif::{ParallelProgressIterator, ProgressIterator};
use itertools::{multizip, Itertools};
use mangatra::config::{Config, InputMode, RuntimeMode};
use mangatra::detection::Detector;
use mangatra::ocr::Ocr;
use mangatra::replacer::Replacer;
use mangatra::utils::{image_conversion, validation};
use opencv::core;
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tracing::error;
use tracing_subscriber;

type InputPaths = Vec<String>;
type OutputPaths = Vec<PathBuf>;
type CleanPagePaths = Vec<PathBuf>;
type FileStems = Vec<String>;

struct DirectoryWalkerState {
    pub input_image_paths: Vec<String>,
    pub output_paths: Vec<PathBuf>,
    pub cleaned_page_paths: Vec<PathBuf>,
    pub file_stems: Vec<String>,
}

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
    /**
     * Creates a new runtime context
     */
    pub fn new() -> Result<Runtime> {
        let config = Arc::new(Config::parse()?);

        Ok(Runtime { config })
    }

    pub fn run(&mut self) -> Result<()> {
        match self.config.runtime_mode {
            RuntimeMode::Extraction => self.extract_mode()?,
            RuntimeMode::Replacement => self.replace_mode()?,
        }

        Ok(())
    }

    // Main function for extraction mode. Depending on input mode, will extract text from a single image or multiple.
    fn extract_mode(&mut self) -> Result<()> {
        if self.config.input_mode == InputMode::Image {
            let (data_result, cleaned_page) =
                Self::extract_text(Arc::clone(&self.config), &self.config.input_files_path)?;

            match cleaned_page {
                Some(clean_page_mat) if self.config.clean => {
                    match image_conversion::mat_to_image_buffer(&clean_page_mat) {
                        Ok(buffer) => {
                            if let Err(e) = buffer.save(&self.config.cleaned_page_path) {
                                error!(
                                    "Error saving cleaned page for {}: {e}",
                                    self.config.input_files_path
                                );
                            }
                        }
                        Err(e) => error!(
                            "Error processing cleaned image for {}: {e}",
                            self.config.input_files_path
                        ),
                    }
                }
                _ => {}
            }

            std::fs::write(
                &self.config.output_path,
                serde_json::to_string_pretty(&data_result)?,
            )?;
        } else {
            let DirectoryWalkerState {
                input_image_paths,
                output_paths,
                cleaned_page_paths,
                ..
            } = self.walk_directories()?;

            let extraction_closure =
                |(input_path, output_path, cleaned_page_path): (String, PathBuf, PathBuf)| {
                    match Self::extract_text(Arc::clone(&self.config), &input_path) {
                        Ok((data_result, cleaned_page)) => {
                            // If a cleaned page was return, write it to the cleaned_page location
                            match cleaned_page {
                                Some(clean_page_mat) if self.config.clean => {
                                    match image_conversion::mat_to_image_buffer(&clean_page_mat) {
                                        Ok(buffer) => {
                                            if let Err(e) = buffer.save(cleaned_page_path) {
                                                error!(
                                                "Error saving cleaned page for {input_path}: {e}"
                                            )
                                            }
                                        }
                                        Err(e) => error!(
                                            "Error processing cleaned image for {input_path}: {e}"
                                        ),
                                    }
                                }
                                _ => {}
                            }

                            // Write the text to a json file
                            match serde_json::to_string_pretty(&data_result) {
                                Ok(json_data) => {
                                    if let Err(e) = std::fs::write(&output_path, json_data) {
                                        error!(
                                        "Error writing extracted text to {} for {input_path}: {e}",
                                        output_path.display()
                                    )
                                    }
                                }
                                Err(e) => error!(
                                    "Error converting JSON value to string for {input_path}: {e}"
                                ),
                            }
                        }
                        Err(e) => {
                            error!("Error extracting text for {input_path}: {e}")
                        }
                    }
                };

            if self.config.single {
                multizip((input_image_paths, output_paths, cleaned_page_paths))
                    .progress()
                    .for_each(extraction_closure)
            } else {
                let total_length = input_image_paths.len() as u64;

                multizip((input_image_paths, output_paths, cleaned_page_paths))
                    .progress_count(total_length)
                    .for_each(extraction_closure)
            }
        }

        Ok(())
    }

    // Main function for replacement mode. Will replace a single image or multiple depending on input mode.
    fn replace_mode(&mut self) -> Result<()> {
        if self.config.input_mode == InputMode::Image {
            // Validation of single image paths is done during configuration
            let data = std::fs::read_to_string(&self.config.text_files_path)?;

            let data = serde_json::from_str::<Json>(&data)?;

            let final_image = Self::replace_text(
                Arc::clone(&self.config),
                &data,
                &self.config.input_files_path,
            )?;

            image_conversion::mat_to_image_buffer(&final_image)?.save(&self.config.output_path)?;
        } else {
            let DirectoryWalkerState {
                input_image_paths,
                output_paths,
                cleaned_page_paths: _,
                file_stems,
            } = self.walk_directories()?;
            let text_data = walk_text_directory(&self.config.text_files_path, file_stems)?;

            let replacement_closure = |((input_path, data), output_path): (
                (String, IndexMap<String, String>),
                PathBuf,
            )| {
                let image_data =
                    Self::replace_text(Arc::clone(&self.config), &Json::from(data), &input_path);

                match (image_data, output_path.to_str()) {
                    // Write to output path
                    (Ok(data), Some(path)) => {
                        match image_conversion::mat_to_image_buffer(&data) {
                            Ok(buffer) => {
                                if let Err(e) = buffer.save(path) {
                                    error!("Error processing {input_path}: {e}")
                                }
                            }
                            Err(e) => error!("Error processing {input_path}: {e}"),
                        };
                    }

                    // Catches errors in translating the image (OpenCV and libtesseract errors)
                    (Err(e), _) => error!("OpenCV/Tesseract error with {input_path}: {e}"),

                    // Catches errors with path not being in UTF-8
                    (_, None) => {
                        let file_name = output_path.display();
                        error!("{file_name} must be UTF-8 compatible.")
                    }
                }
            };

            if self.config.single {
                input_image_paths
                    .into_iter()
                    .zip(text_data)
                    .zip(output_paths)
                    .progress()
                    .for_each(replacement_closure)
            } else {
                let total_length = input_image_paths.len() as u64;

                input_image_paths
                    .into_par_iter()
                    .zip(text_data.into_par_iter())
                    .zip(output_paths.into_par_iter())
                    .progress_count(total_length)
                    .for_each(replacement_closure)
            }
        }

        Ok(())
    }

    // Text extraction helper function to extract and return text from a single image
    fn extract_text(config: Arc<Config>, input: &str) -> Result<(Value, Option<core::Mat>)> {
        let mut detector = Detector::new(&config.model_path, config.padding)?;
        let mut ocr = Ocr::new(&config.lang, &config.tesseract_data_path)?;

        let (text_regions, origins) = detector.run_inference(input)?;

        let extracted_text = ocr.extract_text(&text_regions)?;

        let text_pairs: IndexMap<&str, &str> =
            extracted_text
                .iter()
                .fold(IndexMap::new(), |mut acc, text| {
                    acc.insert(text.as_str(), "");
                    acc
                });

        let data = json!(text_pairs);

        if config.clean {
            let original_image =
                image_conversion::image_buffer_to_mat(image::open(input)?.to_rgb8())?;
            let replacer: Replacer<'_, String> =
                Replacer::new(text_regions, None, origins, original_image, config.padding)?;

            let cleaned_page = replacer.clean_page()?;

            Ok((data, Some(cleaned_page)))
        } else {
            Ok((data, None))
        }
    }

    // Replacement helper function to replace text in single image and return a OpenCV Mat
    fn replace_text(config: Arc<Config>, data: &Json, input: &str) -> Result<core::Mat> {
        let mut detector = Detector::new(&config.model_path, config.padding)?;

        let (text_regions, origins) = detector.run_inference(input)?;

        let original_image = image::open(input)?;
        let original_image = image_conversion::image_buffer_to_mat(original_image.to_rgb8())?;

        let replacer = Replacer::new(
            text_regions,
            Some(&data.text),
            origins,
            original_image,
            config.padding,
        )?;

        let final_image = replacer.replace_text_regions()?;

        Ok(final_image)
    }

    fn walk_directories(&self) -> Result<DirectoryWalkerState> {
        let (input_image_paths, output_paths, cleaned_page_paths, file_stems) =
            walk_image_directory(
                self.config.runtime_mode,
                &self.config.input_files_path,
                &self.config.output_path,
                &self.config.cleaned_page_path,
                self.config.clean,
            )?;
        Ok(DirectoryWalkerState {
            input_image_paths,
            output_paths,
            cleaned_page_paths,
            file_stems,
        })
    }
}

// Get images from input directory for processing
fn walk_image_directory(
    runtime_mode: RuntimeMode,
    input_files_path: &String,
    output_path: &String,
    cleaned_page_path: &String,
    clean_pages: bool,
) -> Result<(InputPaths, OutputPaths, CleanPagePaths, FileStems)> {
    // Build a directory walker for the input path
    let image_walker = GlobWalkerBuilder::from_patterns(
        input_files_path,
        &["*{jpg,JPG,jpeg,JPEG,png,PNG,webp,WEBP,tiff,TIFF}"],
    )
    .follow_links(false)
    .build()?;

    /*
        Walks the input directory and creates three vecs:
        1) Input paths
        2) Output paths (text or replacement)
        3) File paths for cleaned pages
        3) File stems
    */
    Ok(image_walker
        .into_iter()
        .filter_map(|image| match image {
            Ok(image) => {
                if let Some(image_path) = image.path().to_str() {
                    match image.path().file_stem() {
                        Some(file_stem) if file_stem.to_str().is_some() => {
                            let file_stem = file_stem.to_str().unwrap();

                            let mut image_output_path = PathBuf::new();

                            match runtime_mode {
                                RuntimeMode::Extraction => {
                                    image_output_path.push(output_path);
                                    image_output_path.push(file_stem);
                                    image_output_path.set_extension("json");
                                }
                                RuntimeMode::Replacement => {
                                    let image_output_filename = format!("{file_stem}_output");
                                    image_output_path.push(output_path);
                                    image_output_path.push(image_output_filename);
                                    image_output_path.set_extension("png");
                                }
                            }

                            let mut image_cleaned_page_path = PathBuf::new();
                            if clean_pages {
                                image_cleaned_page_path.push(cleaned_page_path);
                                image_cleaned_page_path.push(format!("{file_stem}_cleaned"));
                                image_cleaned_page_path.set_extension("png");
                            }

                            Some((
                                image_path.to_string(),
                                image_output_path,
                                image_cleaned_page_path,
                                file_stem.to_string(),
                            ))
                        }
                        _ => {
                            error!(
                                "{} needs to have a UTF-8 compatible name",
                                image.path().display()
                            );
                            None
                        }
                    }
                } else {
                    error!(
                        "{} needs to have a UTF-8 compatible name",
                        image.path().display()
                    );
                    None
                }
            }
            Err(e) => {
                error!("{e}");
                None
            }
        })
        .multiunzip::<(InputPaths, OutputPaths, CleanPagePaths, FileStems)>())
}

// Get text data from text directory for replacement
fn walk_text_directory(
    text_files_path: &String,
    input_stems: Vec<String>,
) -> Result<Vec<IndexMap<String, String>>> {
    let text_walker = GlobWalkerBuilder::from_patterns(text_files_path, &["*{json,JSON}"])
        .follow_links(false)
        .build()?;

    let text_paths = text_walker
        .into_iter()
        .filter_map(|text| match text {
            Ok(text_res) => {
                if text_res.path().to_str().is_none() {
                    error!(
                        "{} needs to have a UTF-8 compatible name",
                        text_res.path().display()
                    );
                    return None;
                }
                Some(text_res.into_path())
            }
            Err(e) => {
                error!("{e}");
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
                error!("{bad_path} needs to have a UTF-8 compatible name");
            }
        }
    }

    Ok(text_data)
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .event_format(
            tracing_subscriber::fmt::format()
                .with_file(true)
                .with_line_number(true),
        )
        .init();

    let before = Instant::now();

    let run = || -> Result<()> {
        let mut runtime = Runtime::new()?;
        runtime.run()?;
        Ok(())
    };

    match run() {
        Ok(()) => {}
        Err(e) => error!("{e}"),
    }

    println!("Finished in {:.2?}", before.elapsed());

    Ok(())
}

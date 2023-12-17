use crate::utils::validation;
use anyhow::{bail, ensure, Result};
use clap::Parser;
use std::path::{Path, PathBuf};
use tracing::instrument;

#[derive(Clone, Copy, Debug)]
pub enum RuntimeMode {
    Extraction,
    Replacement,
}

#[derive(Debug)]
pub struct Config {
    pub runtime_mode: RuntimeMode,
    pub clean: bool,
    pub text_files_path: String,
    pub input_files_path: String,
    pub output_path: String,
    pub cleaned_page_path: String,
    pub model_path: String,
    pub tesseract_data_path: String,
    pub lang: String,
    pub padding: u16,
    pub input_mode: InputMode,
    pub single: bool,
}

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[arg(
        short,
        long,
        help = "Input path for a directory of images or single image"
    )]
    pub input: PathBuf,
    #[arg(
        short,
        long,
        help = "Specify output location for text or image outputs. If not specified, application will use the same directory as the input"
    )]
    pub output: Option<PathBuf>,
    #[arg(
        short,
        long,
        help = "[Optional] Specify a path to the translated JSONs"
    )]
    pub text: Option<PathBuf>,
    #[arg(
        short,
        long,
        help = "Path to the YOLOv5 detection weights (ONNX format)"
    )]
    pub model: PathBuf,
    #[arg(short, long, help = "Specify the language for tesseract")]
    pub lang: String,
    #[arg(
        short,
        long,
        help = "[Optional] Specify path to the tessdata folder for tesseract. If no path is specified, the application will look under the 'TESSDATA_PREFIX' environment variable"
    )]
    pub data: Option<PathBuf>,
    #[arg(short, long, help = "Specify size of padding for text regions")]
    pub padding: Option<u16>,
    #[arg(long, help = "Use single-threading for image processing")]
    pub single: bool,
    #[arg(
        long,
        help = "If set, the program will output cleaned pages in PNG format in the output directory"
    )]
    pub clean: bool,
}

#[derive(PartialEq, Debug, Clone, Copy)]
pub enum InputMode {
    Directory,
    Image,
}

enum PathType {
    Input(PathBuf),
    Output(PathBuf),
    Text(Option<PathBuf>),
    CleanedPage(Option<PathBuf>),
    Model(PathBuf),
    Data(PathBuf),
}

impl std::fmt::Display for PathType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            PathType::Input(_) => write!(f, "Input"),
            PathType::Output(_) => write!(f, "Output"),
            PathType::Text(_) => write!(f, "Text"),
            PathType::CleanedPage(_) => write!(f, "CleanedPage"),
            PathType::Model(_) => write!(f, "Model"),
            PathType::Data(_) => write!(f, "Data"),
        }
    }
}

impl Config {
    #[instrument(name = "config_parse")]
    pub fn parse() -> Result<Config> {
        // Default values for text and padding
        let mut text: Option<PathBuf> = None;
        let mut padding: u16 = 10;

        let cli = Cli::parse();

        let runtime_mode = match cli.text.is_none() {
            true => RuntimeMode::Extraction,
            false => RuntimeMode::Replacement,
        };
        let clean = cli.text.is_none() && cli.clean;

        // Determining input type (directory or single image)
        let input_mode = Self::get_input_mode(&cli.input)?;

        // If supplied an output path, check to see if it's the same type as the input
        // Otherwise use a default path based on whether running normally or in extract mode
        let output = Self::get_output_path(&cli.input, &cli.output, runtime_mode, input_mode)?;

        // Make sure the model file is in the ONNX format
        validation::validate_model(&cli.model)?;

        let data_path = validation::validate_data(&cli.data)?;

        // If in replace mode, make sure the text file is a JSON
        if let RuntimeMode::Replacement = runtime_mode {
            if let Some(text_path) = cli.text {
                if !text_path.is_dir() {
                    validation::validate_text(&text_path)?;
                }

                text = Some(text_path);
            }
        }

        if let Some(custom_padding) = cli.padding {
            padding = custom_padding;
        }

        let mut clean_page_path = None;
        if clean {
            clean_page_path = Some(Self::get_cleaned_page_path(
                &cli.input,
                &cli.output,
                input_mode,
            )?)
        }

        Ok(Config {
            runtime_mode,
            clean,
            text_files_path: Self::path_into_string(PathType::Text(text))?,
            input_files_path: Self::path_into_string(PathType::Input(cli.input))?,
            output_path: Self::path_into_string(PathType::Output(output))?,
            cleaned_page_path: Self::path_into_string(PathType::CleanedPage(clean_page_path))?,
            model_path: Self::path_into_string(PathType::Model(cli.model))?,
            tesseract_data_path: Self::path_into_string(PathType::Data(data_path))?,
            lang: cli.lang,
            padding,
            input_mode,
            single: cli.single,
        })
    }

    // Helper function to test if paths are valid as well as determine InputMode for input and output
    fn path_into_string(path: PathType) -> Result<String> {
        let pathbuf = match &path {
            PathType::Input(path) => path,
            PathType::Output(path) => path,
            PathType::Text(Some(path)) => path,
            PathType::Text(None) => return Ok(String::new()),
            PathType::CleanedPage(Some(path)) => path,
            PathType::CleanedPage(None) => return Ok(String::new()),
            PathType::Model(path) => path,
            PathType::Data(path) => path,
        };
        match pathbuf.to_str() {
            Some(path_string) => Ok(path_string.to_string()),
            None => {
                bail!("Make sure {path} is UTF-8 comaptible.")
            }
        }
    }

    // Parses input mode from the input path
    fn get_input_mode(input: &Path) -> Result<InputMode> {
        let input_mode = match input.extension() {
            Some(_) => match validation::validate_image(input) {
                Ok(()) => InputMode::Image,
                Err(e) => {
                    bail!("{e}");
                }
            },
            None => {
                if !input.is_dir() {
                    bail!("Input must be either a directory or supported image type.");
                }

                InputMode::Directory
            }
        };

        Ok(input_mode)
    }

    fn get_cleaned_page_path(
        input_path: &Path,
        output_path: &Option<PathBuf>,
        input_mode: InputMode,
    ) -> Result<PathBuf> {
        let input_stem = match &input_path.file_stem() {
            Some(file_stem) if file_stem.to_str().is_some() => file_stem.to_str().unwrap(),
            _ => {
                panic!("Error trying to parse the input path file stem: {} needs to have a UTF-8 compatible name", &input_path.display());
            }
        };

        // If an output path was specified, have the cleaned pages go into the output path's root, else use the input path's root
        let mut cleaned_page_path: PathBuf = match output_path {
            Some(path) => match path.parent() {
                Some(root) => root.to_path_buf(),
                None => panic!("Error trying to get the path root for {}", path.display()),
            },
            // Default path
            None => {
                Path::new(".").to_path_buf()
            }
        };
        cleaned_page_path.push(&format!("{input_stem}_cleaned"));

        if let InputMode::Image = input_mode {
            cleaned_page_path.set_extension("png");
        }

        Ok(cleaned_page_path)
    }

    // Parses output path based on the parameters given
    fn get_output_path(
        input_path: &Path,
        output: &Option<PathBuf>,
        runtime_mode: RuntimeMode,
        input_mode: InputMode,
    ) -> Result<PathBuf> {
        let output: PathBuf = match output {
            Some(path) => {
                match runtime_mode {
                    RuntimeMode::Extraction => {
                        if let Some(extension) = path.extension() {
                            ensure!(
                                Some("json") == extension.to_str(),
                                "Output path must be a JSON if running in extract mode."
                            )
                        } else if !path.is_dir() {
                            bail!("Output path must lead to a directory or json file for writing.")
                        }
                    }
                    RuntimeMode::Replacement => {
                        ensure!(
                            !(path.is_dir() && input_mode == InputMode::Image),
                            "Output and Input must be of the same type."
                        );
                        /* !(!path.is_dir() && input_mode == InputMode::Directory),*/
                        ensure!(
                            path.is_dir() || !(input_mode == InputMode::Directory),
                            "Output and Input must be of the same type."
                        )
                    }
                }

                path.to_path_buf()
            }
            // Create default path
            None => {
                // Get the input file stem so we can build the default output paths
                let input_stem = match &input_path.file_stem() {
                    Some(file_stem) if file_stem.to_str().is_some() => file_stem.to_str().unwrap(),
                    _ => {
                        panic!("Error trying to parse the input path file stem: {} needs to have a UTF-8 compatible name", &input_path.display());
                    }
                };

                let default_text_path = format!("./{input_stem}");
                let default_output_path = format!("./{input_stem}_output");

                match runtime_mode {
                    RuntimeMode::Extraction => match input_mode {
                        InputMode::Image => {
                            Path::new(&format!("{default_text_path}.json")).to_path_buf()
                        }
                        InputMode::Directory => {
                            let default_text_directory_path = format!("{default_text_path}_text");
                            let text_dir = Path::new(&default_text_directory_path);

                            if !text_dir.is_dir() {
                                match std::fs::create_dir(&default_text_path) {
                                    Ok(()) => {}
                                    Err(err) => {
                                        bail!(err)
                                    }
                                }
                            }

                            text_dir.to_path_buf()
                        }
                    },
                    RuntimeMode::Replacement => match input_mode {
                        InputMode::Image => {
                            Path::new(&format!("{default_output_path}.png")).to_path_buf()
                        }
                        InputMode::Directory => {
                            let output_dir = Path::new(&default_output_path);

                            if !output_dir.is_dir() {
                                match std::fs::create_dir(&default_output_path) {
                                    Ok(()) => {}
                                    Err(err) => {
                                        bail!(err)
                                    }
                                }
                            }

                            output_dir.to_path_buf()
                        }
                    },
                }
            }
        };

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use crate::config::{Config, InputMode, PathType};
    use tempfile::{Builder, TempDir};

    // Testing "path_into_string" functionality
    #[test]
    fn test_path_into_string() {
        let utf8_path = Path::new("./temp.jpg");

        match Config::path_into_string(PathType::Input(utf8_path.to_path_buf())) {
            Ok(s) => {
                assert_eq!(&s, "./temp.jpg")
            }
            Err(e) => {
                panic!("Error: {e}")
            }
        }
    }

    // Testing input_mode function for images and directories
    #[test]
    fn test_input_mode() {
        let jpg_path = Path::new("./test.jpg");
        let jpeg_path = Path::new("./test.jpeg");
        let png_path = Path::new("./test.png");
        let webp_path = Path::new("./test.webp");

        assert_eq!(InputMode::Image, Config::get_input_mode(jpg_path).unwrap());

        assert_eq!(InputMode::Image, Config::get_input_mode(jpeg_path).unwrap());

        assert_eq!(InputMode::Image, Config::get_input_mode(png_path).unwrap());

        assert_eq!(InputMode::Image, Config::get_input_mode(webp_path).unwrap());

        let input_dir = TempDir::new().unwrap();

        assert_eq!(
            InputMode::Directory,
            Config::get_input_mode(input_dir.path()).unwrap()
        )
    }

    // Same as above except testing for errors
    #[test]
    fn test_input_mode_error() {
        let bad_input = Path::new("./test.onnx");

        let error = Config::get_input_mode(bad_input).unwrap_err();

        assert_eq!(
            format!("{error}"),
            "Image file must be in one of the specified formats: JPG, PNG, WebP."
        );

        let bad_dir_input = Builder::new().suffix("").tempfile().unwrap();

        let error = Config::get_input_mode(bad_dir_input.path()).unwrap_err();

        assert_eq!(
            format!("{error}"),
            "Input must be either a directory or supported image type."
        )
    }

    // Tests "get_output_path" when given an output path and not running in extract mode.
    #[test]
    fn test_output_replace_path() {
        // Test directory output validation
        let test_dir_path = TempDir::new().unwrap();

        let dir_result = Config::get_output_path(
            &test_dir_path.path().to_path_buf(),
            &Some(test_dir_path.path().to_path_buf()),
            crate::config::RuntimeMode::Replacement,
            InputMode::Directory,
        )
        .unwrap();

        assert_eq!(dir_result, test_dir_path.path());

        // Test single image output validation
        let test_image = Path::new("./test.json");

        let image_result = Config::get_output_path(
            &test_dir_path.path().to_path_buf(),
            &Some(test_image.to_path_buf()),
            crate::config::RuntimeMode::Replacement,
            InputMode::Image,
        )
        .unwrap();

        assert_eq!(image_result, test_image.to_path_buf())
    }

    // Same as above but testing for errors
    #[test]
    fn test_output_replace_path_error() {
        // Test directory output validation
        let test_dir_path = TempDir::new().unwrap();

        let dir_err = Config::get_output_path(
            &test_dir_path.path().to_path_buf(),
            &Some(test_dir_path.path().to_path_buf()),
            crate::config::RuntimeMode::Replacement,
            InputMode::Image,
        )
        .unwrap_err();

        assert_eq!(
            format!("{dir_err}"),
            "Output and Input must be of the same type."
        );

        // Test single image output validation
        let test_image = Path::new("./test.json");

        let image_error = Config::get_output_path(
            &test_dir_path.path().to_path_buf(),
            &Some(test_image.to_path_buf()),
            crate::config::RuntimeMode::Replacement,
            InputMode::Directory,
        )
        .unwrap_err();

        assert_eq!(
            format!("{image_error}"),
            "Output and Input must be of the same type."
        );
    }

    // Tests "get_output_path" when given an output path and running in extract mode
    #[test]
    fn test_output_extract_path() {
        let text_output_path = Path::new("./test.json");

        let json_result = Config::get_output_path(
            &text_output_path.to_path_buf(),
            &Some(text_output_path.to_path_buf()),
            crate::config::RuntimeMode::Extraction,
            InputMode::Image,
        )
        .unwrap();

        assert_eq!(json_result, text_output_path.to_path_buf());

        let test_dir_path = TempDir::new().unwrap();

        let dir_result = Config::get_output_path(
            &text_output_path.to_path_buf(),
            &Some(test_dir_path.path().to_path_buf()),
            crate::config::RuntimeMode::Extraction,
            InputMode::Directory,
        )
        .unwrap();

        assert_eq!(dir_result, test_dir_path.path().to_path_buf());
    }

    // Same as above except testing for errors
    #[test]
    fn test_output_extract_path_error() {
        let bad_dir_input = Builder::new().suffix("").tempfile().unwrap();

        let extract_error = Config::get_output_path(
            &bad_dir_input.path().to_path_buf(),
            &Some(bad_dir_input.path().to_path_buf()),
            crate::config::RuntimeMode::Extraction,
            InputMode::Directory,
        )
        .unwrap_err();

        assert_eq!(
            format!("{extract_error}"),
            "Output path must lead to a directory or json file for writing."
        )
    }

    // Tests "get_output_path" when not given a path and running in extract mode
    #[test]
    fn test_output_extract_mode_default_path() {
        let temp_input_dir = TempDir::new().unwrap();
        let input_stem = &temp_input_dir.path().file_stem().unwrap().to_str().unwrap();

        let default_json_path = Config::get_output_path(
            &temp_input_dir.path().to_path_buf(),
            &None,
            crate::config::RuntimeMode::Extraction,
            InputMode::Image,
        )
        .unwrap();
        assert_eq!(
            Path::new(&format!("./{input_stem}.json")),
            default_json_path
        );

        let default_dir_path = Config::get_output_path(
            &temp_input_dir.path().to_path_buf(),
            &None,
            crate::config::RuntimeMode::Extraction,
            InputMode::Directory,
        )
        .unwrap();
        assert_eq!(Path::new(&format!("./{input_stem}_text")), default_dir_path)
    }

    // Test default paths for directory and image mode
    #[test]
    fn test_output_default_path() {
        let temp_input_dir = TempDir::new().unwrap();
        let input_stem = &temp_input_dir.path().file_stem().unwrap().to_str().unwrap();

        let default_image_path = Config::get_output_path(
            &temp_input_dir.path().to_path_buf(),
            &None,
            crate::config::RuntimeMode::Replacement,
            InputMode::Image,
        )
        .unwrap();
        assert_eq!(
            Path::new(&format!("./{input_stem}_output.png")),
            default_image_path
        );

        let default_dir_path = Config::get_output_path(
            &temp_input_dir.path().to_path_buf(),
            &None,
            crate::config::RuntimeMode::Replacement,
            InputMode::Directory,
        )
        .unwrap();
        assert_eq!(
            Path::new(&format!("./{input_stem}_output")),
            default_dir_path
        )
    }
}

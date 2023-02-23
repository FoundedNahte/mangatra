use crate::utils::validation;
use anyhow::{bail, ensure, Result};
use clap::Parser;
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub struct Config {
    pub extract_mode: bool,
    pub replace_mode: bool,
    pub text: String,
    pub input: String,
    pub output: String,
    pub model: String,
    pub data: String,
    pub padding: u16,
    pub input_mode: InputMode,
    pub single: bool,
}

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[arg(long, help = "Pass '--extract' to extract text from images.")]
    pub extract: bool,
    #[arg(
        long,
        help = "Pass '--replace' to replace text regions in input images from a JSON containing translated text"
    )]
    pub replace: bool,
    #[arg(
        short,
        long,
        help = "If using in \"replace\" mode, a path to a translated text json must be specified"
    )]
    pub text: Option<PathBuf>,
    #[arg(short, long, help = "Input Path - Directory of JPGs or a single JPG")]
    pub input: PathBuf,
    #[arg(
        short,
        long,
        help = "Optional Output Path - Specify output location for text or image outputs. If not specified, application will revert to the current directory."
    )]
    pub output: Option<PathBuf>,
    #[arg(
        short,
        long,
        help = "YoloV5 Model Path - A path to a detection model must be specified (ONNX format)."
    )]
    pub model: PathBuf,
    #[arg(
        short,
        long,
        help = "Libtesseract Data Path - Specify path to libtesseract data folder. If no path is specified, the application will look under the 'TESSDATA_PREFIX' environment variable."
    )]
    pub data: Option<PathBuf>,
    #[arg(
        short,
        long,
        help = "Specify size of padding for text regions (Tinkering may improve OCR)"
    )]
    pub padding: Option<u16>,
    #[arg(long, help = "Use single-threading when processing a folder")]
    pub single: bool,
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
    Model(PathBuf),
    Data(PathBuf),
}

impl std::fmt::Display for PathType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            PathType::Input(_) => write!(f, "Input"),
            PathType::Output(_) => write!(f, "Output"),
            PathType::Model(_) => write!(f, "Model"),
            PathType::Data(_) => write!(f, "Data"),
            PathType::Text(_) => write!(f, "Text"),
        }
    }
}

impl Config {
    pub fn parse() -> Result<Config> {
        // Default values for text and padding
        let mut text: Option<PathBuf> = None;
        let mut padding: u16 = 10;

        let cli = Cli::parse();

        // If extract or replace mode is toggled, make sure only one of the two is toggled
        ensure!(
            !(cli.extract & cli.replace),
            "Run in either extract or replace mode but not both."
        );

        // Ensure that a text path is provided if running in replace mode
        if cli.replace {
            ensure!(
                cli.replace && cli.text.is_some(),
                "A path to a text json is required for replace mode."
            );
        }

        // Determining input type (directory or single image)
        let input_mode = Self::get_input_mode(&cli.input)?;

        // If supplied an output path, check to see if it's the same type as the input
        // Otherwise use a default path based on whether running normally or in extract mode
        let output = Self::get_output_path(cli.output, cli.extract, input_mode)?;

        // Make sure the model file is in the ONNX format
        validation::validate_model(&cli.model)?;

        let data_path = validation::validate_data(&cli.data)?;

        // If in replace mode, make sure the text file is a JSON
        if cli.replace {
            if let Some(text_path) = cli.text {
                validation::validate_text(&text_path)?;

                text = Some(text_path);
            }
        }

        if let Some(custom_padding) = cli.padding {
            padding = custom_padding;
        }

        Ok(Config {
            extract_mode: cli.extract,
            replace_mode: cli.replace,
            text: Self::path_into_string(&PathType::Text(text))?,
            input: Self::path_into_string(&PathType::Input(cli.input))?,
            output: Self::path_into_string(&PathType::Output(output))?,
            model: Self::path_into_string(&PathType::Model(cli.model))?,
            data: Self::path_into_string(&PathType::Data(data_path))?,
            padding,
            input_mode,
            single: cli.single,
        })
    }

    // Helper function to test if paths are valid as well as determine InputMode for input and output
    fn path_into_string(path: &PathType) -> Result<String> {
        let pathbuf = match &path {
            PathType::Input(path) => path,
            PathType::Output(path) => path,
            PathType::Model(path) => path,
            PathType::Data(path) => path,
            PathType::Text(Some(path)) => path,
            PathType::Text(None) => return Ok(String::new()),
        };
        match pathbuf.clone().to_str() {
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

    // Parses output path based on the parameters given
    fn get_output_path(
        output: Option<PathBuf>,
        extract_mode: bool,
        input_mode: InputMode,
    ) -> Result<PathBuf> {
        let output: PathBuf = match output {
            Some(path) => {
                if extract_mode {
                    if let Some(extension) = path.extension() {
                        ensure!(
                            Some("json") == extension.to_str(),
                            "Output path must be a JSON if running in extract mode."
                        )
                    } else if !path.is_dir() {
                        bail!("Output path must lead to a directory or json file for writing.")
                    }
                } else {
                    ensure!(
                        !(path.is_dir() && input_mode == InputMode::Image),
                        "Output and Input must be of the same type."
                    );
                    ensure!(
                        !(!path.is_dir() && input_mode == InputMode::Directory),
                        "Output and Input must be of the same type."
                    )
                }

                path
            }
            // Create default path
            None => {
                if extract_mode {
                    if input_mode == InputMode::Image {
                        Path::new("./text.json").to_path_buf()
                    } else {
                        let text_dir = Path::new("./text");

                        if !text_dir.is_dir() {
                            match std::fs::create_dir("./text") {
                                Ok(()) => {}
                                Err(err) => {
                                    bail!(err)
                                }
                            }
                        }

                        text_dir.to_path_buf()
                    }
                } else {
                    match input_mode {
                        InputMode::Directory => {
                            if !Path::new("./output").is_dir() {
                                match std::fs::create_dir("./output") {
                                    Ok(()) => {}
                                    Err(err) => {
                                        bail!(err)
                                    }
                                }
                            }

                            Path::new("./output").to_path_buf()
                        }
                        InputMode::Image => Path::new("./output.jpg").to_path_buf(),
                    }
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

        match Config::path_into_string(&PathType::Input(utf8_path.to_path_buf())) {
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
            Some(test_dir_path.path().to_path_buf()),
            false,
            InputMode::Directory,
        )
        .unwrap();

        assert_eq!(dir_result, test_dir_path.path());

        // Test single image output validation
        let test_image = Path::new("./test.json");

        let image_result =
            Config::get_output_path(Some(test_image.to_path_buf()), false, InputMode::Image)
                .unwrap();

        assert_eq!(image_result, test_image.to_path_buf())
    }

    // Same as above but testing for errors
    #[test]
    fn test_output_replace_path_error() {
        // Test directory output validation
        let test_dir_path = TempDir::new().unwrap();

        let dir_err = Config::get_output_path(
            Some(test_dir_path.path().to_path_buf()),
            false,
            InputMode::Image,
        )
        .unwrap_err();

        assert_eq!(
            format!("{dir_err}"),
            "Output and Input must be of the same type."
        );

        // Test single image output validation
        let test_image = Path::new("./test.json");

        let image_error =
            Config::get_output_path(Some(test_image.to_path_buf()), false, InputMode::Directory)
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

        let json_result =
            Config::get_output_path(Some(text_output_path.to_path_buf()), true, InputMode::Image)
                .unwrap();

        assert_eq!(json_result, text_output_path.to_path_buf());

        let test_dir_path = TempDir::new().unwrap();

        let dir_result = Config::get_output_path(
            Some(test_dir_path.path().to_path_buf()),
            false,
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
            Some(bad_dir_input.path().to_path_buf()),
            true,
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
        let default_json_path = Config::get_output_path(None, true, InputMode::Image).unwrap();

        assert_eq!(Path::new("./text.json"), default_json_path);

        let default_dir_path = Config::get_output_path(None, true, InputMode::Directory).unwrap();

        assert_eq!(Path::new("./text"), default_dir_path)
    }

    // Test default paths for directory and image mode
    #[test]
    fn test_output_default_path() {
        let default_image_path = Config::get_output_path(None, false, InputMode::Image).unwrap();

        assert_eq!(Path::new("./output.jpg"), default_image_path);

        let default_dir_path = Config::get_output_path(None, false, InputMode::Directory).unwrap();

        assert_eq!(Path::new("./output"), default_dir_path)
    }
}

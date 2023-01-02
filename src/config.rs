use anyhow::{bail, ensure, Result};
use clap::Parser;
use std::path::{Path, PathBuf};

pub struct Config {
    pub extract_mode: bool,
    pub replace_mode: bool,
    pub input: String,
    pub output: String,
    pub model: String,
    pub padding: u16,
    pub input_mode: InputMode,
}

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[arg(
        short,
        long,
        help = "Pass '-e' or '--extract' to extract text from images."
    )]
    pub extract_mode: bool,
    #[arg(
        short,
        long,
        help = "Pass '-r' or '--replace' to replace text regions in input images from a JSON containing translated text"
    )]
    pub replace_mode: bool,
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
        help = "Model Path - A path to a detection model must be specified (ONNX format)."
    )]
    pub model: PathBuf,
    #[arg(
        short,
        long,
        help = "Specify size of padding for text regions (Tinkering may improve OCR)"
    )]
    pub padding: Option<u16>,
}
#[derive(PartialEq)]
pub enum InputMode {
    Directory,
    Image,
}

impl Config {
    pub fn parse() -> Result<Config> {
        let cli = Cli::parse();

        // If extract or replace mode is toggled, make sure only one of the two is toggled
        ensure!(
            cli.extract_mode & cli.replace_mode == false,
            "Run in either extract or replace mode but not both."
        );

        /*
            Determine the type of input:
            - Directory of Images
            - Single Image
        */
        let input_mode: InputMode = match cli.input.extension() {
            Some(os_extension) => {
                if let Some(str_extension) = os_extension.to_str() {
                    match str_extension {
                        "jpg" => InputMode::Image,
                        _ => {
                            bail!("Input must be either a directory or JPG.")
                        }
                    }
                } else {
                    bail!("Input must be either a directory or JPG.")
                }
            }
            None => {
                if cli.input.is_dir() == false {
                    bail!("Input must be either a directory or JPG.");
                }

                InputMode::Directory
            }
        };

        let output = match cli.output {
            Some(path) => {
                if path.is_dir() && input_mode == InputMode::Image {
                    bail!("Output and Input must be of the same type")
                } else if !path.is_dir() && input_mode == InputMode::Directory {
                    bail!("Output and Input must be of the same type")
                }

                path
            }
            None => match input_mode {
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
            },
        };

        let padding: u16;

        if let Some(custom_padding) = cli.padding {
            padding = custom_padding;
        } else {
            padding = 10;
        };

        let input_string = match cli.input.into_os_string().into_string().ok() {
            Some(path) => path,
            None => {
                bail!("Make sure input path name is UTF-8 compatible.")
            }
        };

        let model_string = match cli.model.into_os_string().into_string().ok() {
            Some(path) => path,
            None => {
                bail!("Make sure model path name is UTF-8 compatible.")
            }
        };

        let output_string = match output.into_os_string().into_string().ok() {
            Some(path) => path,
            None => {
                bail!("Make sure output path name is UTF-8 compatible.")
            }
        };

        Ok(Config {
            extract_mode: cli.extract_mode,
            replace_mode: cli.replace_mode,
            input: input_string,
            output: output_string,
            model: model_string,
            padding: padding,
            input_mode: input_mode,
        })
    }
}

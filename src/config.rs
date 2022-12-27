use std::Path::{Path, PathBuf};
use clap::Parser;
use anyhow::{Result, ensure, anyhow};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[arg(short, long, help = "Pass '-e' or '--extract' to extract text from images.")]
    pub extract_mode: bool,
    #[arg(short, long, help = "Pass '-r' or '--replace' to replace text regions in input images from a JSON containing translated text")]
    pub replace_mode: bool,
    #[arg(short, long, help = "Input Path - Directory of jpgs, PDF, or JPG")]
    pub input: PathBuf,
    #[arg(short, long, help = "Optional Output Path - Specify output location for text or image outputs. If not specified, application will revert to the current directory.")]
    pub output: Option<PathBuf>,
    #[arg(short, long, help = "Optional Config Path - Application will read parameters from configuration file if specified.")]
    pub config: Option<PathBuf>,
    #[arg(short, long, help = "Model Path - If no configuration file is specified, a path to a detection model must be specified (ONNX format).")]
    pub model: Option<PathBuf>,
    #[arg(short, long, help = "Specify size of padding for text regions (Tinkering may improve OCR)")]
    pub padding: u16,
}

pub enum InputMode {
    Directory,
    Pdf,
    Image,
}

impl Args {
    pub fn run() -> Result<()> {
        let cli = Cli::parse();
        
        // Determine type of input
        let input_mode: InputMode = match cli.input.extension() {
            Some(extension) => {
                match extension {
                    "pdf" => { InputMode.Pdf },
                    "jpg" => { InputMode.Image },
                    _ => {
                        anyhow!("Input must be either a directory, PDF, or JPG.")
                    }
                }
            },
            None => {
                if cli.input.is_dir() == false {
                    anyhow!("Input must be either a directory, PDF, or JPG.");
                }

                InputMode.Directory
            }
        };

        // Check to see if a model path was provided
        ensure!(cli.config_location.is_some() || cli.model_location.is_some(), anyhow!("Model location needs to be specified either through CLI or a configuration file."));

        // Check to see 
        ensure!((cli.extract_mode.xor(cli.replace_mode).is_some()), anyhow!("Run in either extract or replace mode but not both"));

        let Some(output) = cli.output_location else {
            if cli.extract_mode == true {
                Path::new("./extracted_text.txt");
            }
        };

        let Some(config) = cli.config_location else {

        };

        


        Ok(())
    }
}
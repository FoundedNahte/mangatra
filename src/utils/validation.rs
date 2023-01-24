use anyhow::{bail, Result};
use std::path::Path;

// Validate that model path is in the ONNX file format
pub fn validate_model(model: &Path) -> Result<()> {
    if let Some(extension) = model.extension() {
        match extension.to_str() {
            Some("onnx") => Ok(()),
            Some(_) => {
                bail!("Model must be an ONNX file.");
            }
            None => {
                let bad_path = model.display();
                bail!("{bad_path} needs to have a UTF-8 compatible name.");
            }
        }
    } else {
        bail!("Model must be an ONNX file.");
    }
}

// Validate that text files are JSONs
pub fn validate_text(text: &Path) -> Result<()> {
    if let Some(extension) = text.extension() {
        match extension.to_str() {
            Some("json") => Ok(()),
            Some(_) => {
                bail!("Text file must be a JSON file.");
            }
            None => {
                let bad_path = text.display();
                bail!("{bad_path} needs to have a UTF-8 compatible name.");
            }
        }
    } else {
        bail!("Text file must be a JSON file.");
    }
}

// Validate image is in one of allowed image formats
pub fn validate_image(image: &Path) -> Result<()> {
    if let Some(extension) = image.extension() {
        match extension.to_str() {
            Some("jpg" | "jpeg" | "png" | "webp") => Ok(()),
            Some(_) => {
                bail!("Image file must be in one of the specified formats: JPG, PNG, WebP.");
            }
            None => {
                let bad_path = image.display();
                bail!("{bad_path} needs to have a UTF-8 compatible name.");
            }
        }
    } else {
        bail!("Image file must be in one of the specified formats: JPG, PNG, WebP.");
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use crate::utils::validation::{validate_image, validate_model, validate_text};
    use tempfile::TempDir;

    #[test]
    fn test_model_validation() {
        let good_model_path = Path::new("./model.onnx");

        let bad_model_path = Path::new("./model.ONNX");

        let test_dir_path = TempDir::new().unwrap();

        let good_result = validate_model(good_model_path);

        let bad_err = validate_model(bad_model_path).unwrap_err();

        let dir_err = validate_model(test_dir_path.path()).unwrap_err();

        match good_result {
            Ok(_) => {}
            Err(e) => {
                panic!("{e}")
            }
        }

        assert_eq!(format!("{bad_err}"), "Model must be an ONNX file.");

        assert_eq!(format!("{dir_err}"), "Model must be an ONNX file.");
    }

    #[test]
    fn test_text_validation() {
        let good_text_path = Path::new("./text.json");

        let bad_text_path = Path::new("./text.txt");

        let test_dir_path = TempDir::new().unwrap();

        let good_result = validate_text(good_text_path);

        let bad_err = validate_text(bad_text_path).unwrap_err();

        let dir_err = validate_text(test_dir_path.path()).unwrap_err();

        match good_result {
            Ok(_) => {}
            Err(e) => {
                panic!("{e}")
            }
        }

        assert_eq!(format!("{bad_err}"), "Text file must be a JSON file.");

        assert_eq!(format!("{dir_err}"), "Text file must be a JSON file.");
    }

    #[test]
    fn test_image_validation() {
        let good_jpg_path = Path::new("./image1.jpg");
        let good_jpeg_path = Path::new("./image2.jpeg");
        let good_png_path = Path::new("./image3.png");
        let good_wepb_path = Path::new("./image4.webp");

        let result1 = validate_image(good_jpg_path);
        let result2 = validate_image(good_jpeg_path);
        let result3 = validate_image(good_png_path);
        let result4 = validate_image(good_wepb_path);

        match result1 {
            Ok(_) => {}
            Err(e) => {
                panic!("{e}")
            }
        }

        match result2 {
            Ok(_) => {}
            Err(e) => {
                panic!("{e}")
            }
        }

        match result3 {
            Ok(_) => {}
            Err(e) => {
                panic!("{e}")
            }
        }

        match result4 {
            Ok(_) => {}
            Err(e) => {
                panic!("{e}")
            }
        }

        let test_dir_path = TempDir::new().unwrap();
        let bad_image_path1 = Path::new("./bad_image1.tiff");
        let bad_image_path2 = Path::new("./image2");

        let err1 = validate_image(test_dir_path.path()).unwrap_err();
        let err2 = validate_image(bad_image_path1).unwrap_err();
        let err3 = validate_image(bad_image_path2).unwrap_err();

        assert_eq!(
            format!("{err1}"),
            "Image file must be in one of the specified formats: JPG, PNG, WebP."
        );
        assert_eq!(
            format!("{err2}"),
            "Image file must be in one of the specified formats: JPG, PNG, WebP."
        );
        assert_eq!(
            format!("{err3}"),
            "Image file must be in one of the specified formats: JPG, PNG, WebP."
        );
    }
}

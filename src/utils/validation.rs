use anyhow::{bail, ensure, Result};
use std::path::Path;

// Validate that model path is in the ONNX file format
pub fn validate_model(model: &Path) -> Result<bool> {
    if let Some(extension) = model.extension() {
        ensure!(
            (Some("onnx") == extension.to_str()),
            "Model must be an ONNX file."
        )
    } else {
        bail!("Model must be an ONNX file.")
    }

    Ok(true)
}

// Validate that text files are JSONs
pub fn validate_text(text: &Path) -> Result<bool> {
    if let Some(extension) = text.extension() {
        ensure!(
            (Some("json") == extension.to_str()),
            "Text file must be a JSON file."
        )
    } else {
        bail!("Text file must be a JSON file.")
    }

    Ok(true)
}

// Validate image is in one of allowed image formats
pub fn validate_image(image: &Path) -> bool {
    if let Some(extension) = image.extension() {
        match extension.to_str() {
            Some("jpg" | "jpeg" | "png" | "webp") => return true,
            _ => {
                eprintln!("Image file must be in one of the specified formats: JPG, PNG, WebP.");
                return false;
            }
        }
    } else {
        eprintln!("Image file must be in one of the specified formats: JPG, PNG, WebP.");
    }

    false
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

        let good_result = validate_model(&good_model_path.to_path_buf()).unwrap();

        let bad_err = validate_model(&bad_model_path.to_path_buf()).unwrap_err();

        let dir_err = validate_model(&test_dir_path.path().to_path_buf()).unwrap_err();

        assert_eq!(good_result, true);

        assert_eq!(format!("{bad_err}"), "Model must be an ONNX file.");

        assert_eq!(format!("{dir_err}"), "Model must be an ONNX file.");
    }

    #[test]
    fn test_text_validation() {
        let good_text_path = Path::new("./text.json");

        let bad_text_path = Path::new("./text.txt");

        let test_dir_path = TempDir::new().unwrap();

        let good_result = validate_text(&good_text_path.to_path_buf()).unwrap();

        let bad_err = validate_text(&bad_text_path.to_path_buf()).unwrap_err();

        let dir_err = validate_text(&test_dir_path.path().to_path_buf()).unwrap_err();

        assert_eq!(good_result, true);

        assert_eq!(format!("{bad_err}"), "Text file must be a JSON file.");

        assert_eq!(format!("{dir_err}"), "Text file must be a JSON file.");
    }

    #[test]
    fn test_image_validation() {
        let good_jpg_path = Path::new("./image1.jpg");
        let good_jpeg_path = Path::new("./image2.jpeg");
        let good_png_path = Path::new("./image3.png");
        let good_wepb_path = Path::new("./image4.webp");

        let result1 = validate_image(&good_jpg_path.to_path_buf());
        let result2 = validate_image(&good_jpeg_path.to_path_buf());
        let result3 = validate_image(&good_png_path.to_path_buf());
        let result4 = validate_image(&good_wepb_path.to_path_buf());

        assert_eq!(result1, true);
        assert_eq!(result2, true);
        assert_eq!(result3, true);
        assert_eq!(result4, true);

        let test_dir_path = TempDir::new().unwrap();
        let bad_image_path1 = Path::new("./bad_image1.tiff");
        let bad_image_path2 = Path::new("./image2");

        let err1 = validate_image(&test_dir_path.path().to_path_buf());
        let err2 = validate_image(&bad_image_path1.to_path_buf());
        let err3 = validate_image(&bad_image_path2.to_path_buf());

        assert_eq!(err1, false);
        assert_eq!(err2, false);
        assert_eq!(err3, false);
    }
}

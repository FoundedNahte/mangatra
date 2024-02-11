use anyhow::{anyhow, Result};
use opencv::core::{self as cv, Mat, Rect, Vector};

use crate::detection::Detector;
use crate::ocr::Ocr;
use crate::replacer::Replacer;
use crate::utils::image_conversion::{image_buffer_to_mat, mat_to_image_buffer};

type ExtractedText = Vec<String>;
type ImageRegions = Vector<Mat>;
type Origins = Vec<(i32, i32)>;

pub trait BoundingBox {
    fn x(&self) -> i32;
    fn y(&self) -> i32;
    fn width(&self) -> Result<u32>;
    fn height(&self) -> Result<u32>;
}

pub trait MangatraDetection {
    type B: BoundingBox;

    fn text(&self) -> &String;
    fn bounding_box(&self) -> Option<&Self::B>;
}

pub fn clean_image(
    input_image_bytes: &[u8],
    padding: Option<u16>,
    model_path: &str,
) -> Result<Vec<u8>> {
    let image = image::load_from_memory(input_image_bytes)?;
    let mut detector = Detector::new(model_path, padding)?;

    let (text_regions, origins) = detector.run_inference(&image)?;

    let replacer: Replacer<String> = Replacer::new(
        text_regions,
        None,
        origins,
        image_buffer_to_mat(image.to_rgb8())?,
        padding,
    )?;
    let cleaned_page = replacer.clean_page()?;

    Ok(mat_to_image_buffer(&cleaned_page)?.to_vec())
}

pub fn extract_text(
    input_image_bytes: &[u8],
    padding: Option<u16>,
    model_path: &str,
    tessdata_path: &str,
    lang: &str,
) -> Result<(ExtractedText, ImageRegions, Origins)> {
    let image = image::load_from_memory(input_image_bytes)?;
    let mut detector = Detector::new(model_path, padding)?;
    let mut ocr = Ocr::new(lang, tessdata_path)?;

    let (text_regions, origins) = detector.run_inference(&image)?;

    let extracted_text = ocr.extract_text(&text_regions)?;

    Ok((extracted_text, text_regions, origins))
}

pub fn replace_image<T: MangatraDetection>(
    input_image_bytes: &[u8],
    padding: Option<u16>,
    detections: &[T],
) -> Result<Vec<u8>> {
    let image = image::load_from_memory(input_image_bytes)?;
    let image_mat = image_buffer_to_mat(image.to_rgb8())?;
    let mut text: Vec<String> = Vec::new();
    let mut regions: Vector<Mat> = Vector::new();
    let mut origins: Vec<(i32, i32)> = Vec::new();

    for detection in detections.iter() {
        let detection_text = &detection.text();
        let detection_box = detection.bounding_box().ok_or(anyhow!(format!(
            "Detection box missing with {detection_text}"
        )))?;
        let width: i32 = detection_box.width()?.try_into()?;
        let height: i32 = detection_box.height()?.try_into()?;
        text.push(detection_text.to_string());

        let text_region = cv::Mat::roi(
            &image_mat,
            Rect {
                x: detection_box.x(),
                y: detection_box.y(),
                width,
                height,
            },
        )?;

        let origin = (detection_box.x(), detection_box.y());
        origins.push(origin);
        regions.push(text_region);
    }

    let replacer = Replacer::new(regions, Some(text), origins, image_mat, padding)?;

    let final_image = replacer.replace_text_regions()?;

    Ok(mat_to_image_buffer(&final_image)?.to_vec())
}

pub fn detect_boxes(
    input_image_bytes: &[u8],
    padding: Option<u16>,
    model_path: &str,
) -> Result<(ImageRegions, Origins)> {
    let image = image::load_from_memory(input_image_bytes)?;
    let mut detector = Detector::new(model_path, padding)?;

    let (text_regions, origins) = detector.run_inference(&image)?;

    Ok((text_regions, origins))
}

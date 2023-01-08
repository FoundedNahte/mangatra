use anyhow::Result;
use ndarray::{self as nd, Axis};
use opencv::{self as cv, core::Rect2i, core::ToInputArray, dnn, imgcodecs, prelude::*};
use std::cmp::max;

type Origin = (i32, i32);
type TextRegions = cv::core::Vector<cv::core::Mat>;

struct Detections {
    pub boxes: cv::core::Vector<cv::core::Rect2i>,
}

pub struct Detector {
    model: dnn::Net,
    padding: u16,
}

impl Detector {
    pub fn new(model_path: &str, padding: u16) -> Result<Detector> {
        let model = dnn::read_net(model_path, "", "")?;

        Ok(Detector { model, padding })
    }

    // Main detection function to extract text regions from image
    pub fn run_inference(&mut self, input_image: &str) -> Result<(TextRegions, Vec<Origin>)> {
        let input: cv::core::Mat = Self::format_image(input_image)?;

        let result: cv::core::Mat = dnn::blob_from_image(
            &input.input_array()?,
            1.0 / 255.0,
            cv::core::Size2i::new(640, 640),
            cv::core::Scalar::new(1.0, 1.0, 1.0, 1.0),
            true,
            false,
            cv::core::CV_32F,
        )?;

        self.model
            .set_input(&result, "", 1.0, cv::core::Scalar::new(1.0, 1.0, 1.0, 1.0))?;

        let mut predictions: cv::core::Vector<cv::core::Mat> = cv::core::Vector::new();

        println!("RUNNING INFERENCE");

        self.model.forward(
            &mut predictions,
            &self.model.get_unconnected_out_layers_names()?,
        )?;

        println!("INFERENCE SUCCESSFUL");

        let data = predictions.get(0)?;

        let output = nd::ArrayView3::from_shape((1, 25200, 10), data.data_typed::<f32>()?)?;

        let detections = Self::get_detections(input, output.index_axis(Axis(0), 0))?;

        let boxes = detections.boxes;

        let original_image: cv::core::Mat = imgcodecs::imread("img.jpg", imgcodecs::IMREAD_COLOR)?;
        /*
            for i in 0..boxes.len() {
                let classid = class_ids[i];
                let confidence = confidences[i];
                let bbox = boxes.get(i)?;

                cv::imgproc::rectangle(&mut original_image, bbox, cv::core::Scalar::from((255.0, 255.0, 0.0)), 2, cv::imgproc::LINE_8, 0)?;
            }

            highgui::imshow("boxes", &original_image)?;
            highgui::wait_key(2000)?;
            highgui::destroy_all_windows()?;
        */
        let mut text_regions: cv::core::Vector<cv::core::Mat> = cv::core::Vector::new();
        let mut origins: Vec<(i32, i32)> = Vec::new();

        for bbox in boxes {
            let padded_bbox: Rect2i = Rect2i::new(
                bbox.x,
                bbox.y,
                bbox.width + self.padding as i32,
                bbox.height + self.padding as i32,
            );

            text_regions.push(cv::core::Mat::roi(&original_image, padded_bbox)?);
            origins.push((bbox.x, bbox.y));
        }

        Ok((text_regions, origins))
    }

    // Helper function that pre-processes input image for the YoloV5 model
    fn format_image(input_image: &str) -> Result<cv::core::Mat> {
        let image: cv::core::Mat = imgcodecs::imread(input_image, imgcodecs::IMREAD_COLOR)?;

        let cols: i32 = image.cols();
        let rows: i32 = image.rows();

        let max = max(cols, rows);

        let padding: cv::core::Mat;

        let mut resized: cv::core::Mat =
            cv::core::Mat::zeros(max, max, cv::core::CV_32F)?.to_mat()?;

        if max == rows && max != cols {
            padding = cv::core::Mat::zeros(rows, rows - cols, cv::core::CV_8UC3)?.to_mat()?;
            cv::core::hconcat2(&image, &padding, &mut resized)?;
        } else if max == cols && max != rows {
            padding = cv::core::Mat::zeros(cols - rows, cols, cv::core::CV_8UC3)?.to_mat()?;
            cv::core::vconcat2(&image, &padding, &mut resized)?;
        } else {
            resized = cv::core::Mat::copy(&image)?;
        }
        /*
        highgui::imshow("resized", &resized)?;
        highgui::wait_key(2000)?;
        highgui::destroy_all_windows()?;
        */
        Ok(resized)
    }

    // Function to get text regions from model output
    fn get_detections(
        image: cv::core::Mat,
        output_data: nd::ArrayView2<f32>,
    ) -> Result<Detections> {
        let mut class_ids: Vec<i32> = Vec::new();
        let mut confidences: Vec<f32> = Vec::new();
        let mut boxes: cv::core::Vector<Rect2i> = cv::core::Vector::new();

        let img_height = image.rows();
        let img_width = image.cols();

        let x_factor: f32 = img_width as f32 / 640.0;
        let y_factor: f32 = img_height as f32 / 640.0;

        for i in 0..25200 {
            let row = output_data.index_axis(Axis(0), i);
            let confidence = row[[4]];

            if confidence >= 0.4 {
                let classes_scores = row.to_vec();

                let mut max_indx: cv::core::Point2i = cv::core::Point2i::new(0, 0);

                cv::core::min_max_loc(
                    &Self::convert_to_cv_f32vec(&classes_scores),
                    None,
                    None,
                    None,
                    Some(&mut max_indx),
                    &cv::core::no_array(),
                )?;

                let class_id = max_indx.to_vec2()[1];

                if classes_scores[class_id as usize] > 0.25 {
                    confidences.push(confidence);

                    class_ids.push(class_id);

                    let x: f32 = row[[0]];
                    let y: f32 = row[[1]];
                    let w: f32 = row[[2]];
                    let h: f32 = row[[3]];

                    let left: i32 = ((x - 0.5 * w) * x_factor) as i32;
                    let top: i32 = ((y - 0.5 * h) * y_factor) as i32;
                    let width: i32 = (w * x_factor) as i32;
                    let height: i32 = (h * y_factor) as i32;

                    boxes.push(cv::core::Rect2i::new(left, top, width, height));
                }
            }
        }

        let mut indices: cv::core::Vector<i32> = cv::core::Vector::new();

        dnn::nms_boxes(
            &boxes,
            &Self::convert_to_cv_f32vec(&confidences),
            0.25,
            0.45,
            &mut indices,
            1.0,
            0,
        )?;

        let mut result_class_ids: Vec<i32> = Vec::new();
        let mut result_confidences: Vec<f32> = Vec::new();
        let mut result_boxes: cv::core::Vector<Rect2i> = cv::core::Vector::new();

        for i in indices {
            result_confidences.push(confidences[i as usize]);
            result_class_ids.push(class_ids[i as usize]);
            result_boxes.push(boxes.get(i as usize)?);
        }

        let detections = Detections {
            boxes: result_boxes,
        };

        Ok(detections)
    }

    fn convert_to_cv_f32vec(input: &Vec<f32>) -> cv::core::Vector<f32> {
        let mut result: cv::core::Vector<f32> = cv::core::Vector::new();

        for value in input {
            result.push(*value);
        }

        result
    }
}

use anyhow::Result;
use image::{self, ImageBuffer, Rgb};
use opencv::{self as cv, core, prelude::*};
use std::slice;

// Create a white rectangle in the same dimensions as the input Mat (Used for create writing canvas in replacement)
pub fn get_blank_buffer(image: &core::Mat) -> Result<ImageBuffer<Rgb<u8>, Vec<u8>>> {
    let width: u32 = image.cols() as u32;
    let height: u32 = image.rows() as u32;

    let converted_image_buffer = ImageBuffer::from_pixel(width, height, Rgb::from([255, 255, 255]));

    Ok(converted_image_buffer)
}

// Convert CV mats back into image buffers
// Credit to https://github.com/jerry73204/rust-cv-convert
pub fn mat_to_image_buffer(image: &core::Mat) -> Result<ImageBuffer<Rgb<u8>, Vec<u8>>> {
    let width: u32 = image.cols() as u32;
    let height: u32 = image.rows() as u32;

    let shape: Vec<usize> = image
        .mat_size()
        .iter()
        .map(|&dim| dim as usize)
        .chain([image.channels() as usize])
        .collect();

    let numel = shape.iter().product();
    let ptr = image.ptr(0)?;

    let slice = unsafe { slice::from_raw_parts(ptr, numel) };

    Ok(ImageBuffer::from_vec(width, height, slice.to_vec())
        .expect("Could not convert to image buffer"))
}

// Helper function to convert image buffers to OpenCV Mats
// Credit to https://github.com/jerry73204/rust-cv-convert
pub fn image_buffer_to_mat(image: ImageBuffer<Rgb<u8>, Vec<u8>>) -> Result<core::Mat> {
    let (width, height) = image.dimensions();
    let cv_type = cv::core::CV_MAKETYPE(8, 3);

    let mat = unsafe {
        cv::core::Mat::new_rows_cols_with_data(
            height as i32,
            width as i32,
            cv_type,
            image.as_ptr() as *mut _,
            cv::core::Mat_AUTO_STEP,
        )?
        .try_clone()?
    };

    Ok(mat)
}

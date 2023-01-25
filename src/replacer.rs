use crate::utils::image_conversion;
use anyhow::Result;
use image::{self, ImageBuffer, Rgb};
use imageproc::drawing;
use opencv::{core, prelude::*};
use rusttype::{Font, Scale};

pub struct Replacer {
    text_regions: core::Vector<core::Mat>,
    translated_text: Vec<String>,
    origins: Vec<(i32, i32)>,
    original_image: core::Mat,
    padding: u16,
}

impl Replacer {
    pub fn new(
        text_regions: core::Vector<core::Mat>,
        translated_text: Vec<String>,
        origins: Vec<(i32, i32)>,
        original_image: core::Mat,
        padding: u16,
    ) -> Result<Replacer> {
        Ok(Replacer {
            text_regions,
            translated_text,
            origins,
            original_image,
            padding,
        })
    }

    pub fn replace_text_regions(&self) -> Result<core::Mat> {
        let text_regions = self.write_text()?;

        let full_width = self.original_image.cols();
        let full_height = self.original_image.rows();

        let mut temp_image = core::Mat::copy(&self.original_image)?;

        for i in 0..text_regions.len() {
            let (text_region_x, text_region_y) = self.origins[i];
            let text_region = text_regions.get(i)?;

            let text_region_width = text_region.cols();
            let text_region_height = text_region.rows();

            let (left_panel_x, left_panel_y) = (0, 0);
            let (top_panel_x, top_panel_y) = (text_region_x, 0);
            let (bottom_panel_x, bottom_panel_y) =
                (text_region_x, text_region_y + text_region_height);
            let (right_panel_x, right_panel_y) = (text_region_x + text_region_width, 0);

            let (left_panel_width, left_panel_height) = (text_region_x, full_height);
            let (top_panel_width, top_panel_height) = (text_region_width, text_region_y);
            let (bottom_panel_width, bottom_panel_height) = (
                text_region_width,
                full_height - (text_region_y + text_region_height),
            );
            let (right_panel_width, right_panel_height) = (
                full_width - (text_region_x + text_region_width),
                full_height,
            );

            let left_panel = core::Mat::roi(
                &temp_image,
                core::Rect2i::new(
                    left_panel_x,
                    left_panel_y,
                    left_panel_width,
                    left_panel_height,
                ),
            )?;

            let top_panel = core::Mat::roi(
                &temp_image,
                core::Rect2i::new(top_panel_x, top_panel_y, top_panel_width, top_panel_height),
            )?;

            let bottom_panel = core::Mat::roi(
                &temp_image,
                core::Rect2i::new(
                    bottom_panel_x,
                    bottom_panel_y,
                    bottom_panel_width,
                    bottom_panel_height,
                ),
            )?;

            let right_panel = core::Mat::roi(
                &temp_image,
                core::Rect2i::new(
                    right_panel_x,
                    right_panel_y,
                    right_panel_width,
                    right_panel_height,
                ),
            )?;

            let mut vertical_panels_vec: core::Vector<core::Mat> = core::Vector::new();

            let mut vertical_panel: core::Mat = core::Mat::default();

            vertical_panels_vec.push(top_panel);
            vertical_panels_vec.push(text_region);
            vertical_panels_vec.push(bottom_panel);

            core::vconcat(&vertical_panels_vec, &mut vertical_panel)?;
            /*
                    highgui::imshow("resized", &vertical_panel)?;
                    highgui::wait_key(2000)?;
                    highgui::destroy_all_windows()?;
            */
            let mut horizontal_panels_vec: core::Vector<core::Mat> = core::Vector::new();

            let mut result: core::Mat = core::Mat::default();

            horizontal_panels_vec.push(left_panel);
            horizontal_panels_vec.push(vertical_panel);
            horizontal_panels_vec.push(right_panel);

            core::hconcat(&horizontal_panels_vec, &mut result)?;
            /*
                    highgui::imshow("resized", &result)?;
                    highgui::wait_key(2000)?;
                    highgui::destroy_all_windows()?;
            */
            temp_image = result;
        }

        Ok(temp_image)
    }

    // Write replace japanese text with english text
    fn write_text(&self) -> Result<core::Vector<core::Mat>> {
        let mut canvases: Vec<ImageBuffer<Rgb<u8>, Vec<u8>>> = Vec::new();

        /*
            We iterate through the different each text region and draw its respective translation
            onto a blank, white canvas.
        */
        for i in 0..self.translated_text.len() {
            let region = self.text_regions.get(i)?;
            let text = self.translated_text[i].clone();

            // Get blank, white canvas to draw translated text on
            let mut canvas = image_conversion::mat_to_image_buffer(region)?;

            let (width, height) = canvas.dimensions();

            let height = height as i32;

            let start_x = width / 16;
            let mut start_y = height / 20;

            let stop_x = width - (width / 16);

            let font = Vec::from(include_bytes!("../assets/mangat.ttf") as &[u8]);
            let font = Font::try_from_vec(font).expect("Could not unwrap Font.");

            let mut curr_line_size = 0;

            let split_text = text.split(' ');

            let mut temp_lines: Vec<String> = Vec::new();

            let num_words = split_text
                .clone()
                .map(str::to_string)
                .collect::<Vec<String>>()
                .len();

            /*
                Scaling rules based on width of the region
                and number of words.
            */
            let mut scale = Scale {
                x: height as f32 / 9.0,
                y: height as f32 / 12.0,
            };

            if width < 55 {
                scale.x = height as f32 / 8.0;
                scale.y = height as f32 / 12.0;
            } else if width < 100 {
                scale.x = height as f32 / 10.0;
                scale.y = height as f32 / 14.0;
            }
            /*
            if num_words >= 17 {
                scale.x = height as f32 / 20.0;
                scale.y = height as f32 / 23.0;
            } 
            */
            /*
            if num_words >= 15 {
                scale.x = height as f32 / 18.0;
                scale.y = height as f32 / 21.0;
            } else 
            */
            if num_words >= 16 {
                scale.x = height as f32 / 14.0;
                scale.y = height as f32 / 16.0;
            } else if num_words >= 14 {
                scale.x = height as f32 / 12.0;
                scale.y = height as f32 / 14.0;
            } else if num_words >= 12 {
                scale.x = height as f32 / 10.0;
                scale.y = height as f32 / 12.0;
            } else if num_words >= 10 {
                scale.x = height as f32 / 8.0;
                scale.y = height as f32 / 10.0;
            } else if num_words <= 2 {
                scale.x = height as f32 / 7.0;
                scale.y = height as f32 / 9.0;
            }

            let mut curr_line = String::new();

            let (width_of_space, text_height) = drawing::text_size(scale, &font, " ");

            for word in split_text {
                let (text_width, _) = drawing::text_size(scale, &font, word);

                if curr_line_size + text_width + width_of_space
                    > stop_x as i32 - self.padding as i32
                {
                    temp_lines.push(curr_line);
                    curr_line = String::from(word);
                    curr_line_size = text_width;
                } else if temp_lines.is_empty() && curr_line.is_empty() {
                    curr_line.push_str(word);
                } else {
                    curr_line.push(' ');
                    curr_line.push_str(word);
                    curr_line_size += width_of_space;
                    curr_line_size += text_width;
                }
            }

            println!("lines: {temp_lines:?}");

            temp_lines.push(curr_line);

            let mut lines: Vec<String> = Vec::new();

            for line in temp_lines {
                let (text_width, _) = drawing::text_size(scale, &font, &line);

                if text_width > stop_x as i32 - self.padding as i32 {
                    let num_words = line
                        .split(' ')
                        .map(str::to_string)
                        .collect::<Vec<String>>()
                        .len();

                    /*
                        If the line is a single word and it's still too long,
                        we make a new line at the closest char to the border.
                        If there are multiple words in the line, we find the
                        closest word to the border and make a newline there.
                    */
                    if num_words == 1 {
                        let mut chars: Vec<char> = line.chars().collect();
                        let mut original_line: String = chars.iter().collect();
                        let mut new_line: Vec<char> = Vec::new();

                        while drawing::text_size(scale, &font, &original_line).0
                            > stop_x as i32 - self.padding as i32
                        {
                            // We move the last char from the original line to the beginning of the new line
                            new_line.insert(
                                0,
                                chars
                                    .pop()
                                    .expect("Unexpected error while popping from char vector."),
                            );
                            // Rebuild the updated original line for checking.
                            original_line = chars.iter().collect();
                        }

                        // Push the updated original line
                        lines.push(original_line);

                        // Push the new line
                        if !new_line.is_empty() {
                            let new_line = new_line.iter().collect();

                            lines.push(new_line);
                        }
                    } else {
                        let mut words: Vec<String> = line.split(' ').map(str::to_string).collect();

                        let mut original_line = words.join(" ");
                        let mut new_line: Vec<String> = Vec::new();

                        while drawing::text_size(scale, &font, &original_line).0
                            > stop_x as i32 - self.padding as i32
                        {
                            new_line.insert(
                                0,
                                words
                                    .pop()
                                    .expect("Unexpected error while popping from word vector."),
                            );

                            original_line = words.join(" ");
                        }

                        // Push the updated original line
                        lines.push(original_line);

                        // Push the new line
                        if !new_line.is_empty() {
                            lines.push(new_line.join(" "));
                        }
                    }
                } else {
                    // If the line is fine, append it and continue
                    if !line.is_empty() {
                        lines.push(line.to_string());
                    }
                }
            }

            for line in lines {
                drawing::draw_text_mut(
                    &mut canvas,
                    Rgb([0u8, 0u8, 0u8]),
                    start_x as i32,
                    start_y,
                    scale,
                    &font,
                    &line,
                );

                // 15 is an arbitrary number used to account for space between lines
                start_y += text_height + 15;
            }

            canvases.push(canvas);
        }

        let mut cv_vector: core::Vector<core::Mat> = core::Vector::new();

        for canvas in canvases {
            cv_vector.push(image_conversion::image_buffer_to_mat(canvas)?);
        }

        Ok(cv_vector)
    }
}

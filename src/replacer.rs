use crate::utils::image_conversion;
use anyhow::Result;
use image::{self, ImageBuffer, Rgb};
use imageproc::drawing;
use opencv::{core, prelude::*};
use rusttype::{Font, Scale};

type Coordinates = (i32, i32);
type Width = i32;
type Height = i32;

pub struct Replacer<'a> {
    original_text_regions: core::Vector<core::Mat>,
    translated_text: &'a Vec<String>,
    origins: Vec<(i32, i32)>,
    original_image: core::Mat,
    padding: u16,
}

impl<'a> Replacer<'a> {
    pub fn new(
        original_text_regions: core::Vector<core::Mat>,
        translated_text: &Vec<String>,
        origins: Vec<(i32, i32)>,
        original_image: core::Mat,
        padding: u16,
    ) -> Result<Replacer> {
        Ok(Replacer {
            original_text_regions,
            translated_text,
            origins,
            original_image,
            padding,
        })
    }

    pub fn replace_text_regions(&self) -> Result<core::Mat> {
        let (new_text_regions, new_origins, origin_flags) = self.write_text()?;

        let full_width = self.original_image.cols();
        let full_height = self.original_image.rows();

        let mut temp_image = core::Mat::copy(&self.original_image)?;

        for (i, (x, y)) in new_origins.iter().enumerate().take(new_text_regions.len()) {
            let (x, y) = (*x, *y);
            let text_region = new_text_regions.get(i)?;

            let width = text_region.cols();
            let height = text_region.rows();

            // Establish origins for the four panels
            let (left_panel_x, left_panel_y) = (0, 0);
            let (top_panel_x, top_panel_y) = (x, 0);
            let (bottom_panel_x, bottom_panel_y) = (x, y + height);
            let (right_panel_x, right_panel_y) = (x + width, 0);

            // Establish dimensions for the four panels
            let (left_panel_width, left_panel_height) = (x, full_height);
            let (top_panel_width, top_panel_height) = (width, y);
            let (bottom_panel_width, bottom_panel_height) = (width, full_height - (y + height));
            let (right_panel_width, right_panel_height) = (full_width - (x + width), full_height);

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

            #[cfg(feature = "debug")]
            {
                use imageproc::rect::Rect;

                let tl_br_flag = origin_flags.get(i).unwrap();
                let mut temp_image_buffer = image_conversion::mat_to_image_buffer(&temp_image)?;

                drawing::draw_hollow_rect_mut(
                    &mut temp_image_buffer,
                    Rect::at(x, y).of_size(width as u32, height as u32),
                    Rgb([0, 255, 0]),
                );

                if *tl_br_flag {
                    drawing::draw_filled_circle_mut(
                        &mut temp_image_buffer,
                        (x, y),
                        5,
                        Rgb([255, 0, 0]),
                    );

                    drawing::draw_filled_circle_mut(
                        &mut temp_image_buffer,
                        (x + width as i32, y + height as i32),
                        5,
                        Rgb([255, 0, 0]),
                    );
                } else {
                    drawing::draw_filled_circle_mut(
                        &mut temp_image_buffer,
                        (x + width as i32, y),
                        5,
                        Rgb([255, 0, 0]),
                    );

                    drawing::draw_filled_circle_mut(
                        &mut temp_image_buffer,
                        (x, y + height as i32),
                        5,
                        Rgb([255, 0, 0]),
                    );
                }

                temp_image = image_conversion::image_buffer_to_mat(temp_image_buffer)?;
            }
        }

        Ok(temp_image)
    }

    // Expand text region to fit text bubble
    // Returns new (x, y) coordinates and width/height
    fn expand_text_region(
        &self,
        (tl_x, tl_y): Coordinates,
        old_width: Width,
        old_height: Height,
        original: &core::Mat,
    ) -> Result<(Coordinates, Width, Height, bool)> {
        let (mut tl_x, mut tl_y) = (tl_x as u32, tl_y as u32);
        let old_width = old_width as u32;
        let old_height = old_height as u32;

        let image_buffer = image_conversion::mat_to_image_buffer(original)?;
        let (mut tr_x, mut tr_y) = (tl_x + old_width, tl_y);
        let (mut bl_x, mut bl_y) = (tl_x, tl_y + old_height);
        let (mut br_x, mut br_y) = (bl_x + old_width, bl_y);

        let mut tl_length = 0;
        let mut tr_length = 0;
        let mut bl_length = 0;
        let mut br_length = 0;

        // Expand the top left corner
        let ori_pixel = image_buffer.get_pixel(tl_x, tl_y);
        while tl_x - 1 > 0 && tl_y - 1 > 0 {
            if let Some(pixel) = image_buffer.get_pixel_checked(tl_x - 1, tl_y - 1) {
                if pixel != ori_pixel {
                    break;
                }

                tl_x -= 1;
                tl_y -= 1;
                tl_length += 1;
            } else {
                break;
            }
        }

        // Expand the top right corner
        while tr_x < old_width && tr_y > 0 {
            if let Some(pixel) = image_buffer.get_pixel_checked(tr_x + 1, tr_y - 1) {
                if pixel != ori_pixel {
                    break;
                }

                tr_x += 1;
                tr_y -= 1;
                tr_length += 1;
            } else {
                break;
            }
        }

        // Expand the bottom left corner
        while bl_x > 0 && bl_y + 1 < old_height {
            if let Some(pixel) = image_buffer.get_pixel_checked(bl_x - 1, bl_y + 1) {
                if pixel != ori_pixel {
                    break;
                }

                bl_x -= 1;
                bl_y += 1;
                bl_length += 1;
            } else {
                break;
            }
        }

        // Expand the bottom right corner
        while br_x + 1 < old_width && br_y + 1 < old_height {
            if let Some(pixel) = image_buffer.get_pixel_checked(br_x + 1, br_y + 1) {
                if pixel != ori_pixel {
                    break;
                }

                br_x += 1;
                br_y += 1;
                br_length += 1;
            } else {
                break;
            }
        }

        let new_width;
        let new_height;
        let mut tl_br_flag = false;

        // Determine which pair of opposite corners is smaller
        if (tl_length + br_length) < (bl_length + tr_length) {
            new_width = br_x - tl_x;
            new_height = br_y - tl_y;
            tl_br_flag = true;
        } else {
            new_width = tr_x - bl_x;
            new_height = bl_y - tr_y;

            tl_x = tr_x - new_width;
            tl_y = tr_y;
        }

        Ok((
            (tl_x as i32, tl_y as i32),
            new_width as i32,
            new_height as i32,
            tl_br_flag,
        ))
    }

    // Write replace japanese text with english text
    fn write_text(&self) -> Result<(core::Vector<core::Mat>, Vec<Coordinates>, Vec<bool>)> {
        let mut canvases: Vec<ImageBuffer<Rgb<u8>, Vec<u8>>> = Vec::new();
        let mut new_origins: Vec<Coordinates> = Vec::new();
        let mut origin_flags: Vec<bool> = Vec::new();

        /*
            We iterate through the different each text region and draw its respective translation
            onto a blank, white canvas.
        */
        for i in 0..self.translated_text.len() {
            let (x, y) = self.origins[i];
            let region = self.original_text_regions.get(i)?;
            let text = self.translated_text[i].clone();

            let width = region.cols();
            let height = region.rows();

            let ((x, y), width, height, tl_br_flag) =
                self.expand_text_region((x, y), width, height, &self.original_image)?;

            let region =
                core::Mat::roi(&self.original_image, core::Rect2i::new(x, y, width, height))?;

            new_origins.push((x, y));

            // Get blank, white canvas to draw translated text on
            let mut canvas = image_conversion::get_blank_buffer(&region)?;
            let (width, height) = canvas.dimensions();
            let height = height as i32;

            let stop_x = width - (width / 16);

            // Load manga font from assets
            let font = Vec::from(include_bytes!("../assets/wildwordsroman.ttf") as &[u8]);
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

            let width_of_space = drawing::text_size(scale, &font, " ").0;

            // Initially break the text segment into lines that fit within the region
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

            #[cfg(feature = "debug")]
            {
                println!("lines: {temp_lines:?}");
            }

            temp_lines.push(curr_line);

            let mut lines: Vec<String> = Vec::new();

            /*
                Since we sometimes have long words, some lines may still not fit within the region.
                Now we break up individual words if they are causing their lines to be too long.
            */
            for line in temp_lines {
                let (text_width, _) = drawing::text_size(scale, &font, &line);

                // Check if a line is still too long
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

                        let hypen_width = drawing::text_size(scale, &font, "-").0;

                        while drawing::text_size(scale, &font, &original_line).0 + hypen_width
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
                        original_line.push('-');
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

            // Center the text
            let num_lines = lines.len() as i32;
            if num_lines != 0 {
                let first_line_height = drawing::text_size(scale, &font, &lines[0]).1;
                let mut start_y = (height - (num_lines * first_line_height)) / 2;

                for line in lines {
                    let (line_width, line_height) = drawing::text_size(scale, &font, &line);
                    let start_x = (width as i32 - line_width) / 2;
                    drawing::draw_text_mut(
                        &mut canvas,
                        Rgb([0u8, 0u8, 0u8]),
                        start_x,
                        start_y,
                        scale,
                        &font,
                        &line,
                    );

                    start_y += line_height;
                }
            }

            origin_flags.push(tl_br_flag);

            canvases.push(canvas);
        }

        let mut cv_vector: core::Vector<core::Mat> = core::Vector::new();

        for canvas in canvases {
            cv_vector.push(image_conversion::image_buffer_to_mat(canvas)?);
        }

        Ok((cv_vector, new_origins, origin_flags))
    }
}

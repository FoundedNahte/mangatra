use anyhow::{anyhow, Result};
use image::{self, Rgb};
use imageproc::drawing;
use itertools::izip;
use opencv::{core, prelude::*};
use rusttype::{Font, Scale};

use crate::utils::image_conversion;
use crate::DEFAULT_PADDING;

type Coordinates = (i32, i32);
type Width = i32;
type Height = i32;

enum DiagOrientation {
    TopLeftBottomRight,
    TopRightBottomLeft,
}

struct ReplacementMat {
    pub mat: core::Mat,
    pub origin: Coordinates,
    pub diag: DiagOrientation,
}

pub struct Replacer<T>
where
    T: AsRef<str>,
{
    original_text_regions: core::Vector<core::Mat>,
    text: Option<Vec<T>>,
    origins: Vec<(i32, i32)>,
    original_image: core::Mat,
    padding: u16,
}

impl<T> Replacer<T>
where
    T: AsRef<str>,
{
    pub fn new(
        original_text_regions: core::Vector<core::Mat>,
        text: Option<Vec<T>>,
        origins: Vec<(i32, i32)>,
        original_image: core::Mat,
        padding: Option<u16>,
    ) -> Result<Replacer<T>> {
        Ok(Replacer {
            original_text_regions,
            text,
            origins,
            original_image,
            padding: padding.unwrap_or(DEFAULT_PADDING),
        })
    }

    pub fn clean_page(&self) -> Result<core::Mat> {
        let mut temp_image = core::Mat::copy(&self.original_image)?;
        let blank_mats = self.get_blank_mats()?;

        for ReplacementMat {
            mat: region,
            origin: (x, y),
            diag: diag_orientation,
        } in blank_mats
        {
            temp_image = replace_region(&temp_image, region, (x, y), diag_orientation)?;
        }

        Ok(temp_image)
    }

    pub fn replace_text_regions(&self) -> Result<core::Mat> {
        let translated_mats = self.write_text()?;
        let mut temp_image = core::Mat::copy(&self.original_image)?;

        for ReplacementMat {
            mat: text_region,
            origin: (x, y),
            diag: diag_orientation,
        } in translated_mats
        {
            temp_image = replace_region(&temp_image, text_region, (x, y), diag_orientation)?;
        }

        Ok(temp_image)
    }

    fn get_blank_mats(&self) -> Result<Vec<ReplacementMat>> {
        let mut blank_mats: Vec<ReplacementMat> = Vec::new();

        for ((x, y), region) in izip!(&self.origins, &self.original_text_regions) {
            let width = region.cols();
            let height = region.rows();

            let ((x, y), _width, _height, diag_orientation) =
                expand_text_region((*x, *y), width, height, &self.original_image)?;

            let blank_mat = image_conversion::image_buffer_to_mat(
                image_conversion::get_blank_buffer(&region)?,
            )?;
            blank_mats.push(ReplacementMat {
                mat: blank_mat,
                origin: (x, y),
                diag: diag_orientation,
            });
        }

        Ok(blank_mats)
    }

    /**
     * Takes the stored translated text and writes them onto blank (white) Mats
     */
    fn write_text(&self) -> Result<Vec<ReplacementMat>> {
        let mut translated_mats: Vec<ReplacementMat> = Vec::new();

        let Some(translated_text) = &self.text else {
            return Err(anyhow!("Translated text is missing"));
        };

        /*
            We iterate through the different each text region and draw its respective translation
            onto a blank, white canvas.
        */
        for (i, text) in translated_text.iter().enumerate() {
            let (x, y) = self.origins[i];
            let region = self.original_text_regions.get(i)?;

            let width = region.cols();
            let height = region.rows();

            let ((x, y), width, height, diag_orientation) =
                expand_text_region((x, y), width, height, &self.original_image)?;

            let region =
                core::Mat::roi(&self.original_image, core::Rect2i::new(x, y, width, height))?;

            // Get blank, white canvas to draw translated text on
            let mut canvas = image_conversion::get_blank_buffer(&region)?;
            let (width, height) = canvas.dimensions();
            let height = height as i32;

            let stop_x = width - (width / 16);

            // Load manga font from assets
            let font = Vec::from(include_bytes!("../assets/wildwordsroman.ttf") as &[u8]);
            let font = Font::try_from_vec(font).expect("Could not unwrap Font.");

            let mut curr_line_size = 0;

            let split_text = text.as_ref().split(' ');

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

            translated_mats.push(ReplacementMat {
                mat: image_conversion::image_buffer_to_mat(canvas)?,
                origin: (x, y),
                diag: diag_orientation,
            });
        }

        Ok(translated_mats)
    }
}

/**
 * Expands a text region to fit a text bubble
 *
 * * Returns new (x, y) coordinates for the region origin, width/height, and which diagonal was chosen to expand to
 */
fn expand_text_region(
    (tl_x, tl_y): Coordinates,
    old_width: Width,
    old_height: Height,
    original: &core::Mat,
) -> Result<(Coordinates, Width, Height, DiagOrientation)> {
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
    let mut diag_orientation = DiagOrientation::TopLeftBottomRight;

    // Determine which pair of opposite corners is smaller
    if (tl_length + br_length) < (bl_length + tr_length) {
        new_width = br_x - tl_x;
        new_height = br_y - tl_y;
    } else {
        new_width = tr_x - bl_x;
        new_height = bl_y - tr_y;

        tl_x = tr_x - new_width;
        tl_y = tr_y;
        diag_orientation = DiagOrientation::TopRightBottomLeft;
    }

    Ok((
        (tl_x as i32, tl_y as i32),
        new_width as i32,
        new_height as i32,
        diag_orientation,
    ))
}

/**
 * Replaces a image region within the background image
 *
 * @param background The background image that the region comes from
 * @param region The replacement image region
 * @param (x, y) The coordinates for the image region in the background image
 */
#[allow(unused_variables)]
fn replace_region(
    background: &core::Mat,
    region: core::Mat,
    (x, y): Coordinates,
    diag_orientation: DiagOrientation,
) -> Result<core::Mat> {
    let mut temp_image = core::Mat::copy(background)?;
    let full_width = background.cols();
    let full_height = background.rows();
    let width = region.cols();
    let height = region.rows();

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
    vertical_panels_vec.push(region);
    vertical_panels_vec.push(bottom_panel);

    core::vconcat(&vertical_panels_vec, &mut vertical_panel)?;

    let mut horizontal_panels_vec: core::Vector<core::Mat> = core::Vector::new();

    let mut result: core::Mat = core::Mat::default();

    horizontal_panels_vec.push(left_panel);
    horizontal_panels_vec.push(vertical_panel);
    horizontal_panels_vec.push(right_panel);

    core::hconcat(&horizontal_panels_vec, &mut result)?;

    temp_image = result;

    #[cfg(feature = "debug")]
    {
        use imageproc::rect::Rect;

        let mut temp_image_buffer = image_conversion::mat_to_image_buffer(&temp_image)?;

        drawing::draw_hollow_rect_mut(
            &mut temp_image_buffer,
            Rect::at(x, y).of_size(width as u32, height as u32),
            Rgb([0, 255, 0]),
        );

        match diag_orientation {
            DiagOrientation::TopLeftBottomRight => {
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
            }
            DiagOrientation::TopRightBottomLeft => {
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
        }

        temp_image = image_conversion::image_buffer_to_mat(temp_image_buffer)?;
    }

    Ok(temp_image)
}

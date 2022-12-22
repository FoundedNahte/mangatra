use anyhow::Result;
use leptess::{LepTess, Variable};
use opencv::{core, imgcodecs};

// The Tesseract API only accepts in-memory files in the TIFF format;
// We encode each text region as a TIFF file
fn encode_in_tiff(data: &core::Mat, count: i32) -> Result<Vec<u8>> {
    let mut buffer: core::Vector<u8> = core::Vector::new();

    //imgcodecs::imwrite(&format!("{count}.png"), data, &core::Vector::new())?;

    imgcodecs::imencode(".tiff", &data, &mut buffer, &core::Vector::new())?;

    let mut copied_buffer: Vec<u8> = vec![0; buffer.len()];

    copied_buffer[..].copy_from_slice(buffer.as_slice());

    Ok(copied_buffer)
}

// Extract text for a vector of text regions.
pub fn extract_text(text_boxes: &core::Vector<core::Mat>) -> Result<Vec<String>> {
    let mut lt = LepTess::new(Some("C:/tools/tesseract/tessdata"), "jpn_vert")?;

    lt.set_variable(Variable::TesseditPagesegMode, "5")?;

    let mut extracted_text: Vec<String> = Vec::new();

    let mut count: i32 = 0;

    for bbox in text_boxes {
        let encoded_data = encode_in_tiff(&bbox, count)?;

        lt.set_image_from_mem(&encoded_data[..])?;

        extracted_text.push(lt.get_utf8_text()?);

        count += 1;
    }

    Ok(extracted_text)
}

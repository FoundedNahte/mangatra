use anyhow::Result;
use leptess::{LepTess, Variable};
use opencv::{core, imgcodecs};

pub struct OCR {
    leptess: LepTess,
}

impl OCR {
    pub fn new(data_path: &str) -> Result<OCR> {
        let leptess = LepTess::new(Some(data_path), "jpn_vert")?;

        Ok(OCR { leptess })
    }

    pub fn extract_text(&mut self, text_boxes: &core::Vector<core::Mat>) -> Result<Vec<String>> {
        self.leptess
            .set_variable(Variable::TesseditPagesegMode, "5")?;

        let mut extracted_text: Vec<String> = Vec::new();

        let mut count: i32 = 0;

        for bbox in text_boxes {
            let encoded_data = Self::encode_in_tiff(&bbox, count)?;

            self.leptess.set_image_from_mem(&encoded_data[..])?;

            extracted_text.push(self.leptess.get_utf8_text()?);

            count += 1;
        }

        Ok(extracted_text)
    }

    // The Tesseract API only accepts in-memory files in the TIFF format;
    // We encode each text region as a TIFF file
    fn encode_in_tiff(data: &core::Mat, count: i32) -> Result<Vec<u8>> {
        let mut buffer: core::Vector<u8> = core::Vector::new();

        imgcodecs::imwrite(&format!("{count}.png"), data, &core::Vector::new())?;

        imgcodecs::imencode(".tiff", &data, &mut buffer, &core::Vector::new())?;

        let mut copied_buffer: Vec<u8> = vec![0; buffer.len()];

        copied_buffer[..].copy_from_slice(buffer.as_slice());

        Ok(copied_buffer)
    }
}

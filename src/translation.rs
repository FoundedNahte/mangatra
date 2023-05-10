use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct Request {
    content: Vec<String>,
    message: String,
}

// Translation through Sugoi Translator
pub fn translate(text: &[String]) -> Result<Vec<String>> {
    let client = reqwest::blocking::Client::new();

    let json_data = Request {
        message: "translate sentences".to_string(),
        content: text.to_vec(),
    };

    let res = client
        .post("http://localhost:14366")
        .json(&json_data)
        .send()?;

    Ok(res.json()?)
}

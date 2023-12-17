<div align="center">

<a href="https://github.com/FoundedNahte/mangatra"><img src="assets/mangatra_logo.png" role="img"></a>

</div>

## About

Mangatra is a CLI tool that aims to provide the ability to rapidly translate manga.

It can function as a tool for translators as well as an end-to-end solution for hands-off translation.
 - Extract/Replace modes allow translators to extract text into JSONs for translation and replace text boxes with translated JSONs.
 - Translation mode is the combination of extract and replace mode as well as translation from ct2sugoi to provide end-to-end translation. 

It uses YOLOv5 to identify text boxes, OpenCV for image manipulation, and libtesseract for OCR capabilities. Because of the use of YOLOv5 for text box detection, the application can also be generalized to other forms of text besides manga.

## Demo
<div align="center">

### Translating

| Input             | Output |
:-------------------------:|:-------------------------:
![](https://github.com/FoundedNahte/mangatra/blob/master/assets/input.png)  | ![](https://github.com/FoundedNahte/mangatra/blob/master/assets/output.png)

### Cleaning

| Cleaned |
:-------------------------:
![](https://github.com/FoundedNahte/mangatra/blob/master/assets/cleaned.png)

</div>

## Usage
```
Usage: mangatra [OPTIONS] --input <INPUT> --model <MODEL> --lang <LANG>

Options:
  -i, --input <INPUT>      Input path for a directory of images or single image
  -o, --output <OUTPUT>    Specify output location for text or image outputs. If not specified, application will use the same directory as the input
  -t, --text <TEXT>        [Optional] Specify a path to the translated JSONs
  -m, --model <MODEL>      Path to the YOLOv5 detection weights (ONNX format)
  -l, --lang <LANG>        Specify the language for tesseract
  -d, --data <DATA>        [Optional] Specify path to the tessdata folder for tesseract. If no path is specified, the application will look under the 'TESSDATA_PREFIX' environment variable
  -p, --padding <PADDING>  Specify size of padding for text regions
      --single             Use single-threading for image processing
      --clean              If set, the program will output cleaned pages in PNG format in the output directory
  -h, --help               Print help information
  -V, --version            Print version information
```

## Installation
You need three things:
- OpenCV
- tesseract
- tesseract language data
### Debian-based linux systems
OpenCV and tesseract
```
sudo apt-get install libopencv-dev libleptonica-dev libtesseract-dev
```
Tesseract language specific data
```
sudo apt-get install tesseract-ocr-jpn
```
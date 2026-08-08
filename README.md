---
title: Sylheti Nagri OCR
emoji: 🔤
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
pinned: false
---

# Sylheti Nagri OCR Application

Detects, crops, and performs OCR on Sylheti Nagri script images, then reconstructs the text as HTML.

## Run locally

```bash
source sylheti_ocr_env/bin/activate && pip install -r requirements.txt && python app.py
```

## Features

- Word detection and cropping
- OCR on cropped word images
- HTML reconstruction of detected text
- Downloadable ZIP of cropped words and metadata

## Requirements

- Python 3.10
- Dependencies in `requirements.txt`

## Installation

```bash
python -m venv sylheti_ocr_env
# macOS/Linux:
source sylheti_ocr_env/bin/activate
# Windows:
sylheti_ocr_env\Scripts\activate
pip install -r requirements.txt
mkdir -p ~/.apsis_ocr/line
```

## Run

```bash
python app.py
```

(Opens the Gradio UI at `http://127.0.0.1:7860`.)

## Usage

1. Upload an image (`.png`, `.jpg`, `.jpeg`)
2. Wait for word detection and sorting
3. Download cropped words/metadata or click **Run OCR and Create HTML**
4. Preview and download the HTML reconstruction

## Notes

- CPU inference.
- Requires `model_weights.weights.h5`, `vit_model_weights.weights.h5`, and `Surma-4.000/Surma-Regular.ttf` in the project directory.

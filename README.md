---
title: Sylheti Nagri OCR
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: true
license: mit
thumbnail: >-
  https://cdn-uploads.huggingface.co/production/uploads/6562971fb9218ed1a77be0c1/5hS1aGQDxSvNXCOvTV1dg.png
short_description: Detects and crops Sylheti Nagri script words in images, runs
---

# Sylheti Nagri OCR

Detects and crops Sylheti Nagri script words in images, runs OCR (CRNN + ViT), and reconstructs the text as HTML with the Surma font embedded.

## Layout

- `app.py` — Gradio app (deployed product)
- `samples/` — sample images loaded by the app
- `assets/`, `models/detector/word/`, `fonts/`, `weights/` — logo, detector models, fonts, OCR weights
- `wheels/` — patched `fastdeploy_tools` wheel fetched by `requirements.txt`
- `scripts/` — assert-based test scripts

## Features

- Word detection and cropping
- OCR on cropped word images (CRNN and ViT models)
- HTML reconstruction of detected text
- Downloadable ZIP of cropped words and metadata

## Run locally

Requires Python 3.10 and model weights (see below).

```bash
python -m venv sylheti_ocr_env
source sylheti_ocr_env/bin/activate
pip install -r requirements.txt
python app.py
```

Opens the Gradio UI at `http://127.0.0.1:7860`.

First model load seeds detector files from `models/detector/word/` into `~/.apsis_ocr/word/` automatically.

## Tests

Plain assert scripts, run from the repo root inside the venv:

```bash
python scripts/test_core.py
python scripts/test_loaders.py
python scripts/test_pipeline.py
python scripts/test_seed.py
python scripts/smoke_test.py   # slow: end-to-end on 3 samples, both models
```

## Model weights

Required at runtime:

- `weights/model_weights.weights.h5` — CRNN OCR (~1.1 MB)
- `weights/vit_model_weights.weights.h5` — ViT OCR (~11 MB)
- `fonts/Surma-Regular.ttf` — output font

## Font license

`fonts/Surma-Regular.ttf` is Copyright (c) 1999-2021, Sylheti Translation And Research (<http://www.sylheti.org.uk/>), with Reserved Font Name "Surma", and is licensed under the SIL Open Font License, Version 1.1. The license and full details are available at <https://openfontlicense.org/>.

## Notes

- CPU inference only.
- Do not upgrade pinned dependencies in `requirements.txt` (numpy, opencv, apsisocr, fastdeploy-python, tensorflow) — see `AGENTS.md`.
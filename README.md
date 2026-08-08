# Sylheti Nagri OCR

Detects and crops Sylheti Nagri script words in images, runs OCR (CRNN + ViT), and reconstructs the text as HTML with the Surma font embedded.

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

Keep at the repo root (all required at runtime):

- `model_weights.weights.h5` — CRNN OCR (~1.1 MB)
- `vit_model_weights.weights.h5` — ViT OCR (~11 MB)
- `Surma-4.000/Surma-Regular.ttf` — output font

## Notes

- CPU inference only.
- Do not upgrade pinned dependencies in `requirements.txt` (numpy, opencv, apsisocr, fastdeploy-python, tensorflow) — see `AGENTS.md`.

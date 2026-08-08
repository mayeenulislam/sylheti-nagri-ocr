# AGENTS.md

Sylheti Nagri OCR: detects/crops Sylheti script words, OCRs them (CRNN + ViT), reconstructs HTML. Two app variants live on two branches; the deployed product is the Gradio app.

## Branch & remote model (critical)

- `main` = **Gradio app** (`app.py`). This is the deployed, working product.
- `streamlit` = **Streamlit app** (`streamlit_app.py` / `streamlit_vit.py`, Docker). Legacy variant; keep in sync when touching shared code.
- `origin` = GitHub, `hf` = HuggingFace Space (`spaces/mayeenulislam/sylheti-nagri-ocr`). Deployment = pushing `main` to `hf` (Space builds from `main`).
- Do NOT push binary/LFS assets to `hf` with plain `git push`; use `git lfs push` for large files. `wheels/` is tracked on `main` because the HF build only mounts `requirements.txt`.
- The `streamlit` branch (6e8e765) is a rewritten history with real file blobs baked in and **no `.gitattributes`**; `main` uses git-lfs pointers. They are the same app code — do not "reconcile" the trees.

## Local dev

```bash
source sylheti_ocr_env/bin/activate   # Python 3.10, gitignored
pip install -r requirements.txt
python app.py                         # Gradio UI at http://127.0.0.1:7860
```

- First model load seeds detector files from `models/detector/word/` into `~/.apsis_ocr/word/` automatically. If README mentions `mkdir -p ~/.apsis_ocr/line`, ensure it exists before running.
- `requirements.txt` installs a patched wheel by URL from the HF space itself (`fastdeploy_tools-0.0.6`) because upstream `fastdeploy-tools==0.0.5` pins `uvicorn==0.16.0`, which conflicts with Gradio. Do not unpin or drop the URL line.

## Tests

No pytest — plain assert scripts run directly from repo root (imports are `sys.path`-hacked):

```bash
python scripts/test_core.py
python scripts/test_loaders.py
python scripts/test_pipeline.py
python scripts/test_seed.py
python scripts/smoke_test.py   # slow: runs CRNN + ViT on 3 samples, needs model weights present
```

Run inside the venv. `smoke_test.py` is the full end-to-end check.

## Hard version constraints (do not upgrade)

`numpy==1.23.5`, `opencv-python==4.11.0.86`, `apsisocr==0.0.7`, `fastdeploy-python==1.0.7`, `tensorflow==2.17.1` (loaded via `tf.keras.config.enable_unsafe_deserialization()`), `mltu` (provides `CTCloss`, `CWERMetric`, `ImageResizer`). Custom model architectures (CRNN+BiLSTM, ViT) are defined inline in `app.py`/`streamlit_vit.py`.

## HF Space quirks

- Gradio handlers are decorated with `@spaces.GPU(duration=60)` — ZeroGPU requires this on HF; harmless locally.
- `sample_page/` must NOT be excluded from the Space build (app loads samples at startup) — `.dockerignore` excludes it for local Docker only.
- CPU inference; weights `model_weights.weights.h5` (CRNN, ~1.1 MB) and `vit_model_weights.weights.h5` (ViT, ~11 MB) must be present at repo root.

## Not committed on purpose

`docs/` and `specs/` are gitignored (working files, not tracked). `docs/deployment.md` is stale — it predates the working Gradio Space and claims HF deployment fails.

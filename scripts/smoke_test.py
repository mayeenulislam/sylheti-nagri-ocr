import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import app

SAMPLES = ["3.PNG", "old_testament.PNG", "old_testament2.JPG"]

def line_count(html):
    return html.count('<div class="text-line"') if html else 0

def run_sample(sample, model_name):
    img = app.load_sample(sample)
    assert img is not None
    state, viz, meta, zip_path, json_path, status = app.detect_and_crop(img)
    assert not status, f"{sample}: detect_and_crop error: {status}"
    assert viz is not None and len(meta["word_data"]) > 0, f"{sample}: no words"
    assert os.path.exists(zip_path) and os.path.getsize(zip_path) > 0, f"{sample}: zip missing"
    html, html_path, status = app.run_ocr(model_name, state)
    assert not status, f"{sample}: ocr error: {status}"
    assert html and "page-container" in html, f"{sample}: bad html"
    return len(meta["word_data"]), line_count(html)

def main():
    for model in ["CRNN", "ViT"]:
        print(f"=== {model} ===")
        for sample in SAMPLES:
            words, lines = run_sample(sample, model)
            print(f"{sample}: {words} words, {lines} lines")
    print("smoke_test: PASS")

if __name__ == "__main__":
    main()

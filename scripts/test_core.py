import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import app

def test_load_sample():
    img = app.load_sample("Gospel-Mathew.PNG")
    assert img is not None and img.shape[2] == 3, "sample should load as RGB"
    assert app.load_sample("Upload my own") is None

def test_run_ocr_without_state():
    html, path, msg = app.run_ocr("CRNN", None)
    assert html is None and "Step 1" in msg, msg

if __name__ == "__main__":
    test_load_sample()
    test_run_ocr_without_state()
    print("test_core: PASS")

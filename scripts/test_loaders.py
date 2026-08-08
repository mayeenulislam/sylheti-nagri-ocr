import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import app

def test_get_recognizer_rejects_bad_name():
    try:
        app.get_recognizer("Bogus")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass

def test_recognizer_fields_present():
    configs, model, resizer, vocab = app.get_recognizer("CRNN")
    assert vocab and len(vocab) > 10
    assert configs.height == 32 and configs.width == 128
    assert model is not None
    print("CRNN load OK; vocab length:", len(vocab))

if __name__ == "__main__":
    test_get_recognizer_rejects_bad_name()
    test_recognizer_fields_present()
    print("test_loaders: PASS")

import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import seed_detector_models, DETECTOR_FILES, BUNDLED_DETECTOR_DIR

def test_seeds_missing_and_skips_existing():
    tmp = os.path.join(os.path.dirname(__file__), ".seed_tmp")
    shutil.rmtree(tmp, ignore_errors=True)

    seed_detector_models(home_dir=tmp)
    word_dir = os.path.join(tmp, ".apsis_ocr", "word")
    for f in DETECTOR_FILES:
        assert os.path.exists(os.path.join(word_dir, f)), f"{f} not seeded"
        assert os.path.getsize(os.path.join(word_dir, f)) == os.path.getsize(
            os.path.join(BUNDLED_DETECTOR_DIR, f)
        ), f"{f} size mismatch"

    sentinel = os.path.join(word_dir, "inference.pdmodel")
    before = os.path.getmtime(sentinel)
    seed_detector_models(home_dir=tmp)
    assert os.path.getmtime(sentinel) == before, "existing model was overwritten"

    shutil.rmtree(tmp, ignore_errors=True)
    print("test_seeds_missing_and_skips_existing: PASS")

if __name__ == "__main__":
    test_seeds_missing_and_skips_existing()

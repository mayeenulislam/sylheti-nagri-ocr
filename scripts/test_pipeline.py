import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import sort_boxes_into_lines, generate_html

def test_sort_simple_boxes():
    boxes = [
        [10, 10, 50, 30, 50, 30, 10, 30],   # line 1, word A
        [60, 10, 100, 30, 100, 30, 60, 30], # line 1, word B
        [10, 50, 50, 70, 50, 70, 10, 70],   # line 2, word C
    ]
    out = sort_boxes_into_lines(boxes)
    assert len(out) == 3, out
    # horizontal order within line 1
    assert out[0]["sort_key"] == 1 and out[1]["sort_key"] == 2
    # line 2 comes after line 1 vertically
    assert out[2]["sort_key"] == 3 and out[2]["box"][1] > out[0]["box"][1]

def test_generate_html_rejects_empty():
    assert generate_html([], [], None, None, "abc", "x.ttf") is None

if __name__ == "__main__":
    test_sort_simple_boxes()
    test_generate_html_rejects_empty()
    print("test_pipeline: PASS")

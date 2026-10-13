"""Self-check for the demo's pure helpers: `python test_app.py`.

The model-running paths need torch and the Hub; these are the bits that can be
silently wrong (frame ordering, padded-index mapping, box geometry).
"""

import tempfile
from pathlib import Path

from app import build_animation, collect_frames, draw_boxes, input_index, tube_color
from PIL import Image


def test_input_index():
    # No padding: identity.
    assert [input_index(i, []) for i in range(3)] == [0, 1, 2]
    # Frames 0 and 3 are synthetic duplicates; the rest shift down past them.
    assert input_index(0, [0, 3]) is None
    assert input_index(1, [0, 3]) == 0
    assert input_index(2, [0, 3]) == 1
    assert input_index(3, [0, 3]) is None
    assert input_index(4, [0, 3]) == 2


def test_collect_frames_orders_by_filename():
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        unordered = [str(work / "b_02.jpg"), str(work / "a_01.jpg")]
        assert [p.name for p in collect_frames(unordered, None, 8, work)] == [
            "a_01.jpg",
            "b_02.jpg",
        ]


def test_collect_frames_rejects_non_images():
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        try:
            collect_frames([str(work / "notes.txt")], None, 8, work)
        except Exception as e:
            assert "notes.txt" in str(e)
        else:
            raise AssertionError("expected a rejection")


def test_draw_boxes_marks_the_box_pixels():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "frame.jpg"
        Image.new("RGB", (200, 100), "black").save(path)
        # Centred box, half the frame: outline lands on the mid-height edges.
        out = draw_boxes(path, [((0.5, 0.5, 0.5, 0.5), 0.42, "#ff0000")])
        assert out.size == (200, 100)
        assert out.getpixel((50, 50))[0] > 100, "left edge should be drawn"
        assert out.getpixel((150, 50))[0] > 100, "right edge should be drawn"
        assert out.getpixel((100, 50)) == (0, 0, 0), "box interior stays untouched"


def test_build_animation_is_animated_and_downscaled():
    assert build_animation([]) is None
    frames = [Image.new("RGB", (1800, 900), c) for c in ("red", "green", "blue")]
    anim = Image.open(build_animation(frames))
    assert anim.n_frames == 3
    assert anim.size == (900, 450), "wider frames scale down to ANIM_MAX_WIDTH"


def test_tube_color_cycles():
    assert tube_color(0) == tube_color(10) != tube_color(1)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")

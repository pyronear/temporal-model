from PIL import Image

from temporal_model.eval.render import (
    _lightning_polygon,
    correctness_label,
    crop_around_bbox,
    day_of,
    draw_bboxes,
    frame_bboxes_by_input_index,
    legend_html,
    processed_to_input_index,
    row_background,
    triggering_tube_ids,
    tube_color,
    tube_input_boxes,
    tube_timeline_df,
)


def test_day_of():
    assert day_of("2026-05-19T14:10:01.123") == "2026-05-19"
    assert day_of(None) == "unknown"


def test_processed_to_input_index_no_padding_is_identity():
    assert processed_to_input_index(0, []) == 0
    assert processed_to_input_index(5, []) == 5


def test_processed_to_input_index_with_padding():
    # 2 real frames padded to 4 -> synthetic slots [0, 3], real at 1,2 -> input 0,1
    padded = [0, 3]
    assert processed_to_input_index(1, padded) == 0
    assert processed_to_input_index(2, padded) == 1
    assert processed_to_input_index(0, padded) is None
    assert processed_to_input_index(3, padded) is None


def test_frame_bboxes_by_input_index_skips_none_bbox():
    details = {
        "preprocessing": {"padded_frame_indices": []},
        "tubes": {
            "kept": [
                {
                    "tube_id": 0,
                    "entries": [
                        {
                            "frame_idx": 0,
                            "bbox": [0.5, 0.5, 0.1, 0.1],
                            "confidence": 0.7,
                        },
                        {"frame_idx": 1, "bbox": None},
                        {"frame_idx": 2, "bbox": [0.2, 0.2, 0.05, 0.05]},
                    ],
                }
            ]
        },
    }
    out = frame_bboxes_by_input_index(details)
    assert out == {
        0: [((0.5, 0.5, 0.1, 0.1), 0.7, 0)],
        2: [((0.2, 0.2, 0.05, 0.05), None, 0)],
    }


def test_tube_input_boxes():
    tube = {
        "entries": [
            {"frame_idx": 0, "bbox": [0.5, 0.5, 0.1, 0.1], "confidence": 0.7},
            {"frame_idx": 1, "bbox": None},
        ]
    }
    assert tube_input_boxes(tube, []) == [(0, (0.5, 0.5, 0.1, 0.1), 0.7)]


def test_lightning_polygon_has_six_points():
    pts = _lightning_polygon(10, 20, 16)
    assert len(pts) == 6
    assert all(len(p) == 2 for p in pts)


def test_draw_bboxes_preserves_size(tmp_path):
    p = tmp_path / "f.jpg"
    Image.new("RGB", (100, 80), "white").save(p)
    out = draw_bboxes(
        p,
        [
            ((0.5, 0.5, 0.2, 0.2), 0.9, "#1f77b4", "decisive"),
            ((0.4, 0.4, 0.1, 0.1), 0.5, "#2ca02c", "would"),
            ((0.3, 0.3, 0.1, 0.1), None, "#ff7f0e", None),
        ],
    )
    assert out.size == (100, 80)


def test_crop_around_bbox_returns_square_patch(tmp_path):
    p = tmp_path / "f.jpg"
    Image.new("RGB", (320, 240), "white").save(p)
    out = crop_around_bbox(p, (0.5, 0.5, 0.1, 0.1), context_factor=2.0, patch_size=224)
    assert out.size == (224, 224)


def test_correctness_label():
    assert correctness_label("discarded-smoke") == "🔴 missed smoke"
    assert correctness_label("kept-fp") == "🟠 false alarm"
    assert correctness_label("n/a") == "—"


def test_row_background_errors_and_unknown():
    # errors get their own colours, regardless of verdict
    assert row_background("discard", "🔴 missed smoke") == "#f4b4b4"
    assert row_background("keep", "🟠 false alarm") == "#fbdca0"
    # unknown ground truth ("—") falls back to a verdict-based tint
    assert row_background("keep", "—") != row_background("discard", "—")


def test_legend_html_mentions_each_colour():
    html = legend_html()
    assert "#f4b4b4" in html and "#fbdca0" in html  # error colours present
    assert "missed smoke" in html and "false alarm" in html


def test_tube_color_stable_and_distinct():
    assert tube_color(0) != tube_color(1)  # distinct for different tubes
    assert tube_color(0) == tube_color(0)  # stable
    assert tube_color(0).startswith("#")


def test_tube_timeline_df_one_row_per_present_frame():
    df = tube_timeline_df([("T0", {2: 0.8, 3: 0.6}), ("T1", {5: 0.9})])
    assert list(df.columns) == ["tube", "frame", "frame_end", "confidence"]
    assert len(df) == 3
    t0 = df[df["tube"] == "T0"].sort_values("frame")
    assert list(t0["frame"]) == [2, 3]
    assert list(t0["frame_end"]) == [3, 4]
    assert list(t0["confidence"]) == [0.8, 0.6]


def test_triggering_tube_ids_logistic_uses_probability():
    details = {
        "decision": {"aggregation": "logistic", "threshold": 0.5},
        "tubes": {
            "kept": [
                {"tube_id": 0, "probability": 0.89},
                {"tube_id": 2, "probability": 0.03},
                {"tube_id": 3, "probability": 0.72},
            ]
        },
    }
    assert triggering_tube_ids(details) == {0, 3}


def test_triggering_tube_ids_max_logit_uses_logit():
    details = {
        "decision": {"aggregation": "max_logit", "threshold": 1.0},
        "tubes": {
            "kept": [
                {"tube_id": 0, "logit": 2.5, "probability": 0.4},
                {"tube_id": 1, "logit": 0.2, "probability": 0.9},
            ]
        },
    }
    # Qualifies on logit >= threshold, not on the calibrated probability.
    assert triggering_tube_ids(details) == {0}


def test_triggering_tube_ids_no_threshold_is_empty():
    assert triggering_tube_ids({}) == set()
    assert triggering_tube_ids({"decision": {}, "tubes": {"kept": []}}) == set()


def test_app_module_imports():
    # App must import cleanly (no Streamlit calls at import time).
    import temporal_model.eval.app as app  # noqa: PLC0415

    assert hasattr(app, "main")
    assert hasattr(app, "reporting_dirs")

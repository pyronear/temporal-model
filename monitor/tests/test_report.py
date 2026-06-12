import json

from temporal_model.monitor.report import (
    OrgReport,
    compute_outcome,
    reshape_details,
    result_row,
    write_report,
)
from temporal_model.monitor.store import FrameMeta, SequenceMeta

VERBOSE_RESPONSE = {
    "is_smoke": True,
    "probability": 0.93,
    "version": {"api": "0.3.1", "model": "0.1.0"},
    "details": {
        "decision": {
            "aggregation": "logistic",
            "threshold": 0.52,
            "threshold_overridden": False,
            "packaged_threshold": None,
        },
        "preprocessing": {
            "num_frames_input": 10,
            "num_truncated": 0,
            "padded_frame_indices": [],
            "num_tube_candidates": 3,
            "num_tubes_outside_roi": 1,
        },
        "tubes": [
            {
                "tube_id": 7,
                "start_frame": 2,
                "end_frame": 4,
                "logit": 3.41,
                "probability": 0.93,
                "entries": [
                    {
                        "frame_idx": 2,
                        "bbox": [0.2, 0.2, 0.2, 0.2],
                        "is_gap": False,
                        "confidence": 0.81,
                    },
                    {
                        "frame_idx": 3,
                        "bbox": [0.5, 0.5, 0.2, 0.2],
                        "is_gap": False,
                        "confidence": 0.7,
                    },
                ],
            }
        ],
        "profiling": None,
    },
}


def make_meta() -> SequenceMeta:
    return SequenceMeta(
        key="alert-api_42307",
        sequence_id=42307,
        label="smoke",
        camera_name="donon-sarrebourg-01",
        organization_name="sis-67",
        started_at="2026-05-15T13:08:18",
        temporal_model_score=0.93,
        temporal_model_version="0.1.0",
        temporal_api_version="0.3.1",
        frames=[
            FrameMeta(
                file="images/detection_100.jpg",
                detection_id=100,
                created_at="2026-05-15T13:08:18",
                bucket_key="cam122/frame-100.jpg",
                bbox="[(0.1,0.2,0.3,0.4,0.9)]",
            )
        ],
    )


def test_compute_outcome_matches_eval():
    assert compute_outcome("keep", "smoke") == "kept-smoke"
    assert compute_outcome("discard", "smoke") == "discarded-smoke"
    assert compute_outcome("keep", "fp") == "kept-fp"
    assert compute_outcome("discard", "fp") == "discarded-fp"
    assert compute_outcome("keep", "unknown") == "n/a"


def test_reshape_details_to_eval_shape():
    details = reshape_details(VERBOSE_RESPONSE["details"])
    assert set(details) == {"preprocessing", "tubes", "decision"}
    assert details["preprocessing"] == {
        "num_frames_input": 10,
        "num_truncated": 0,
        "padded_frame_indices": [],
    }
    assert details["tubes"]["num_candidates"] == 3
    assert details["tubes"]["num_outside_roi"] == 1
    kept = details["tubes"]["kept"]
    assert len(kept) == 1
    tube = kept[0]
    # absent on <= v0.3.1 -> filled with None / derived
    assert tube["first_crossing_frame"] is None
    cx, cy, w, h = tube["stabilized_window"]
    assert (round(cx, 6), round(cy, 6), round(w, 6), round(h, 6)) == (
        0.35,
        0.35,
        0.5,
        0.5,
    )
    assert details["decision"] == {
        "aggregation": "logistic",
        "threshold": 0.52,
        "trigger_tube_id": None,
    }


def test_reshape_preserves_trigger_fields_when_present():
    resp = json.loads(json.dumps(VERBOSE_RESPONSE))  # deep copy
    resp["details"]["decision"]["trigger_tube_id"] = 7
    resp["details"]["tubes"][0]["first_crossing_frame"] = 3
    details = reshape_details(resp["details"])
    assert details["decision"]["trigger_tube_id"] == 7
    assert details["tubes"]["kept"][0]["first_crossing_frame"] == 3


def test_result_row():
    meta = make_meta()
    details = reshape_details(VERBOSE_RESPONSE["details"])
    row = result_row(
        meta=meta,
        response=VERBOSE_RESPONSE,
        details=details,
        replay_matches=True,
    )
    # decision and probability come from production's recorded score (0.93 > 0.52)
    assert row == {
        "key": "alert-api_42307",
        "source": "alert-api",
        "label": "smoke",
        "decision": "keep",
        "outcome": "kept-smoke",
        "score": 3.41,
        "probability": 0.93,
        "num_tubes_kept": 1,
        "trigger_frame_index": None,
        "organization_name": "sis-67",
        "camera_name": "donon-sarrebourg-01",
        "started_at": "2026-05-15T13:08:18",
        "replayed_probability": 0.93,
        "replayed_decision": "keep",
        "replay_matches": True,
        "matched_window_frames": None,
        "temporal_model_version": "0.1.0",
        "temporal_api_version": "0.3.1",
    }


def test_result_row_source_is_fixed_alert_api():
    meta = make_meta().model_copy(update={"organization_name": "SIS 67"})
    details = reshape_details(VERBOSE_RESPONSE["details"])
    row = result_row(
        meta=meta, response=VERBOSE_RESPONSE, details=details, replay_matches=True
    )
    # source is always the fixed slug; organization_name carries the raw name
    assert row["source"] == "alert-api"
    assert row["organization_name"] == "SIS 67"


def test_write_report_tree(tmp_path):
    meta = make_meta()
    details = reshape_details(VERBOSE_RESPONSE["details"])
    row = result_row(
        meta=meta, response=VERBOSE_RESPONSE, details=details, replay_matches=True
    )
    report = OrgReport(org_slug="sis-67")
    report.add(
        row=row,
        details=details,
        view={
            "key": "alert-api_42307",
            "source": "sis-67",
            "label": "smoke",
            "organization_name": "sis-67",
            "camera_name": "donon-sarrebourg-01",
            "started_at": "2026-05-15T13:08:18",
            "frames": [
                "data/01_raw/sequences/sis-67/donon-sarrebourg-01/seq_42307/images/detection_100.jpg"
            ],
        },
        model_config={"variant": "vit_dinov2_finetune"},
    )
    report.drop("alert-api_999", "no_temporal_version")
    write_report(tmp_path, report)

    out = tmp_path / "sis-67" / "vit_dinov2_finetune"
    rows = json.loads((out / "results.json").read_text())
    assert rows[0]["key"] == "alert-api_42307"
    written = json.loads((out / "details" / "alert-api_42307.json").read_text())
    assert written["tubes"]["kept"][0]["tube_id"] == 7
    view = json.loads((out / "sequences" / "alert-api_42307.json").read_text())
    assert view["frames"][0].startswith("data/01_raw/sequences/")
    assert json.loads((out / "model_config.json").read_text()) == {
        "variant": "vit_dinov2_finetune"
    }
    assert json.loads((out / "dropped.json").read_text()) == [
        {"sequence_id": "alert-api_999", "reason": "no_temporal_version"}
    ]

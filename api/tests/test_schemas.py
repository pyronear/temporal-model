from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from temporal_model.api.schemas import PredictRequest, to_response


def _details(kept):
    return {
        "decision": {
            "aggregation": "max_logit",
            "threshold": 0.5,
            "trigger_tube_id": kept[0]["tube_id"] if kept else None,
        },
        "preprocessing": {
            "num_frames_input": 30,
            "num_truncated": 0,
            "padded_frame_indices": [],
        },
        "tubes": {"num_candidates": len(kept) + 1, "kept": kept},
    }


def _tube(tube_id, prob):
    return {
        "tube_id": tube_id,
        "start_frame": 2,
        "end_frame": 12,
        "logit": 3.4,
        "probability": prob,
        "first_crossing_frame": 3,
        "entries": [
            {
                "frame_idx": 2,
                "bbox": [1.0, 2.0, 3.0, 4.0],
                "is_gap": False,
                "confidence": 0.8,
            },
            {"frame_idx": 3, "bbox": None, "is_gap": True, "confidence": None},
        ],
    }


def test_request_rejects_empty():
    with pytest.raises(ValidationError):
        PredictRequest(frames=[])


def test_request_rejects_scheme():
    with pytest.raises(ValidationError):
        PredictRequest(frames=["s3://bucket/a.jpg"])


def test_smoke_default_uses_trigger_tube_probability():
    out = SimpleNamespace(
        is_positive=True, trigger_frame_index=3, details=_details([_tube(7, 0.98)])
    )
    resp = to_response(out, name="m", version="1.2.0", calibrated=True, verbose=False)
    dumped = resp.model_dump(exclude_unset=True)
    assert dumped == {
        "is_smoke": True,
        "probability": 0.98,
        "trigger_frame_index": 3,
        "model": {"name": "m", "version": "1.2.0"},
    }


def test_negative_uses_max_kept_probability():
    out = SimpleNamespace(
        is_positive=False,
        trigger_frame_index=None,
        details=_details([_tube(1, 0.1), _tube(2, 0.41)]),
    )
    resp = to_response(out, name="m", version="1.2.0", calibrated=True, verbose=False)
    assert resp.probability == 0.41
    assert resp.is_smoke is False
    assert resp.trigger_frame_index is None


def test_negative_no_tubes_is_zero_when_calibrated():
    out = SimpleNamespace(
        is_positive=False, trigger_frame_index=None, details=_details([])
    )
    resp = to_response(out, name="m", version="1.2.0", calibrated=True, verbose=False)
    assert resp.probability == 0.0


def test_uncalibrated_probability_is_null():
    out = SimpleNamespace(
        is_positive=False, trigger_frame_index=None, details=_details([_tube(1, None)])
    )
    resp = to_response(out, name="m", version=None, calibrated=False, verbose=False)
    assert resp.probability is None


def test_smoke_trigger_tube_missing_returns_none():
    details = _details([_tube(7, 0.98)])
    details["decision"]["trigger_tube_id"] = 999  # not among kept tubes
    out = SimpleNamespace(is_positive=True, trigger_frame_index=3, details=details)
    resp = to_response(out, name="m", version="1.2.0", calibrated=True, verbose=False)
    assert resp.probability is None


def test_verbose_adds_details_block():
    out = SimpleNamespace(
        is_positive=True, trigger_frame_index=3, details=_details([_tube(7, 0.98)])
    )
    resp = to_response(out, name="m", version="1.2.0", calibrated=True, verbose=True)
    dumped = resp.model_dump(exclude_unset=True)
    assert dumped["details"]["decision"] == {
        "aggregation": "max_logit",
        "threshold": 0.5,
        "trigger_tube_id": 7,
    }
    assert dumped["details"]["preprocessing"]["num_tube_candidates"] == 2
    assert dumped["details"]["tubes"][0]["tube_id"] == 7
    assert dumped["details"]["tubes"][0]["entries"][1]["bbox"] is None

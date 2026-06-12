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
        "tubes": {"num_candidates": len(kept) + 1, "num_outside_roi": 0, "kept": kept},
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


def test_request_bucket_defaults_to_none():
    assert PredictRequest(frames=["a.jpg"]).bucket is None


def test_request_accepts_bucket():
    assert PredictRequest(frames=["a.jpg"], bucket="my-bucket").bucket == "my-bucket"


def test_request_rejects_empty_bucket():
    with pytest.raises(ValidationError):
        PredictRequest(frames=["a.jpg"], bucket="")


def test_request_accepts_alert_api_bucket():
    bucket = "2eb7ac42fbbf-alert-api-2"
    assert PredictRequest(frames=["a.jpg"], bucket=bucket).bucket == bucket


@pytest.mark.parametrize(
    "bad",
    ["s3://my-bucket", "My-Bucket", "has space", "a/b", "ab", "x" * 64, "a..b"],
)
def test_request_rejects_malformed_bucket(bad):
    with pytest.raises(ValidationError):
        PredictRequest(frames=["a.jpg"], bucket=bad)


def test_smoke_uses_max_kept_probability():
    # Trigger tube (id 7) has the LOWER prob; reported value is the max (0.91).
    out = SimpleNamespace(
        is_positive=True,
        trigger_frame_index=3,
        details=_details([_tube(7, 0.62), _tube(2, 0.91)]),
    )
    resp = to_response(
        out, api_version="0.3.0", model_version="1.2.0", calibrated=True, verbose=False
    )
    dumped = resp.model_dump(exclude_unset=True)
    assert dumped == {
        "is_smoke": True,
        "probability": 0.91,
        "version": {"api": "0.3.0", "model": "1.2.0"},
    }


def test_negative_uses_max_kept_probability():
    out = SimpleNamespace(
        is_positive=False,
        trigger_frame_index=None,
        details=_details([_tube(1, 0.1), _tube(2, 0.41)]),
    )
    resp = to_response(
        out, api_version=None, model_version="1.2.0", calibrated=True, verbose=False
    )
    assert resp.probability == 0.41
    assert resp.is_smoke is False


def test_negative_no_tubes_is_zero_when_calibrated():
    out = SimpleNamespace(
        is_positive=False, trigger_frame_index=None, details=_details([])
    )
    resp = to_response(
        out, api_version=None, model_version="1.2.0", calibrated=True, verbose=False
    )
    assert resp.probability == 0.0


def test_uncalibrated_probability_is_null():
    out = SimpleNamespace(
        is_positive=False, trigger_frame_index=None, details=_details([_tube(1, None)])
    )
    resp = to_response(
        out, api_version=None, model_version=None, calibrated=False, verbose=False
    )
    assert resp.probability is None


def test_verbose_adds_details_block():
    out = SimpleNamespace(
        is_positive=True, trigger_frame_index=3, details=_details([_tube(7, 0.98)])
    )
    resp = to_response(
        out, api_version=None, model_version="1.2.0", calibrated=True, verbose=True
    )
    dumped = resp.model_dump(exclude_unset=True)
    assert dumped["details"]["decision"] == {
        "aggregation": "max_logit",
        "threshold": 0.5,
        "threshold_overridden": False,
        "packaged_threshold": None,
    }
    assert dumped["details"]["preprocessing"]["num_tube_candidates"] == 2
    assert dumped["details"]["tubes"][0]["tube_id"] == 7
    assert dumped["details"]["tubes"][0]["entries"][1]["bbox"] is None


def test_verbose_surfaces_threshold_override():
    out = SimpleNamespace(
        is_positive=True, trigger_frame_index=3, details=_details([_tube(7, 0.98)])
    )
    resp = to_response(
        out,
        api_version=None,
        model_version="1.2.0",
        calibrated=True,
        verbose=True,
        threshold_overridden=True,
        packaged_threshold=0.5,
    )
    decision = resp.model_dump(exclude_unset=True)["details"]["decision"]
    assert decision["threshold_overridden"] is True
    assert decision["packaged_threshold"] == 0.5


def test_to_response_includes_profiling_when_verbose():
    details = {
        "decision": {
            "aggregation": "max_logit",
            "threshold": 0.5,
            "trigger_tube_id": None,
        },
        "preprocessing": {
            "num_frames_input": 6,
            "num_truncated": 0,
            "padded_frame_indices": [],
        },
        "tubes": {"num_candidates": 0, "num_outside_roi": 0, "kept": []},
    }
    out = SimpleNamespace(is_positive=False, trigger_frame_index=None, details=details)
    profiling = {
        "stages_ms": {"s3_fetch": 1.0},
        "total_ms": 1.0,
        "n_frames": 6,
        "cache_hits": 4,
        "cache_misses": 2,
    }

    resp = to_response(
        out,
        api_version=None,
        model_version="1",
        calibrated=False,
        verbose=True,
        profiling=profiling,
    )
    assert resp.details.profiling == profiling

    # Omitted when not verbose, and harmless when profiling is None.
    resp2 = to_response(
        out,
        api_version=None,
        model_version="1",
        calibrated=False,
        verbose=False,
        profiling=profiling,
    )
    assert resp2.details is None
    resp3 = to_response(
        out,
        api_version=None,
        model_version="1",
        calibrated=False,
        verbose=True,
        profiling=None,
    )
    assert resp3.details.profiling is None


def test_request_roi_defaults_to_none():
    assert PredictRequest(frames=["a.jpg"]).roi_xyxyn is None


def test_request_accepts_valid_roi():
    req = PredictRequest(frames=["a.jpg"], roi_xyxyn=[0.1, 0.2, 0.3, 0.4])
    assert req.roi_xyxyn == (0.1, 0.2, 0.3, 0.4)


def test_request_accepts_whole_frame_roi():
    req = PredictRequest(frames=["a.jpg"], roi_xyxyn=[0.0, 0.0, 1.0, 1.0])
    assert req.roi_xyxyn == (0.0, 0.0, 1.0, 1.0)


@pytest.mark.parametrize(
    "roi",
    [
        [-0.1, 0.2, 0.3, 0.4],  # out of range low
        [0.1, 0.2, 0.3, 1.4],  # out of range high
        [0.3, 0.2, 0.1, 0.4],  # x_min >= x_max
        [0.1, 0.4, 0.3, 0.4],  # y_min >= y_max (zero height)
        [0.1, 0.2, 0.3],  # too short
        [0.1, 0.2, 0.3, 0.4, 0.5],  # too long
        ["a", 0.2, 0.3, 0.4],  # non-numeric
    ],
)
def test_request_rejects_invalid_roi(roi):
    with pytest.raises(ValidationError):
        PredictRequest(frames=["a.jpg"], roi_xyxyn=roi)


def test_verbose_details_map_num_tubes_outside_roi():
    details = _details([_tube(1, 0.9)])
    details["tubes"]["num_outside_roi"] = 3
    out = SimpleNamespace(is_positive=True, trigger_frame_index=3, details=details)
    resp = to_response(
        out, api_version=None, model_version="1", calibrated=True, verbose=True
    )
    assert resp.details.preprocessing.num_tubes_outside_roi == 3


def test_request_source_defaults_to_none():
    assert PredictRequest(frames=["a.jpg"]).source is None


@pytest.mark.parametrize("source", ["s3", "local"])
def test_request_accepts_source(source):
    assert PredictRequest(frames=["a.jpg"], source=source).source == source


def test_request_rejects_unknown_source():
    with pytest.raises(ValidationError):
        PredictRequest(frames=["a.jpg"], source="ftp")


def test_version_block_carries_nulls_independently():
    # Each identity is null on its own: api when not a release build, model
    # when the package is a legacy unstamped one.
    out = SimpleNamespace(
        is_positive=False, trigger_frame_index=None, details=_details([])
    )
    resp = to_response(
        out, api_version=None, model_version="1.2.0", calibrated=True, verbose=False
    )
    assert resp.version.api is None
    assert resp.version.model == "1.2.0"
    resp2 = to_response(
        out, api_version="0.3.0", model_version=None, calibrated=True, verbose=False
    )
    assert resp2.version.api == "0.3.0"
    assert resp2.version.model is None


def test_response_has_no_top_level_model_key():
    # The old model: {name, version} block is gone (breaking change, agreed
    # in the spec) — its content lives at version.model.
    out = SimpleNamespace(
        is_positive=False, trigger_frame_index=None, details=_details([])
    )
    resp = to_response(
        out, api_version="0.3.0", model_version="1.2.0", calibrated=True, verbose=False
    )
    assert "model" not in resp.model_dump(exclude_unset=True)


def test_verbose_details_num_tubes_outside_roi_is_strict():
    # Read strictly like num_candidates: core always emits the key, and a
    # missing key must fail loudly rather than silently report 0.
    details = _details([_tube(1, 0.9)])
    del details["tubes"]["num_outside_roi"]
    out = SimpleNamespace(is_positive=True, trigger_frame_index=3, details=details)
    with pytest.raises(KeyError):
        to_response(
            out, api_version=None, model_version="1", calibrated=True, verbose=True
        )


def test_compute_trigger_sets_top_level_trigger_frame_index():
    out = SimpleNamespace(
        is_positive=True, trigger_frame_index=3, details=_details([_tube(7, 0.98)])
    )
    resp = to_response(
        out,
        api_version=None,
        model_version="1.2.0",
        calibrated=True,
        verbose=False,
        compute_trigger=True,
    )
    dumped = resp.model_dump(exclude_unset=True)
    assert dumped["trigger_frame_index"] == 3
    assert "details" not in dumped


def test_compute_trigger_no_crossing_is_explicit_null():
    # Searched but nothing crossed: the key is present with an explicit null.
    out = SimpleNamespace(
        is_positive=False, trigger_frame_index=None, details=_details([])
    )
    resp = to_response(
        out,
        api_version=None,
        model_version="1.2.0",
        calibrated=True,
        verbose=False,
        compute_trigger=True,
    )
    dumped = resp.model_dump(exclude_unset=True)
    assert "trigger_frame_index" in dumped
    assert dumped["trigger_frame_index"] is None


def test_default_omits_trigger_frame_index():
    # Even when the core output carries a trigger, the flag gates exposure.
    out = SimpleNamespace(
        is_positive=True, trigger_frame_index=3, details=_details([_tube(7, 0.98)])
    )
    resp = to_response(
        out, api_version=None, model_version="1.2.0", calibrated=True, verbose=False
    )
    assert "trigger_frame_index" not in resp.model_dump(exclude_unset=True)


def test_compute_trigger_verbose_adds_trigger_details():
    out = SimpleNamespace(
        is_positive=True, trigger_frame_index=3, details=_details([_tube(7, 0.98)])
    )
    resp = to_response(
        out,
        api_version=None,
        model_version="1.2.0",
        calibrated=True,
        verbose=True,
        compute_trigger=True,
    )
    details = resp.model_dump(exclude_unset=True)["details"]
    assert details["decision"]["trigger_tube_id"] == 7
    assert details["tubes"][0]["first_crossing_frame"] == 3


def test_verbose_without_compute_trigger_omits_trigger_details():
    out = SimpleNamespace(
        is_positive=True, trigger_frame_index=3, details=_details([_tube(7, 0.98)])
    )
    resp = to_response(
        out, api_version=None, model_version="1.2.0", calibrated=True, verbose=True
    )
    details = resp.model_dump(exclude_unset=True)["details"]
    assert "trigger_tube_id" not in details["decision"]
    assert "first_crossing_frame" not in details["tubes"][0]

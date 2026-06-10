"""Tests for BboxTubeDetails and sub-models."""

import pytest
from pydantic import ValidationError

from temporal_model.core.details_schema import (
    BboxTubeDetails,
    Decision,
    KeptTube,
    KeptTubeEntry,
    Preprocessing,
    Tubes,
)


def _sample_details() -> BboxTubeDetails:
    return BboxTubeDetails(
        preprocessing=Preprocessing(
            num_frames_input=6,
            num_truncated=0,
            padded_frame_indices=[],
        ),
        tubes=Tubes(
            num_candidates=1,
            kept=[
                KeptTube(
                    tube_id=0,
                    start_frame=0,
                    end_frame=5,
                    logit=1.25,
                    probability=None,
                    first_crossing_frame=3,
                    entries=[
                        KeptTubeEntry(
                            frame_idx=0,
                            bbox=(0.5, 0.5, 0.1, 0.1),
                            is_gap=False,
                            confidence=0.9,
                        ),
                        KeptTubeEntry(
                            frame_idx=1,
                            bbox=None,
                            is_gap=True,
                            confidence=None,
                        ),
                    ],
                )
            ],
        ),
        decision=Decision(
            aggregation="max_logit",
            threshold=0.0,
            trigger_tube_id=0,
        ),
    )


def test_round_trip_via_model_dump_and_validate() -> None:
    original = _sample_details()
    dumped = original.model_dump()
    parsed = BboxTubeDetails.model_validate(dumped)
    assert parsed == original


def test_decision_rejects_unknown_aggregation() -> None:
    with pytest.raises(ValidationError):
        Decision(aggregation="bogus", threshold=0.0, trigger_tube_id=None)  # type: ignore[arg-type]


def test_tube_entry_rejects_wrong_length_bbox() -> None:
    with pytest.raises(ValidationError):
        KeptTubeEntry(
            frame_idx=0,
            bbox=(0.1, 0.2, 0.3),  # type: ignore[arg-type]
            is_gap=False,
            confidence=0.5,
        )


def test_empty_sequence_shape_validates() -> None:
    details = BboxTubeDetails(
        preprocessing=Preprocessing(
            num_frames_input=0, num_truncated=0, padded_frame_indices=[]
        ),
        tubes=Tubes(num_candidates=0, kept=[]),
        decision=Decision(aggregation="max_logit", threshold=0.0, trigger_tube_id=None),
    )
    assert details.model_dump()["tubes"]["kept"] == []


def test_no_tubes_kept_shape_validates() -> None:
    details = BboxTubeDetails(
        preprocessing=Preprocessing(
            num_frames_input=5, num_truncated=0, padded_frame_indices=[]
        ),
        tubes=Tubes(num_candidates=3, kept=[]),
        decision=Decision(aggregation="max_logit", threshold=0.0, trigger_tube_id=None),
    )
    assert details.tubes.num_candidates == 3
    assert details.decision.trigger_tube_id is None


def test_tubes_num_outside_roi_defaults_to_zero():
    assert Tubes(num_candidates=2, kept=[]).num_outside_roi == 0


def test_details_parses_legacy_dump_without_num_outside_roi():
    # Dumps serialized before the ROI feature must still validate.
    details = BboxTubeDetails(
        preprocessing=Preprocessing(
            num_frames_input=1, num_truncated=0, padded_frame_indices=[]
        ),
        tubes=Tubes(num_candidates=0, kept=[]),
        decision=Decision(aggregation="max_logit", threshold=0.0, trigger_tube_id=None),
    )
    dump = details.model_dump()
    del dump["tubes"]["num_outside_roi"]
    parsed = BboxTubeDetails.model_validate(dump)
    assert parsed.tubes.num_outside_roi == 0

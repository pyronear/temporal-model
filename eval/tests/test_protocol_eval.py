"""Unit tests for protocol_eval record + metrics helpers."""

import math
from pathlib import Path

from temporal_model.core.protocol import Frame, TemporalModelOutput
from temporal_model.eval.protocol_eval import (
    SequenceRecord,
    build_record,
    compute_metrics,
)


def _rec(label, is_positive, *, score=0.0, trigger=None, ttd=None, num_kept=0):
    return SequenceRecord(
        sequence_id=f"seq_{label}_{is_positive}",
        label=label,
        is_positive=is_positive,
        trigger_frame_index=trigger,
        score=score,
        num_tubes_kept=num_kept,
        tube_logits=[],
        ttd_frames=ttd,
    )


def test_compute_metrics_all_correct():
    records = [
        _rec("smoke", True, score=1.0, trigger=5, ttd=5),
        _rec("smoke", True, score=0.9, trigger=3, ttd=3),
        _rec("fp", False, score=-1.0),
        _rec("fp", False, score=-0.5),
    ]

    m = compute_metrics("my-model", records)

    assert m["model_name"] == "my-model"
    assert m["num_sequences"] == 4
    assert m["tp"] == 2
    assert m["fp"] == 0
    assert m["fn"] == 0
    assert m["tn"] == 2
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0
    assert m["fpr"] == 0.0
    assert m["mean_ttd_frames"] == 4.0
    assert m["median_ttd_frames"] == 4.0


def test_compute_metrics_all_wrong():
    records = [
        _rec("smoke", False, score=-1.0),
        _rec("fp", True, score=0.9, trigger=1),
    ]

    m = compute_metrics("my-model", records)

    assert m["tp"] == 0
    assert m["fp"] == 1
    assert m["fn"] == 1
    assert m["tn"] == 0
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0
    assert m["fpr"] == 1.0
    assert m["mean_ttd_frames"] is None
    assert m["median_ttd_frames"] is None


def test_compute_metrics_rounds_to_four_decimals():
    records = [
        _rec("smoke", True, score=1.0, trigger=0, ttd=1),
        _rec("smoke", False, score=-1.0),
        _rec("smoke", False, score=-1.0),
        _rec("fp", True, score=0.5, trigger=0),
        _rec("fp", False, score=-1.0),
    ]

    m = compute_metrics("my-model", records)

    assert m["precision"] == round(1 / 2, 4)
    assert m["recall"] == round(1 / 3, 4)
    assert m["f1"] == round(2 * (1 / 2) * (1 / 3) / ((1 / 2) + (1 / 3)), 4)
    assert m["fpr"] == round(1 / 2, 4)
    assert m["mean_ttd_frames"] == 1.0
    assert m["median_ttd_frames"] == 1.0


def test_compute_metrics_auc_handles_minus_inf_scores():
    records = [
        _rec("smoke", False, score=-math.inf),
        _rec("smoke", True, score=2.0, trigger=0),
        _rec("fp", False, score=-math.inf),
        _rec("fp", False, score=-1.0),
    ]

    m = compute_metrics("my-model", records)

    assert 0.0 <= m["pr_auc"] <= 1.0
    assert 0.0 <= m["roc_auc"] <= 1.0
    assert not math.isnan(m["pr_auc"])
    assert not math.isnan(m["roc_auc"])


def test_compute_metrics_auc_zero_when_only_one_class():
    records = [
        _rec("fp", False, score=-1.0),
        _rec("fp", True, score=1.0, trigger=0),
    ]

    m = compute_metrics("my-model", records)

    assert m["roc_auc"] == 0.0
    assert m["pr_auc"] == 0.0


def test_compute_metrics_empty_records():
    m = compute_metrics("my-model", [])

    assert m["num_sequences"] == 0
    assert m["tp"] == m["fp"] == m["fn"] == m["tn"] == 0
    assert m["precision"] == m["recall"] == m["f1"] == m["fpr"] == 0.0
    assert m["mean_ttd_frames"] is None
    assert m["median_ttd_frames"] is None
    assert m["pr_auc"] == 0.0
    assert m["roc_auc"] == 0.0


def test_compute_metrics_ttd_median_three_values():
    records = [
        _rec("smoke", True, score=1.0, trigger=0, ttd=1),
        _rec("smoke", True, score=1.0, trigger=0, ttd=3),
        _rec("smoke", True, score=1.0, trigger=0, ttd=5),
    ]

    m = compute_metrics("my-model", records)

    assert m["mean_ttd_frames"] == 3.0
    assert m["median_ttd_frames"] == 3.0


def _frame(stem="f", ts=None):
    return Frame(frame_id=stem, image_path=Path(f"/tmp/{stem}.jpg"), timestamp=ts)


# ── build_record ────────────────────────────────────────────────────────


def _kept_tube(tube_id: int, logit: float) -> dict:
    return {
        "tube_id": tube_id,
        "start_frame": 0,
        "end_frame": 5,
        "logit": logit,
        "probability": None,
        "first_crossing_frame": None,
        "entries": [],
    }


def _details(kept: list[dict], *, trigger_tube_id: int | None, **extra) -> dict:
    base = {
        "preprocessing": {
            "num_frames_input": 6,
            "num_truncated": 0,
            "padded_frame_indices": [],
        },
        "tubes": {"num_candidates": len(kept), "kept": kept},
        "decision": {
            "aggregation": "max_logit",
            "threshold": 0.0,
            "trigger_tube_id": trigger_tube_id,
        },
    }
    base.update(extra)
    return base


def test_build_record_extracts_score_and_tube_logits_from_details():
    kept = [_kept_tube(i, logit) for i, logit in enumerate([1.5, 0.3, -0.2])]
    output = TemporalModelOutput(
        is_positive=True,
        trigger_frame_index=0,
        details=_details(kept, trigger_tube_id=0),
    )

    rec = build_record(
        sequence_dir=Path("/data/wildfire/seq_a"),
        label="smoke",
        frames=[_frame("f0")],
        output=output,
    )

    assert rec.sequence_id == "seq_a"
    assert rec.label == "smoke"
    assert rec.is_positive is True
    assert rec.trigger_frame_index == 0
    assert rec.tube_logits == [1.5, 0.3, -0.2]
    assert rec.num_tubes_kept == 3
    assert rec.score == 1.5  # max of the three logits


def test_build_record_empty_tube_logits_yields_minus_inf_score():
    output = TemporalModelOutput(
        is_positive=False,
        trigger_frame_index=None,
        details=_details([], trigger_tube_id=None),
    )

    rec = build_record(
        sequence_dir=Path("/data/fp/seq_b"),
        label="fp",
        frames=[_frame("f0")],
        output=output,
    )

    assert rec.score == float("-inf")
    assert rec.num_tubes_kept == 0
    assert rec.tube_logits == []


def test_build_record_copies_details_dict():
    """build_record should snapshot output.details, not alias it."""
    details = _details([_kept_tube(0, 0.9)], trigger_tube_id=0, extra="foo")
    output = TemporalModelOutput(
        is_positive=True, trigger_frame_index=0, details=details
    )

    rec = build_record(
        sequence_dir=Path("/data/wildfire/seq_c"),
        label="smoke",
        frames=[_frame("f0")],
        output=output,
    )

    details["extra"] = "mutated"
    assert rec.details["extra"] == "foo"


# ── compute_metrics TTD filter regression ────────────────────────────────


def test_compute_metrics_ignores_ttd_for_non_tp_records():
    """TTD is filtered to TPs even if non-TP records carry a ttd_frames.

    Protects against misuse where a caller constructs a record with
    label!="smoke" or is_positive=False but a non-None ttd_frames.
    """
    records = [
        _rec("smoke", True, score=1.0, trigger=0, ttd=1),
        _rec("smoke", False, score=-1.0, ttd=99),  # FN with bogus ttd
        _rec("fp", True, score=0.5, trigger=0, ttd=88),  # FP with bogus ttd
        _rec("fp", False, score=-1.0, ttd=77),  # TN with bogus ttd
    ]

    m = compute_metrics("my-model", records)

    # Only the one TP record's ttd should be counted
    assert m["mean_ttd_frames"] == 1.0
    assert m["median_ttd_frames"] == 1.0

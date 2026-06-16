import json
from pathlib import Path

import pandas as pd

from temporal_model.triage.report import MODEL_DIR, write_triage_report
from temporal_model.triage.score import ScoredSequence
from temporal_model.triage.store import FrameRef, SequenceMeta


def _scored(key, sid, score, bucket):
    meta = SequenceMeta(
        key=key,
        sequence_id=sid,
        camera_name="cam",
        organization_name="org",
        started_at="2026-06-01T10:00:00",
        frames=[FrameRef(file="images/detection_1.jpg", detection_id=1)],
    )
    return ScoredSequence(
        key=key,
        sequence_id=sid,
        score=score,
        bucket=bucket,
        meta=meta,
        details={
            "tubes": {"kept": [{"tube_id": 0, "logit": 1.0, "probability": score}]},
            "decision": {"aggregation": "logistic", "threshold": 0.5},
        },
        trigger_frame_index=0 if bucket == "review" else None,
        frame_paths=[
            Path("data/01_raw/sequences/org/cam/seq_1/images/detection_1.jpg")
        ],
    )


def test_write_triage_report_emits_contract_and_worklists(tmp_path):
    scored = [
        _scored("pyro-annotator_1", 1, 0.92, "review"),
        _scored("pyro-annotator_2", 2, 0.10, "unlabeled"),
    ]
    write_triage_report(
        tmp_path,
        scored,
        dropped=[{"sequence_id": "pyro-annotator_9", "reason": "no_images"}],
        threshold=0.35,
        model_config={"model_version": "0.1.0"},
    )
    out = tmp_path / "pyro-annotator" / MODEL_DIR

    rows = json.loads((out / "results.json").read_text())
    assert {r["key"]: r["triage_bucket"] for r in rows} == {
        "pyro-annotator_1": "review",
        "pyro-annotator_2": "unlabeled",
    }
    assert {r["key"]: r["decision"] for r in rows} == {
        "pyro-annotator_1": "keep",
        "pyro-annotator_2": "discard",
    }
    assert (out / "details" / "pyro-annotator_1.json").exists()
    assert (out / "sequences" / "pyro-annotator_1.json").exists()
    assert json.loads((out / "model_config.json").read_text())["threshold"] == 0.35
    assert json.loads((out / "dropped.json").read_text())[0]["reason"] == "no_images"

    unlabeled = json.loads((out / "unlabeled.json").read_text())
    assert unlabeled["threshold"] == 0.35
    assert unlabeled["sequence_ids"] == [2]
    assert unlabeled["bulk_payload"]["false_positive_type"] == "unlabeled"
    assert unlabeled["bulk_payload"]["sequence_ids"] == [2]

    review = json.loads((out / "review.json").read_text())
    assert review["sequence_ids"] == [1]
    assert review["items"][0]["score"] == 0.92


def test_results_parquet_mirrors_results_json(tmp_path):
    scored = [
        _scored("pyro-annotator_1", 1, 0.92, "review"),
        _scored("pyro-annotator_2", 2, 0.10, "unlabeled"),
    ]
    write_triage_report(
        tmp_path, scored, dropped=[], threshold=0.35, model_config={}
    )
    out = tmp_path / "pyro-annotator" / MODEL_DIR
    df = pd.read_parquet(out / "results.parquet")
    rows = json.loads((out / "results.json").read_text())
    assert len(df) == len(rows) == 2
    assert dict(zip(df["key"], df["triage_bucket"], strict=True)) == {
        "pyro-annotator_1": "review",
        "pyro-annotator_2": "unlabeled",
    }

"""End-to-end smoke test for temporal_model.eval.evaluate.

Monkeypatches BboxTubeTemporalModel so the driver never touches YOLO
or a real classifier — purely exercises the iteration / aggregation /
output-writing path.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from temporal_model.core import model as model_module
from temporal_model.core.protocol import Frame, TemporalModelOutput
from temporal_model.eval import evaluate as evaluate_packaged


def _write_jpg(path: Path) -> None:
    """Write a minimal 1x1 JPEG placeholder.

    Driver never decodes the image (predict is monkeypatched), so a
    plausible-looking 1-byte payload is fine.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\xff")


def _make_sequence(split_dir: Path, category: str, seq_name: str, n_frames: int):
    seq_dir = split_dir / category / seq_name
    for i in range(n_frames):
        _write_jpg(seq_dir / "images" / f"cam_2024-01-01T10-00-{i:02d}.jpg")
    return seq_dir


class _FakeModel:
    """Stand-in for BboxTubeTemporalModel.

    load_sequence: defers to pyrocore's default (Frame per path).
    predict: returns a canned positive-or-not output based on the
    seq name prefix.
    """

    last_compute_trigger = None

    def load_sequence(self, frames):
        return [Frame(frame_id=p.stem, image_path=p, timestamp=None) for p in frames]

    def predict(self, frames, *, compute_trigger=False):
        type(self).last_compute_trigger = compute_trigger
        # Decide based on how many frames we got — purely to vary outputs.
        is_pos = len(frames) >= 3
        kept = (
            [
                {
                    "tube_id": 0,
                    "start_frame": 0,
                    "end_frame": len(frames) - 1,
                    "logit": 2.5,
                    "probability": None,
                    "first_crossing_frame": len(frames) - 1,
                    "entries": [],
                }
            ]
            if is_pos
            else []
        )
        return TemporalModelOutput(
            is_positive=is_pos,
            trigger_frame_index=(len(frames) - 1) if is_pos else None,
            details={
                "preprocessing": {
                    "num_frames_input": len(frames),
                    "num_truncated": 0,
                    "padded_frame_indices": [],
                },
                "tubes": {
                    "num_candidates": 1 if is_pos else 0,
                    "kept": kept,
                },
                "decision": {
                    "aggregation": "max_logit",
                    "threshold": 0.5,
                    "trigger_tube_id": 0 if is_pos else None,
                },
            },
        )


def test_evaluate_packaged_writes_expected_outputs(tmp_path, monkeypatch):
    sequences_dir = tmp_path / "sequences"
    output_dir = tmp_path / "out"
    _make_sequence(sequences_dir, "wildfire", "wf_seq_a", n_frames=4)  # TP
    _make_sequence(sequences_dir, "wildfire", "wf_seq_b", n_frames=2)  # FN
    _make_sequence(sequences_dir, "fp", "fp_seq_c", n_frames=4)  # FP
    _make_sequence(sequences_dir, "fp", "fp_seq_d", n_frames=1)  # TN

    monkeypatch.setattr(
        model_module.BboxTubeTemporalModel,
        "from_archive",
        classmethod(lambda cls, path, device=None: _FakeModel()),
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_packaged.py",
            "--model-zip",
            str(tmp_path / "placeholder.zip"),
            "--sequences-dir",
            str(sequences_dir),
            "--output-dir",
            str(output_dir),
            "--model-name",
            "fake-variant-fake-split",
        ],
    )

    evaluate_packaged.main()

    # Eval must request the trigger output — TTD needs trigger_frame_index.
    assert _FakeModel.last_compute_trigger is True

    assert (output_dir / "metrics.json").is_file()
    assert (output_dir / "predictions.json").is_file()
    assert (output_dir / "dropped.json").is_file()
    assert (output_dir / "confusion_matrix.png").is_file()
    assert (output_dir / "confusion_matrix_normalized.png").is_file()
    assert (output_dir / "pr_curve.png").is_file()
    assert (output_dir / "roc_curve.png").is_file()

    metrics = json.loads((output_dir / "metrics.json").read_text())
    assert metrics["model_name"] == "fake-variant-fake-split"
    assert metrics["num_sequences"] == 4
    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["tn"] == 1
    assert "pr_auc" in metrics and "roc_auc" in metrics

    predictions = json.loads((output_dir / "predictions.json").read_text())
    assert len(predictions) == 4
    assert {p["sequence_id"] for p in predictions} == {
        "wf_seq_a",
        "wf_seq_b",
        "fp_seq_c",
        "fp_seq_d",
    }

    # Predictions now carry the full per-tube details so downstream
    # diagnostics (e.g. the error-analysis notebook) can inspect every
    # tube the model saw, not just the winner.
    positive_records = [p for p in predictions if p["is_positive"]]
    assert positive_records, "at least one positive record expected"
    a_positive = positive_records[0]
    assert a_positive["num_tubes_total"] == 1
    assert a_positive["trigger_tube_id"] == 0
    assert a_positive["threshold"] == 0.5
    assert isinstance(a_positive["kept_tubes"], list)
    assert len(a_positive["kept_tubes"]) == a_positive["num_tubes_kept"]
    assert a_positive["kept_tubes"][0]["logit"] == 2.5


def test_evaluate_packaged_writes_viewer_artifacts(tmp_path, monkeypatch):
    sequences_dir = tmp_path / "sequences"
    output_dir = tmp_path / "out"
    _make_sequence(sequences_dir, "wildfire", "wf_seq_a", n_frames=4)  # TP
    _make_sequence(sequences_dir, "fp", "fp_seq_c", n_frames=4)  # FP

    monkeypatch.setattr(
        model_module.BboxTubeTemporalModel,
        "from_archive",
        classmethod(lambda cls, path, device=None: _FakeModel()),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_packaged.py",
            "--model-zip",
            str(tmp_path / "placeholder.zip"),
            "--sequences-dir",
            str(sequences_dir),
            "--output-dir",
            str(output_dir),
            "--model-name",
            "vit_dinov2_finetune-train",
            "--source",
            "train",
        ],
    )
    evaluate_packaged.main()

    # per-sequence details + view records
    assert (output_dir / "details" / "wf_seq_a.json").is_file()
    assert (output_dir / "sequences" / "wf_seq_a.json").is_file()
    view = json.loads((output_dir / "sequences" / "wf_seq_a.json").read_text())
    assert view["source"] == "train"
    assert view["label"] == "smoke"
    assert len(view["frames"]) == 4

    # results table (json + parquet), one row per sequence
    assert (output_dir / "results.parquet").is_file()
    # model_config.json is always emitted (here {} since the fake zip is absent)
    assert (output_dir / "model_config.json").is_file()
    assert json.loads((output_dir / "model_config.json").read_text()) == {}
    rows = json.loads((output_dir / "results.json").read_text())
    by_key = {r["key"]: r for r in rows}
    assert set(by_key) == {"wf_seq_a", "fp_seq_c"}
    assert by_key["wf_seq_a"]["decision"] == "keep"
    assert by_key["wf_seq_a"]["outcome"] == "kept-smoke"
    assert by_key["wf_seq_a"]["source"] == "train"
    assert by_key["wf_seq_a"]["num_tubes_kept"] == 1  # FakeModel keeps one tube
    assert by_key["fp_seq_c"]["outcome"] == "kept-fp"  # FakeModel keeps 4-frame seqs
    df = pd.read_parquet(output_dir / "results.parquet")
    assert len(df) == 2


def test_evaluate_packaged_strict_errors_abort(tmp_path, monkeypatch):
    """Any predict() exception must bubble out — strict policy."""
    sequences_dir = tmp_path / "sequences"
    output_dir = tmp_path / "out"
    _make_sequence(sequences_dir, "wildfire", "wf_seq", n_frames=3)

    class _BrokenModel(_FakeModel):
        def predict(self, frames, *, compute_trigger=False):
            raise RuntimeError("simulated inference crash")

    monkeypatch.setattr(
        model_module.BboxTubeTemporalModel,
        "from_archive",
        classmethod(lambda cls, path, device=None: _BrokenModel()),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_packaged.py",
            "--model-zip",
            str(tmp_path / "placeholder.zip"),
            "--sequences-dir",
            str(sequences_dir),
            "--output-dir",
            str(output_dir),
            "--model-name",
            "fake",
        ],
    )
    with pytest.raises(RuntimeError, match="simulated inference crash"):
        evaluate_packaged.main()


def test_evaluate_packaged_skips_sequences_without_images(tmp_path, monkeypatch):
    """No images/ subdir → logged under dropped.json, not evaluated."""
    sequences_dir = tmp_path / "sequences"
    output_dir = tmp_path / "out"
    _make_sequence(sequences_dir, "wildfire", "wf_seq_ok", n_frames=3)
    (sequences_dir / "fp" / "fp_seq_bad").mkdir(parents=True)

    monkeypatch.setattr(
        model_module.BboxTubeTemporalModel,
        "from_archive",
        classmethod(lambda cls, path, device=None: _FakeModel()),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_packaged.py",
            "--model-zip",
            str(tmp_path / "placeholder.zip"),
            "--sequences-dir",
            str(sequences_dir),
            "--output-dir",
            str(output_dir),
            "--model-name",
            "fake",
        ],
    )
    evaluate_packaged.main()

    dropped = json.loads((output_dir / "dropped.json").read_text())
    assert len(dropped) == 1
    assert dropped[0]["sequence_id"] == "fp_seq_bad"
    assert dropped[0]["reason"] == "no_images"

    metrics = json.loads((output_dir / "metrics.json").read_text())
    assert metrics["num_sequences"] == 1


def _write_store_seq(root, key, label, n_frames):
    seq = root / "org-a" / "cam-1" / key
    (seq / "images").mkdir(parents=True)
    frames = []
    for i in range(n_frames):
        (seq / "images" / f"f{i}.jpg").write_bytes(b"\xff")
        frames.append(
            {"file": f"images/f{i}.jpg", "detection_id": None, "created_at": None}
        )
    meta = {
        "key": key,
        "sequence_id": key,
        "source": "pyro-annotator",
        "label": label,
        "label_detail": None,
        "label_source": "pyro_annotator_folder",
        "frames": frames,
        "camera_id": 1,
        "camera_name": "cam-1",
        "organization_id": 7,
        "organization_name": "org-a",
        "started_at": "2026-05-19T14:10:01",
    }
    (seq / "meta.json").write_text(json.dumps(meta))
    return seq


def test_evaluate_store_source_excludes_unknown_from_metrics(tmp_path, monkeypatch):
    store_dir = tmp_path / "pyro"
    output_dir = tmp_path / "out"
    _write_store_seq(store_dir, "seq_smoke", "smoke", n_frames=4)  # TP
    _write_store_seq(store_dir, "seq_unknown", "unknown", n_frames=4)  # not labeled

    monkeypatch.setattr(
        model_module.BboxTubeTemporalModel,
        "from_archive",
        classmethod(lambda cls, path, device=None: _FakeModel()),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_packaged.py",
            "--model-zip",
            str(tmp_path / "placeholder.zip"),
            "--sequences-dir",
            str(store_dir),
            "--output-dir",
            str(output_dir),
            "--model-name",
            "vit_dinov2_finetune-pyro-annotator",
            "--source",
            "pyro-annotator",
            "--store",
        ],
    )
    evaluate_packaged.main()

    metrics = json.loads((output_dir / "metrics.json").read_text())
    assert metrics["num_sequences"] == 1  # unknown excluded from metrics

    rows = json.loads((output_dir / "results.json").read_text())
    by_key = {r["key"]: r for r in rows}
    assert set(by_key) == {"seq_smoke", "seq_unknown"}  # both viewable
    assert by_key["seq_unknown"]["outcome"] == "n/a"
    assert by_key["seq_smoke"]["organization_name"] == "org-a"
    assert by_key["seq_smoke"]["camera_name"] == "cam-1"

    view = json.loads((output_dir / "sequences" / "seq_unknown.json").read_text())
    assert view["organization_name"] == "org-a"

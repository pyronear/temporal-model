"""Tests for collect_pipeline_records (model + data mocked)."""

from pathlib import Path
from unittest.mock import patch

from temporal_model.train.package_predict import collect_pipeline_records


class _FakeOut:
    def __init__(self, kept: list[dict]) -> None:
        self.details = {"tubes": {"kept": kept}}


class _FakeModel:
    """Duck-types BboxTubeTemporalModel.predict_sequence."""

    def predict_sequence(self, frame_paths: list[Path]):
        return _FakeOut(
            [{"logit": 1.0, "start_frame": 0, "end_frame": 1, "entries": []}]
        )


def test_collect_pipeline_records_labels_and_tubes(tmp_path: Path) -> None:
    # Two sequences: one wildfire (smoke), one fp.
    seqs = [("wildfire/seqA", "smoke"), ("fp/seqB", "fp")]
    fake_seq_dirs = [tmp_path / s for s, _ in seqs]
    for d in fake_seq_dirs:
        (d / "images").mkdir(parents=True)
        (d / "images" / "frame.jpg").write_bytes(b"x")

    def fake_list_sequences(_raw):
        return fake_seq_dirs

    def fake_is_wf(seq_dir):
        return "wildfire" in str(seq_dir)

    def fake_sorted_frames(seq_dir):
        return [seq_dir / "images" / "frame.jpg"]

    with (
        patch(
            "temporal_model.train.package_predict.list_sequences", fake_list_sequences
        ),
        patch("temporal_model.train.package_predict.is_wf_sequence", fake_is_wf),
        patch(
            "temporal_model.train.package_predict.get_sorted_frames",
            fake_sorted_frames,
        ),
    ):
        records = collect_pipeline_records(model=_FakeModel(), raw_dir=tmp_path)

    assert {r["label"] for r in records} == {"smoke", "fp"}
    assert all(r["kept_tubes"] for r in records)

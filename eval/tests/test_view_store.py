import json

from temporal_model.eval.view_store import SequenceView, write_sequence_view


def test_write_sequence_view_roundtrips(tmp_path):
    view = SequenceView(
        key="wf_seq_a",
        source="train",
        label="smoke",
        organization_name=None,
        camera_name=None,
        started_at=None,
        frames=["data/01_raw/datasets/train/wildfire/wf_seq_a/images/f0.jpg"],
    )
    out_dir = tmp_path / "sequences"
    write_sequence_view(out_dir, view)
    payload = json.loads((out_dir / "wf_seq_a.json").read_text())
    assert payload["key"] == "wf_seq_a"
    assert payload["source"] == "train"
    assert payload["label"] == "smoke"
    assert payload["frames"] == [
        "data/01_raw/datasets/train/wildfire/wf_seq_a/images/f0.jpg"
    ]

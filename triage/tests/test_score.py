from dataclasses import dataclass

from temporal_model.triage.score import (
    ScoredSequence,
    bucket_for,
    score_sequences,
    sequence_score,
)
from temporal_model.triage.store import (
    FrameRef,
    SequenceMeta,
    sequence_dir,
    write_meta,
)


def _details(probs):
    return {"tubes": {"kept": [{"probability": p} for p in probs]}}


def test_sequence_score_is_max_tube_probability():
    assert sequence_score(_details([0.1, 0.8, 0.3])) == 0.8


def test_sequence_score_zero_when_no_tubes():
    assert sequence_score(_details([])) == 0.0


def test_sequence_score_zero_when_all_probabilities_none():
    assert sequence_score({"tubes": {"kept": [{"probability": None}]}}) == 0.0


def test_bucket_split_at_threshold():
    assert bucket_for(0.35, 0.35) == "review"  # boundary is inclusive (>=)
    assert bucket_for(0.34, 0.35) == "unlabeled"
    assert bucket_for(0.9, 0.35) == "review"


@dataclass
class _Output:
    is_positive: bool
    trigger_frame_index: int | None
    details: dict


class _FakeModel:
    def __init__(self, prob):
        self._prob = prob

    def predict(self, frames, compute_trigger=True):
        return _Output(self._prob >= 0.5, 0, _details([self._prob]))


def test_score_sequences_classifies_and_carries_through(tmp_path):
    meta = SequenceMeta(
        key="pyro-annotator_1",
        sequence_id=1,
        camera_name="cam",
        organization_name="org",
        frames=[FrameRef(file="images/detection_1.jpg", detection_id=1)],
    )
    seq_dir = sequence_dir(tmp_path, meta)
    (seq_dir / "images").mkdir(parents=True)
    (seq_dir / "images/detection_1.jpg").write_bytes(b"x")
    write_meta(seq_dir, meta)

    scored, dropped = score_sequences(_FakeModel(0.8), tmp_path, threshold=0.35)
    assert dropped == []
    assert len(scored) == 1
    s = scored[0]
    assert isinstance(s, ScoredSequence)
    assert s.key == "pyro-annotator_1"
    assert s.score == 0.8
    assert s.bucket == "review"

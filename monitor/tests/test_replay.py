import json
import subprocess
from pathlib import Path

from temporal_model.monitor.cli import _parse_args
from temporal_model.monitor.replay import SCORE_TOLERANCE, run_replay
from temporal_model.monitor.stack import StackError
from temporal_model.monitor.store import (
    FrameMeta,
    SequenceMeta,
    sequence_dir,
    write_meta,
)

VERBOSE_DETAILS = {
    "decision": {"aggregation": "logistic", "threshold": 0.52},
    "preprocessing": {
        "num_frames_input": 4,
        "num_truncated": 0,
        "padded_frame_indices": [],
        "num_tube_candidates": 1,
        "num_tubes_outside_roi": 0,
    },
    "tubes": [
        {
            "tube_id": 0,
            "start_frame": 0,
            "end_frame": 3,
            "logit": 3.41,
            "probability": 0.93,
            "entries": [
                {
                    "frame_idx": 0,
                    "bbox": [0.2, 0.2, 0.2, 0.2],
                    "is_gap": False,
                    "confidence": 0.8,
                }
            ],
        }
    ],
}


def store_sequence(
    store: Path,
    sequence_id: int,
    *,
    api_version: str | None = "0.3.1",
    model_version: str | None = "0.1.0",
    score: float | None = 0.93,
    n_frames: int = 4,
    with_images: bool = True,
) -> SequenceMeta:
    meta = SequenceMeta(
        key=f"alert-api_{sequence_id}",
        sequence_id=sequence_id,
        label="smoke",
        label_detail="wildfire_smoke",
        camera_name="cam-01",
        organization_name="sis-67",
        started_at="2026-05-15T13:08:18",
        temporal_model_score=score,
        temporal_model_version=model_version,
        temporal_api_version=api_version,
        frames=[
            FrameMeta(
                file=f"images/detection_{i}.jpg",
                detection_id=i,
                created_at=f"2026-05-15T13:{i:02d}:00",
                bucket_key=f"cam/seq{sequence_id}-f{i}.jpg",
                bbox="[(0.1,0.1,0.3,0.3,0.9)]",
            )
            for i in range(n_frames)
        ],
    )
    seq_dir = sequence_dir(store, meta)
    write_meta(seq_dir, meta)
    if with_images:
        (seq_dir / "images").mkdir(exist_ok=True)
        for i in range(n_frames):
            (seq_dir / "images" / f"detection_{i}.jpg").write_bytes(b"jpg")
    return meta


class FakeStack:
    instances: list["FakeStack"] = []
    fail_versions: set[str] = set()

    def __init__(self, compose_file, version):
        self.version = version
        self.uploaded: list[dict] = []
        self.up_called = self.down_called = False
        FakeStack.instances.append(self)

    def up(self):
        self.up_called = True
        if self.version in FakeStack.fail_versions:
            raise subprocess.CalledProcessError(1, ["docker"])

    def down(self):
        self.down_called = True

    def wait_healthy(self, **kwargs):
        return {"status": "ok", "model_loaded": True, "model_version": "0.1.0"}

    def upload_frames(self, files_by_key):
        self.uploaded.append(dict(files_by_key))


def fake_predict_ok(frames, roi_xyxyn):
    return {
        "is_smoke": True,
        "probability": 0.93,
        "version": {"api": "0.3.1", "model": "0.1.0"},
        "details": VERBOSE_DETAILS,
    }


def run(store, out, predict=fake_predict_ok):
    FakeStack.instances = []
    return run_replay(
        store_dir=store,
        output_dir=out,
        compose_file=Path("dc.yml"),
        stack_factory=FakeStack,
        predict=predict,
    )


def test_happy_path_writes_org_tree(tmp_path):
    store, out = tmp_path / "store", tmp_path / "out"
    store_sequence(store, 1)
    summary = run(store, out)
    assert summary["replayed"] == 1
    assert summary["mismatched"] == 0
    rows = json.loads(
        (out / "sis-67" / "vit_dinov2_finetune" / "results.json").read_text()
    )
    assert rows[0]["replay_matches"] is True
    assert rows[0]["probability"] == 0.93
    stack = FakeStack.instances[0]
    assert stack.up_called and stack.down_called
    # 4 distinct keys uploaded once
    assert len(stack.uploaded[0]) == 4
    view = json.loads(
        (
            out / "sis-67" / "vit_dinov2_finetune" / "sequences" / "alert-api_1.json"
        ).read_text()
    )
    # viewer frames = the kept (replayed) frames, relative to monitor/
    assert view["frames"] == [
        f"data/01_raw/sequences/sis-67/cam-01/seq_1/images/detection_{i}.jpg"
        for i in range(4)
    ]


def test_groups_by_api_version_one_stack_each(tmp_path):
    store, out = tmp_path / "store", tmp_path / "out"
    store_sequence(store, 1, api_version="0.3.0")
    store_sequence(store, 2, api_version="0.3.1")
    store_sequence(store, 3, api_version="0.3.1")
    run(store, out)
    assert sorted(s.version for s in FakeStack.instances) == ["0.3.0", "0.3.1"]


def test_drop_reasons(tmp_path):
    store, out = tmp_path / "store", tmp_path / "out"
    store_sequence(store, 1, api_version=None)  # no_temporal_version
    store_sequence(store, 2, n_frames=2)  # too_few_frames
    store_sequence(store, 3, with_images=False)  # no_images
    store_sequence(store, 4, model_version="9.9.9")  # model_version_mismatch
    store_sequence(store, 5)  # ok
    summary = run(store, out)
    assert summary["replayed"] == 1
    dropped = json.loads(
        (out / "sis-67" / "vit_dinov2_finetune" / "dropped.json").read_text()
    )
    reasons = {d["sequence_id"]: d["reason"] for d in dropped}
    assert reasons == {
        "alert-api_1": "no_temporal_version",
        "alert-api_2": "too_few_frames",
        "alert-api_3": "no_images",
        "alert-api_4": "model_version_mismatch",
    }


def test_image_pull_failure_drops_whole_group(tmp_path):
    store, out = tmp_path / "store", tmp_path / "out"
    store_sequence(store, 1, api_version="0.0.9")
    FakeStack.fail_versions = {"0.0.9"}
    try:
        summary = run(store, out)
    finally:
        FakeStack.fail_versions = set()
    assert summary["replayed"] == 0
    dropped = json.loads(
        (out / "sis-67" / "vit_dinov2_finetune" / "dropped.json").read_text()
    )
    assert dropped[0]["reason"] == "image_pull_failed"


def test_predict_failure_drops_sequence_and_continues(tmp_path):
    store, out = tmp_path / "store", tmp_path / "out"
    store_sequence(store, 1)
    store_sequence(store, 2)
    calls = []

    def flaky_predict(frames, roi_xyxyn):
        calls.append(frames)
        if len(calls) == 1:
            raise RuntimeError("boom")
        return fake_predict_ok(frames, roi_xyxyn)

    summary = run(store, out, predict=flaky_predict)
    assert summary["replayed"] == 1
    dropped = json.loads(
        (out / "sis-67" / "vit_dinov2_finetune" / "dropped.json").read_text()
    )
    assert dropped[0]["reason"] == "predict_failed"


def test_score_mismatch_flagged(tmp_path):
    store, out = tmp_path / "store", tmp_path / "out"
    store_sequence(store, 1, score=0.5)  # recorded 0.5, replay says 0.93
    summary = run(store, out)
    assert summary["mismatched"] == 1
    assert summary["window_drift"] == 0
    rows = json.loads(
        (out / "sis-67" / "vit_dinov2_finetune" / "results.json").read_text()
    )
    assert rows[0]["replay_matches"] is False
    assert rows[0]["matched_window_frames"] is None
    assert SCORE_TOLERANCE == 1e-5


def test_cli_replay_defaults_point_at_package_compose_file():
    args = _parse_args(["replay"])
    assert args.compose_file.name == "docker-compose.yml"
    assert args.compose_file.parent.name == "monitor"
    assert args.compose_file.is_file()


def test_unhealthy_stack_drops_group_and_continues(tmp_path):
    store, out = tmp_path / "store", tmp_path / "out"
    store_sequence(store, 1, api_version="0.0.8")
    store_sequence(store, 2, api_version="0.3.1")

    class SickStack(FakeStack):
        def wait_healthy(self, **kwargs):
            if self.version == "0.0.8":
                raise StackError("api never became healthy")
            return super().wait_healthy(**kwargs)

    FakeStack.instances = []
    summary = run_replay(
        store_dir=store,
        output_dir=out,
        compose_file=Path("dc.yml"),
        stack_factory=SickStack,
        predict=fake_predict_ok,
    )
    assert summary["replayed"] == 1  # the 0.3.1 group still ran
    dropped = json.loads(
        (out / "sis-67" / "vit_dinov2_finetune" / "dropped.json").read_text()
    )
    assert {d["reason"] for d in dropped} == {"stack_unhealthy"}
    # the sick stack was still torn down
    sick = next(s for s in FakeStack.instances if s.version == "0.0.8")
    assert sick.down_called


def test_window_drift_found_by_probing(tmp_path):
    store, out = tmp_path / "store", tmp_path / "out"
    # 12 distinct frames; production scored at the first-5 window (recorded 0.5)
    store_sequence(store, 1, score=0.5, n_frames=12)

    def window_predict(frames, roi_xyxyn):
        resp = json.loads(json.dumps(fake_predict_ok(frames, roi_xyxyn)))
        # full final window (last 10 of 12) -> 0.93; the first-5 window -> 0.5
        if frames == [f"cam/seq1-f{i}.jpg" for i in range(5)]:
            resp["probability"] = 0.5
        return resp

    summary = run(store, out, predict=window_predict)
    assert summary == {
        "replayed": 1,
        "mismatched": 1,
        "window_drift": 1,
        "dropped": 0,
    }
    rows = json.loads(
        (out / "sis-67" / "vit_dinov2_finetune" / "results.json").read_text()
    )
    assert rows[0]["replay_matches"] is False
    assert rows[0]["matched_window_frames"] == 5

    # probing stopped at n=5: 1 main replay upload + 2 probe uploads (n=4, n=5)
    assert len(FakeStack.instances[0].uploaded) == 3

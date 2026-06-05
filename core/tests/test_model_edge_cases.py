"""Edge-case tests for BboxTubeTemporalModel.predict()."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image

from temporal_model.core.model import BboxTubeTemporalModel
from temporal_model.core.protocol import Frame
from temporal_model.core.temporal_classifier import TemporalSmokeClassifier

TEST_CONFIG: dict = {
    "infer": {
        "confidence_threshold": 0.01,
        "iou_nms": 0.2,
        "image_size": 1024,
        "pad_to_min_frames": 0,
        "pad_strategy": "symmetric",
    },
    "tubes": {
        "iou_threshold": 0.2,
        "max_misses": 2,
        "min_tube_length": 4,
        "infer_min_tube_length": 2,
        "min_detected_entries": 2,
        "interpolate_gaps": True,
    },
    "model_input": {
        "context_factor": 1.5,
        "patch_size": 224,
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
    },
    "classifier": {
        "backbone": "vit_small_patch16_224",
        "max_frames": 6,
        "pretrained": False,
        "global_pool": "token",
    },
    "decision": {
        "aggregation": "max_logit",
        "threshold": 0.0,
        "target_recall": 0.95,
        "trigger_rule": "end_of_winner",
    },
}


def _fake_yolo_factory(
    per_frame_xywhn: list[list[tuple[float, float, float, float, float]]],
):
    """Return a mock YOLO whose ``.predict`` yields fixed detections per frame."""

    def fake_predict(paths: list[str], **_kwargs):
        assert len(paths) == len(per_frame_xywhn)
        results = []
        for boxes in per_frame_xywhn:
            r = MagicMock()
            if not boxes:
                r.boxes = MagicMock()
                r.boxes.__len__ = lambda self: 0
                r.boxes.xywhn = torch.zeros(0, 4)
                r.boxes.conf = torch.zeros(0)
                r.boxes.cls = torch.zeros(0)
            else:
                n = len(boxes)
                r.boxes = MagicMock()
                r.boxes.__len__ = lambda self, n=n: n
                r.boxes.xywhn = torch.tensor(
                    [[c, cy, w, h] for (c, cy, w, h, _) in boxes]
                )
                r.boxes.conf = torch.tensor([conf for (_, _, _, _, conf) in boxes])
                r.boxes.cls = torch.zeros(n)
            results.append(r)
        return results

    m = MagicMock()
    m.predict.side_effect = fake_predict
    return m


@pytest.fixture()
def tiny_classifier() -> TemporalSmokeClassifier:
    model = TemporalSmokeClassifier(
        backbone="vit_small_patch16_224",
        pretrained=False,
        global_pool="token",
        transformer_num_layers=1,
        transformer_num_heads=6,
        transformer_ffn_dim=64,
        transformer_dropout=0.0,
        max_frames=6,
    )
    model.eval()
    return model


@pytest.fixture()
def red_frames(tmp_path: Path) -> list[Frame]:
    frames = []
    for i in range(6):
        arr = np.full((64, 64, 3), fill_value=[180, 30, 30], dtype=np.uint8)
        p = tmp_path / f"f_{i:02d}.jpg"
        Image.fromarray(arr).save(p, format="JPEG")
        frames.append(Frame(frame_id=p.stem, image_path=p, timestamp=None))
    return frames


class TestEmptyFrames:
    def test_returns_negative(self, tiny_classifier: TemporalSmokeClassifier) -> None:
        yolo = MagicMock()
        model = BboxTubeTemporalModel(
            yolo_model=yolo, classifier=tiny_classifier, config=TEST_CONFIG
        )
        out = model.predict(frames=[])
        assert out.is_positive is False
        assert out.trigger_frame_index is None
        assert out.details["preprocessing"]["num_frames_input"] == 0
        assert out.details["tubes"]["num_candidates"] == 0
        assert out.details["tubes"]["kept"] == []
        assert out.details["decision"]["trigger_tube_id"] is None
        yolo.predict.assert_not_called()


class TestZeroDetections:
    def test_no_tubes_means_negative(
        self, tiny_classifier: TemporalSmokeClassifier, red_frames: list[Frame]
    ) -> None:
        yolo = _fake_yolo_factory([[] for _ in red_frames])
        model = BboxTubeTemporalModel(
            yolo_model=yolo, classifier=tiny_classifier, config=TEST_CONFIG
        )
        out = model.predict(frames=red_frames)
        assert out.is_positive is False
        assert out.trigger_frame_index is None
        assert out.details["tubes"]["num_candidates"] == 0
        assert out.details["tubes"]["kept"] == []


class TestShortTubeBelowInferFloor:
    def test_single_frame_detection_discarded(
        self, tiny_classifier: TemporalSmokeClassifier, red_frames: list[Frame]
    ) -> None:
        # Only frame 0 has a detection — tube length 1, below infer_min_tube_length=2.
        per_frame = [[(0.5, 0.5, 0.1, 0.1, 0.9)]] + [[] for _ in red_frames[1:]]
        yolo = _fake_yolo_factory(per_frame)
        model = BboxTubeTemporalModel(
            yolo_model=yolo, classifier=tiny_classifier, config=TEST_CONFIG
        )
        out = model.predict(frames=red_frames)
        assert out.is_positive is False
        assert out.details["tubes"]["num_candidates"] == 1
        assert out.details["tubes"]["kept"] == []


class TestTruncation:
    def test_sequence_longer_than_max_frames(
        self, tiny_classifier: TemporalSmokeClassifier, red_frames: list[Frame]
    ) -> None:
        # red_frames has 6; TEST_CONFIG max_frames=6; extend to 9.
        extra = red_frames + red_frames[:3]
        # YOLO will only see the first 6 (truncated).
        per_frame = [[(0.5, 0.5, 0.1, 0.1, 0.9)] for _ in range(6)]
        yolo = _fake_yolo_factory(per_frame)
        model = BboxTubeTemporalModel(
            yolo_model=yolo, classifier=tiny_classifier, config=TEST_CONFIG
        )
        out = model.predict(frames=extra)
        assert out.details["preprocessing"]["num_frames_input"] == 9
        assert out.details["preprocessing"]["num_truncated"] == 3


class TestShortSequencePadding:
    """``infer.pad_to_min_frames`` symmetrically pads a short sequence by
    alternately prepending the first frame and appending the last until the
    configured minimum is reached. Mirrors the ``pad_sequence`` helper in
    ``tracking_fsm_baseline`` so both experiments handle truncated benchmark
    sequences the same way."""

    def test_pad_disabled_by_default(
        self, tiny_classifier: TemporalSmokeClassifier, red_frames: list[Frame]
    ) -> None:
        short = red_frames[:2]
        per_frame = [[(0.5, 0.5, 0.1, 0.1, 0.9)] for _ in short]
        yolo = _fake_yolo_factory(per_frame)
        model = BboxTubeTemporalModel(
            yolo_model=yolo, classifier=tiny_classifier, config=TEST_CONFIG
        )
        out = model.predict(frames=short)
        assert out.details["preprocessing"]["padded_frame_indices"] == []
        assert out.details["preprocessing"]["num_frames_input"] == 2

    def test_pad_extends_short_sequence_symmetrically(
        self, tiny_classifier: TemporalSmokeClassifier, red_frames: list[Frame]
    ) -> None:
        # Enable padding up to 5 frames; input has 2 real frames.
        cfg = {
            **TEST_CONFIG,
            "infer": {**TEST_CONFIG["infer"], "pad_to_min_frames": 5},
        }
        short = red_frames[:2]
        # Alternating prepend/append: [A,B] -> [A,A,B] -> [A,A,B,B] -> [A,A,A,B,B].
        # YOLO sees 5 frames; fake YOLO needs a 5-length list.
        per_frame = [[(0.5, 0.5, 0.1, 0.1, 0.9)] for _ in range(5)]
        yolo = _fake_yolo_factory(per_frame)
        model = BboxTubeTemporalModel(
            yolo_model=yolo, classifier=tiny_classifier, config=cfg
        )
        out = model.predict(frames=short)

        # Symmetric pad of [A, B] to length 5: real frames end up at slots 2, 3.
        assert out.details["preprocessing"]["padded_frame_indices"] == [0, 1, 4]
        # num_frames_input reports the original length (pre-pad).
        assert out.details["preprocessing"]["num_frames_input"] == 2

    def test_pad_noop_when_sequence_already_long_enough(
        self, tiny_classifier: TemporalSmokeClassifier, red_frames: list[Frame]
    ) -> None:
        cfg = {
            **TEST_CONFIG,
            "infer": {**TEST_CONFIG["infer"], "pad_to_min_frames": 3},
        }
        per_frame = [[(0.5, 0.5, 0.1, 0.1, 0.9)] for _ in red_frames]  # 6 frames
        yolo = _fake_yolo_factory(per_frame)
        model = BboxTubeTemporalModel(
            yolo_model=yolo, classifier=tiny_classifier, config=cfg
        )
        out = model.predict(frames=red_frames)
        assert out.details["preprocessing"]["padded_frame_indices"] == []

    def test_pad_strategy_uniform_spreads_duplicates(
        self, tiny_classifier: TemporalSmokeClassifier, red_frames: list[Frame]
    ) -> None:
        """``pad_strategy='uniform'`` uses nearest-neighbor upsampling so the
        real frames are spread evenly across the padded sequence rather than
        clustered at the boundaries (cf. symmetric)."""
        cfg = {
            **TEST_CONFIG,
            "infer": {
                **TEST_CONFIG["infer"],
                "pad_to_min_frames": 6,
                "pad_strategy": "uniform",
            },
        }
        short = red_frames[:2]  # N=2 real frames, pad to M=6
        # Uniform nearest-neighbor map: i*2//6 for i in 0..5 = [0,0,0,1,1,1]
        # so the first 3 slots sample frame 0 and the last 3 sample frame 1.
        per_frame = [[(0.5, 0.5, 0.1, 0.1, 0.9)] for _ in range(6)]
        yolo = _fake_yolo_factory(per_frame)
        model = BboxTubeTemporalModel(
            yolo_model=yolo, classifier=tiny_classifier, config=cfg
        )
        out = model.predict(frames=short)
        # Uniform i*2//6 for i in 0..5 -> [0,0,0,1,1,1]; dupes at slots 1, 2, 4, 5.
        assert out.details["preprocessing"]["padded_frame_indices"] == [1, 2, 4, 5]
        assert out.details["preprocessing"]["num_frames_input"] == 2

    def test_pad_strategy_unknown_raises(
        self, tiny_classifier: TemporalSmokeClassifier, red_frames: list[Frame]
    ) -> None:
        cfg = {
            **TEST_CONFIG,
            "infer": {
                **TEST_CONFIG["infer"],
                "pad_to_min_frames": 5,
                "pad_strategy": "bogus",
            },
        }
        short = red_frames[:2]
        yolo = MagicMock()
        model = BboxTubeTemporalModel(
            yolo_model=yolo, classifier=tiny_classifier, config=cfg
        )
        with pytest.raises(ValueError, match="unknown pad_strategy"):
            model.predict(frames=short)


class TestDeviceSelection:
    def test_explicit_cpu_puts_classifier_on_cpu(
        self, tiny_classifier: TemporalSmokeClassifier
    ) -> None:
        yolo = MagicMock()
        model = BboxTubeTemporalModel(
            yolo_model=yolo,
            classifier=tiny_classifier,
            config=TEST_CONFIG,
            device="cpu",
        )
        assert model.device.type == "cpu"
        assert next(model._classifier.parameters()).device.type == "cpu"

    def test_predict_on_cpu_runs_end_to_end(
        self, tiny_classifier: TemporalSmokeClassifier, red_frames: list[Frame]
    ) -> None:
        per_frame = [[(0.5, 0.5, 0.1, 0.1, 0.9)] for _ in red_frames]
        yolo = _fake_yolo_factory(per_frame)
        model = BboxTubeTemporalModel(
            yolo_model=yolo,
            classifier=tiny_classifier,
            config=TEST_CONFIG,
            device="cpu",
        )
        out = model.predict(frames=red_frames)
        kept = out.details["tubes"]["kept"]
        assert len(kept) == 1

    def test_predict_details_include_per_tube_entries(
        self, tiny_classifier: TemporalSmokeClassifier, red_frames: list[Frame]
    ) -> None:
        """``details['tubes']['kept']`` exposes every kept tube's full entries
        alongside its logit, so downstream diagnostics can render any tube —
        not just the trigger."""
        per_frame = [[(0.5, 0.5, 0.1, 0.1, 0.9)] for _ in red_frames]
        yolo = _fake_yolo_factory(per_frame)
        # Force positive sequence so the single kept tube is the trigger
        # (under D2, trigger_tube_id is None when no tube qualifies).
        cfg = {
            **TEST_CONFIG,
            "decision": {**TEST_CONFIG["decision"], "threshold": -1e6},
        }
        model = BboxTubeTemporalModel(
            yolo_model=yolo,
            classifier=tiny_classifier,
            config=cfg,
            device="cpu",
        )
        out = model.predict(frames=red_frames)

        kept = out.details["tubes"]["kept"]
        assert isinstance(kept, list)
        assert len(kept) == 1
        tube = kept[0]
        assert set(tube.keys()) == {
            "tube_id",
            "start_frame",
            "end_frame",
            "logit",
            "probability",
            "first_crossing_frame",
            "entries",
        }
        assert out.details["decision"]["trigger_tube_id"] == tube["tube_id"]
        assert isinstance(tube["entries"], list)
        entry = tube["entries"][0]
        assert set(entry.keys()) == {"frame_idx", "bbox", "is_gap", "confidence"}

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_predict_on_cuda_runs_end_to_end(
        self, tiny_classifier: TemporalSmokeClassifier, red_frames: list[Frame]
    ) -> None:
        per_frame = [[(0.5, 0.5, 0.1, 0.1, 0.9)] for _ in red_frames]
        yolo = _fake_yolo_factory(per_frame)
        model = BboxTubeTemporalModel(
            yolo_model=yolo,
            classifier=tiny_classifier,
            config=TEST_CONFIG,
            device="cuda",
        )
        assert model.device.type == "cuda"
        assert next(model._classifier.parameters()).device.type == "cuda"
        out = model.predict(frames=red_frames)
        assert len(out.details["tubes"]["kept"]) == 1

    def test_auto_detect_picks_cuda_when_available(
        self, tiny_classifier: TemporalSmokeClassifier
    ) -> None:
        yolo = MagicMock()
        model = BboxTubeTemporalModel(
            yolo_model=yolo, classifier=tiny_classifier, config=TEST_CONFIG
        )
        expected = "cuda" if torch.cuda.is_available() else "cpu"
        if not torch.cuda.is_available() and torch.backends.mps.is_available():
            expected = "mps"
        assert model.device.type == expected


class TestLogisticThreshold:
    def _model(self, classifier, config=TEST_CONFIG):
        return BboxTubeTemporalModel(
            yolo_model=MagicMock(), classifier=classifier, config=config
        )

    def test_get_defaults_to_half_when_absent(
        self, tiny_classifier: TemporalSmokeClassifier
    ) -> None:
        # TEST_CONFIG["decision"] has no logistic_threshold key.
        assert self._model(tiny_classifier).logistic_threshold == 0.5

    def test_get_returns_configured_value(
        self, tiny_classifier: TemporalSmokeClassifier
    ) -> None:
        cfg = {
            **TEST_CONFIG,
            "decision": {**TEST_CONFIG["decision"], "logistic_threshold": 0.8},
        }
        assert self._model(tiny_classifier, cfg).logistic_threshold == 0.8

    def test_set_updates_config(self, tiny_classifier: TemporalSmokeClassifier) -> None:
        model = self._model(tiny_classifier)
        model.logistic_threshold = 0.7
        assert model.logistic_threshold == 0.7
        assert model._cfg["decision"]["logistic_threshold"] == 0.7

    @pytest.mark.parametrize("value", [0.0, 1.0])
    def test_set_accepts_boundaries(
        self, tiny_classifier: TemporalSmokeClassifier, value: float
    ) -> None:
        model = self._model(tiny_classifier)
        model.logistic_threshold = value
        assert model.logistic_threshold == value

    @pytest.mark.parametrize("value", [-0.1, 1.5])
    def test_set_rejects_out_of_range(
        self, tiny_classifier: TemporalSmokeClassifier, value: float
    ) -> None:
        model = self._model(tiny_classifier)
        with pytest.raises(ValueError, match=r"logistic_threshold must be in \[0, 1\]"):
            model.logistic_threshold = value

    def test_aggregation_reflects_config(
        self, tiny_classifier: TemporalSmokeClassifier
    ) -> None:
        # TEST_CONFIG uses max_logit; a logistic config reports logistic.
        assert self._model(tiny_classifier).aggregation == "max_logit"
        cfg = {
            **TEST_CONFIG,
            "decision": {**TEST_CONFIG["decision"], "aggregation": "logistic"},
        }
        assert self._model(tiny_classifier, cfg).aggregation == "logistic"


class TestFirstCrossingTrigger:
    def test_first_crossing_trigger_never_exceeds_end_frame(
        self, tiny_classifier: TemporalSmokeClassifier, red_frames: list[Frame]
    ) -> None:
        """Trigger frame must not exceed the trigger tube's end_frame, and
        the trigger tube's ``first_crossing_frame`` must equal the trigger."""
        per_frame = [[(0.5, 0.5, 0.1, 0.1, 0.9)] for _ in red_frames]
        yolo = _fake_yolo_factory(per_frame)
        cfg = {
            **TEST_CONFIG,
            "decision": {
                **TEST_CONFIG["decision"],
                "threshold": -1e6,
            },
        }
        model = BboxTubeTemporalModel(
            yolo_model=yolo,
            classifier=tiny_classifier,
            config=cfg,
            device="cpu",
        )
        out = model.predict(frames=red_frames)

        assert out.is_positive is True
        assert out.trigger_frame_index is not None

        trigger_tube_id = out.details["decision"]["trigger_tube_id"]
        trigger_tube = next(
            t for t in out.details["tubes"]["kept"] if t["tube_id"] == trigger_tube_id
        )
        assert out.trigger_frame_index <= trigger_tube["end_frame"]
        assert trigger_tube["first_crossing_frame"] == out.trigger_frame_index

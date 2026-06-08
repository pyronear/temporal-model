"""Pure-function unit tests for inference helpers."""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image

from temporal_model.core.inference import (
    build_tubes_for_inference,
    crop_tube_patches,
    filter_and_interpolate_tubes,
    find_first_crossing_trigger,
    run_yolo_on_frames,
    score_tubes,
)
from temporal_model.core.logistic_calibrator import LogisticCalibrator
from temporal_model.core.protocol import Frame
from temporal_model.core.tubes import build_tubes
from temporal_model.core.types import Detection, FrameDetections, Tube, TubeEntry


def _fake_yolo_result(
    xywhn: list[tuple[float, float, float, float, float]],
) -> MagicMock:
    """Build a MagicMock shaped like ultralytics Results for one image.

    xywhn: list of (cx, cy, w, h, conf) tuples.
    """
    boxes = MagicMock()
    boxes.__len__ = lambda self: len(xywhn)
    boxes.xywhn = torch.tensor([[c, cy, w, h] for (c, cy, w, h, _) in xywhn])
    boxes.conf = torch.tensor([conf for (_, _, _, _, conf) in xywhn])
    boxes.cls = torch.zeros(len(xywhn))
    result = MagicMock()
    result.boxes = boxes
    return result


class TestRunYoloOnFrames:
    def test_batches_all_frames_in_single_call(self) -> None:
        yolo = MagicMock()
        yolo.predict.return_value = [_fake_yolo_result([]), _fake_yolo_result([])]
        frames = [
            Frame(frame_id="f0", image_path=Path("/x/f0.jpg"), timestamp=None),
            Frame(frame_id="f1", image_path=Path("/x/f1.jpg"), timestamp=None),
        ]

        run_yolo_on_frames(
            yolo, frames, confidence_threshold=0.01, iou_nms=0.2, image_size=1024
        )

        assert yolo.predict.call_count == 1
        args, kwargs = yolo.predict.call_args
        assert args[0] == ["/x/f0.jpg", "/x/f1.jpg"]
        assert kwargs["conf"] == 0.01
        assert kwargs["iou"] == 0.2
        assert kwargs["imgsz"] == 1024
        assert kwargs["verbose"] is False

    def test_converts_detections(self) -> None:
        yolo = MagicMock()
        yolo.predict.return_value = [
            _fake_yolo_result([(0.5, 0.4, 0.1, 0.2, 0.9)]),
            _fake_yolo_result([]),
        ]
        ts = datetime(2024, 1, 1, 12, 0, 0)
        frames = [
            Frame(frame_id="f0", image_path=Path("/x/f0.jpg"), timestamp=ts),
            Frame(frame_id="f1", image_path=Path("/x/f1.jpg"), timestamp=None),
        ]

        result = run_yolo_on_frames(
            yolo, frames, confidence_threshold=0.01, iou_nms=0.2, image_size=1024
        )

        assert len(result) == 2

        # Frame 0: one detection, check structural + numeric approximately.
        fd0 = result[0]
        assert fd0.frame_idx == 0
        assert fd0.frame_id == "f0"
        assert fd0.timestamp == ts
        assert len(fd0.detections) == 1
        d0 = fd0.detections[0]
        assert d0.class_id == 0
        assert d0.cx == pytest.approx(0.5, rel=1e-6)
        assert d0.cy == pytest.approx(0.4, rel=1e-6)
        assert d0.w == pytest.approx(0.1, rel=1e-6)
        assert d0.h == pytest.approx(0.2, rel=1e-6)
        assert d0.confidence == pytest.approx(0.9, rel=1e-6)

        # Frame 1: no detections.
        assert result[1] == FrameDetections(
            frame_idx=1, frame_id="f1", timestamp=None, detections=[]
        )

    def test_empty_frames_returns_empty(self) -> None:
        yolo = MagicMock()
        yolo.predict.return_value = []
        result = run_yolo_on_frames(
            yolo, [], confidence_threshold=0.01, iou_nms=0.2, image_size=1024
        )
        assert result == []
        yolo.predict.assert_not_called()


def _tube(tid: int, entries: list[tuple[int, Detection | None]]) -> Tube:
    return Tube(
        tube_id=tid,
        entries=[TubeEntry(frame_idx=i, detection=d) for (i, d) in entries],
        start_frame=entries[0][0],
        end_frame=entries[-1][0],
    )


def _det(cx: float = 0.5, cy: float = 0.5, w: float = 0.1, h: float = 0.1) -> Detection:
    return Detection(class_id=0, cx=cx, cy=cy, w=w, h=h, confidence=0.9)


class TestFilterAndInterpolate:
    def test_drops_tubes_shorter_than_min_length(self) -> None:
        tubes = [
            _tube(0, [(0, _det()), (1, _det())]),  # length 2 - keep
            _tube(1, [(3, _det())]),  # length 1 - drop
        ]
        out = filter_and_interpolate_tubes(
            tubes, min_tube_length=2, min_detected_entries=1, interpolate_gaps=False
        )
        assert [t.tube_id for t in out] == [0]

    def test_drops_tubes_with_too_few_observed(self) -> None:
        tubes = [
            _tube(0, [(0, _det()), (1, None), (2, None), (3, None)]),
            _tube(1, [(0, _det()), (1, _det()), (2, None), (3, None)]),
        ]
        out = filter_and_interpolate_tubes(
            tubes, min_tube_length=2, min_detected_entries=2, interpolate_gaps=False
        )
        assert [t.tube_id for t in out] == [1]

    def test_interpolation_applied_when_enabled(self) -> None:
        tubes = [
            _tube(
                0,
                [
                    (0, _det(cx=0.2)),
                    (1, None),
                    (2, _det(cx=0.4)),
                ],
            ),
        ]
        out = filter_and_interpolate_tubes(
            tubes, min_tube_length=2, min_detected_entries=2, interpolate_gaps=True
        )
        assert len(out) == 1
        mid = out[0].entries[1]
        assert mid.is_gap is True
        assert mid.detection is not None
        assert mid.detection.cx == pytest.approx(0.3)

    def test_interpolation_skipped_when_disabled(self) -> None:
        tubes = [
            _tube(0, [(0, _det()), (1, None), (2, _det())]),
        ]
        out = filter_and_interpolate_tubes(
            tubes, min_tube_length=2, min_detected_entries=2, interpolate_gaps=False
        )
        assert out[0].entries[1].detection is None

    def test_empty_input(self) -> None:
        assert (
            filter_and_interpolate_tubes(
                [], min_tube_length=2, min_detected_entries=1, interpolate_gaps=True
            )
            == []
        )


@pytest.fixture()
def red_image_sequence(tmp_path: Path) -> list[Path]:
    """Three 128x128 solid-red JPGs."""
    paths = []
    for i in range(3):
        img = np.full((128, 128, 3), fill_value=[200, 30, 30], dtype=np.uint8)
        p = tmp_path / f"frame_{i:02d}.jpg"
        Image.fromarray(img).save(p, format="JPEG", quality=95)
        paths.append(p)
    return paths


@pytest.fixture()
def gradient_image_sequence(tmp_path: Path) -> list[Path]:
    """Three 128x128 horizontal-gradient JPGs (x-varying), so different crop
    windows produce different patches. All three frames are identical, so a
    fixed (stabilized) window crops identically across frames."""
    paths = []
    ramp = np.tile(np.linspace(0, 255, 128, dtype=np.uint8), (128, 1))
    img = np.stack([ramp, ramp, ramp], axis=-1)
    for i in range(3):
        p = tmp_path / f"frame_{i:02d}.jpg"
        Image.fromarray(img).save(p, format="JPEG", quality=95)
        paths.append(p)
    return paths


class TestCropTubePatches:
    def test_output_shape(self, red_image_sequence: list[Path]) -> None:
        frames = [
            Frame(frame_id=p.stem, image_path=p, timestamp=None)
            for p in red_image_sequence
        ]
        tube = _tube(
            0,
            [
                (0, _det(cx=0.5, cy=0.5, w=0.2, h=0.2)),
                (1, _det(cx=0.5, cy=0.5, w=0.2, h=0.2)),
                (2, _det(cx=0.5, cy=0.5, w=0.2, h=0.2)),
            ],
        )
        patches, mask = crop_tube_patches(
            tube,
            frames,
            context_factor=1.5,
            patch_size=224,
            max_frames=5,
            normalization_mean=[0.485, 0.456, 0.406],
            normalization_std=[0.229, 0.224, 0.225],
        )
        assert patches.shape == (5, 3, 224, 224)
        assert patches.dtype == torch.float32
        assert mask.shape == (5,)
        assert mask.tolist() == [True, True, True, False, False]

    def test_padding_slots_are_zero(self, red_image_sequence: list[Path]) -> None:
        frames = [
            Frame(frame_id=p.stem, image_path=p, timestamp=None)
            for p in red_image_sequence
        ]
        tube = _tube(0, [(0, _det()), (1, _det())])
        patches, mask = crop_tube_patches(
            tube,
            frames,
            context_factor=1.5,
            patch_size=224,
            max_frames=5,
            normalization_mean=[0.485, 0.456, 0.406],
            normalization_std=[0.229, 0.224, 0.225],
        )
        assert torch.all(patches[2:] == 0.0)

    def test_truncates_tubes_longer_than_max_frames(
        self, red_image_sequence: list[Path]
    ) -> None:
        frames = [
            Frame(frame_id=p.stem, image_path=p, timestamp=None)
            for p in red_image_sequence
        ]
        tube = _tube(0, [(0, _det()), (1, _det()), (2, _det())])
        patches, mask = crop_tube_patches(
            tube,
            frames,
            context_factor=1.5,
            patch_size=224,
            max_frames=2,
            normalization_mean=[0.485, 0.456, 0.406],
            normalization_std=[0.229, 0.224, 0.225],
        )
        assert patches.shape == (2, 3, 224, 224)
        assert mask.tolist() == [True, True]

    def test_stabilize_uses_constant_union_window(
        self, gradient_image_sequence: list[Path]
    ) -> None:
        frames = [
            Frame(frame_id=p.stem, image_path=p, timestamp=None)
            for p in gradient_image_sequence
        ]
        # Two frames, boxes at different x; union window is centered between them.
        tube = _tube(
            0,
            [
                (0, _det(cx=0.3, cy=0.5, w=0.1, h=0.1)),
                (1, _det(cx=0.7, cy=0.5, w=0.1, h=0.1)),
            ],
        )
        stab, _ = crop_tube_patches(
            tube, frames, context_factor=1.5, patch_size=224, max_frames=5,
            normalization_mean=[0.485, 0.456, 0.406],
            normalization_std=[0.229, 0.224, 0.225],
            stabilize=True,
        )
        per_frame, _ = crop_tube_patches(
            tube, frames, context_factor=1.5, patch_size=224, max_frames=5,
            normalization_mean=[0.485, 0.456, 0.406],
            normalization_std=[0.229, 0.224, 0.225],
            stabilize=False,
        )
        # Stabilized: both real frames share the same fixed window -> identical patches.
        assert torch.equal(stab[0], stab[1])
        # Per-frame: different boxes -> different patches.
        assert not torch.equal(per_frame[0], per_frame[1])

    def test_default_is_stabilized(self, gradient_image_sequence: list[Path]) -> None:
        frames = [
            Frame(frame_id=p.stem, image_path=p, timestamp=None)
            for p in gradient_image_sequence
        ]
        tube = _tube(
            0,
            [
                (0, _det(cx=0.3, cy=0.5, w=0.1, h=0.1)),
                (1, _det(cx=0.7, cy=0.5, w=0.1, h=0.1)),
            ],
        )
        default, _ = crop_tube_patches(  # no stabilize arg -> defaults to True
            tube, frames, context_factor=1.5, patch_size=224, max_frames=5,
            normalization_mean=[0.485, 0.456, 0.406],
            normalization_std=[0.229, 0.224, 0.225],
        )
        assert torch.equal(default[0], default[1])


class TestScoreTubes:
    def test_empty_input_returns_empty(self) -> None:
        classifier = MagicMock()
        logits = score_tubes(classifier, patches_per_tube=[], masks_per_tube=[])
        assert logits.shape == (0,)
        classifier.assert_not_called()

    def test_single_batched_forward(self) -> None:
        classifier = MagicMock(return_value=torch.tensor([1.2, -0.3]))
        patches = [torch.zeros(4, 3, 8, 8), torch.zeros(4, 3, 8, 8)]
        masks = [
            torch.tensor([True, True, True, True]),
            torch.tensor([True, True, False, False]),
        ]

        logits = score_tubes(classifier, patches_per_tube=patches, masks_per_tube=masks)

        assert classifier.call_count == 1
        args, _ = classifier.call_args
        assert args[0].shape == (2, 4, 3, 8, 8)
        assert args[1].shape == (2, 4)
        assert logits.tolist() == pytest.approx([1.2, -0.3], rel=1e-5)


def _make_patches_and_masks(
    tubes: list[Tube], max_frames: int = 8
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Build dummy patches/masks for tubes; mask reflects tube length."""
    patches: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for t in tubes:
        n = len(t.entries)
        patches.append(torch.zeros(max_frames, 3, 8, 8))
        m = torch.zeros(max_frames, dtype=torch.bool)
        m[:n] = True
        masks.append(m)
    return patches, masks


class TestFindFirstCrossingTrigger:
    """Unit tests for find_first_crossing_trigger (max_logit mode)."""

    def test_empty_tubes_returns_negative(self) -> None:
        res = find_first_crossing_trigger(
            classifier=MagicMock(),
            tubes=[],
            patches_per_tube=[],
            masks_per_tube=[],
            full_logits=torch.zeros(0),
            aggregation="max_logit",
            threshold=0.0,
            min_prefix_length=2,
        )
        assert res == (False, None, None, {})

    def test_no_qualifying_tubes_returns_negative(self) -> None:
        tubes = [_tube(1, [(0, _det()), (1, _det())])]
        patches, masks = _make_patches_and_masks(tubes)
        full_logits = torch.tensor([-1.0])
        classifier = MagicMock()

        res = find_first_crossing_trigger(
            classifier=classifier,
            tubes=tubes,
            patches_per_tube=patches,
            masks_per_tube=masks,
            full_logits=full_logits,
            aggregation="max_logit",
            threshold=0.0,
            min_prefix_length=2,
        )
        assert res == (False, None, None, {})
        classifier.assert_not_called()

    def test_single_qualifying_tube_crosses_at_min_prefix(self) -> None:
        tubes = [
            _tube(7, [(3, _det()), (4, _det()), (5, _det()), (6, _det())]),
        ]
        patches, masks = _make_patches_and_masks(tubes)
        full_logits = torch.tensor([1.0])
        classifier = MagicMock(return_value=torch.tensor([2.0]))

        is_positive, trigger, winner, diag = find_first_crossing_trigger(
            classifier=classifier,
            tubes=tubes,
            patches_per_tube=patches,
            masks_per_tube=masks,
            full_logits=full_logits,
            aggregation="max_logit",
            threshold=0.0,
            min_prefix_length=2,
        )
        assert is_positive is True
        assert winner == 7
        assert trigger == 4
        assert diag == {7: {"crossing_frame": 4, "prefix_length": 2}}
        assert classifier.call_count == 1

    def test_earliest_crossing_wins_across_tubes(self) -> None:
        tube_a = _tube(1, [(5, _det()), (6, _det()), (7, _det())])
        tube_b = _tube(2, [(0, _det()), (1, _det()), (2, _det())])
        tubes = [tube_a, tube_b]
        patches, masks = _make_patches_and_masks(tubes)
        full_logits = torch.tensor([1.0, 1.0])
        classifier = MagicMock(return_value=torch.tensor([2.0]))

        is_positive, trigger, winner, diag = find_first_crossing_trigger(
            classifier=classifier,
            tubes=tubes,
            patches_per_tube=patches,
            masks_per_tube=masks,
            full_logits=full_logits,
            aggregation="max_logit",
            threshold=0.0,
            min_prefix_length=2,
        )
        assert is_positive is True
        assert winner == 2
        assert trigger == 1
        assert diag == {
            1: {"crossing_frame": 6, "prefix_length": 2},
            2: {"crossing_frame": 1, "prefix_length": 2},
        }

    def test_tie_on_crossing_frame_breaks_on_smallest_tube_id(self) -> None:
        tube_a = _tube(9, [(0, _det()), (1, _det()), (2, _det())])
        tube_b = _tube(5, [(0, _det()), (1, _det()), (2, _det())])
        tubes = [tube_a, tube_b]
        patches, masks = _make_patches_and_masks(tubes)
        full_logits = torch.tensor([1.0, 1.0])
        classifier = MagicMock(return_value=torch.tensor([2.0]))

        _, _, winner, _ = find_first_crossing_trigger(
            classifier=classifier,
            tubes=tubes,
            patches_per_tube=patches,
            masks_per_tube=masks,
            full_logits=full_logits,
            aggregation="max_logit",
            threshold=0.0,
            min_prefix_length=2,
        )
        assert winner == 5

    def test_d2_guard_ignores_non_qualifying_tube(self) -> None:
        tube_q = _tube(1, [(4, _det()), (5, _det()), (6, _det())])
        tube_skip = _tube(2, [(0, _det()), (1, _det()), (2, _det())])
        tubes = [tube_q, tube_skip]
        patches, masks = _make_patches_and_masks(tubes)
        full_logits = torch.tensor([1.0, -1.0])
        classifier = MagicMock(return_value=torch.tensor([2.0]))

        _, trigger, winner, diag = find_first_crossing_trigger(
            classifier=classifier,
            tubes=tubes,
            patches_per_tube=patches,
            masks_per_tube=masks,
            full_logits=full_logits,
            aggregation="max_logit",
            threshold=0.0,
            min_prefix_length=2,
        )
        assert winner == 1
        assert trigger == 5
        assert diag == {1: {"crossing_frame": 5, "prefix_length": 2}}

    def test_loop_walks_prefix_lengths_until_crossing(self) -> None:
        tube = _tube(3, [(0, _det()), (1, _det()), (2, _det()), (3, _det())])
        tubes = [tube]
        patches, masks = _make_patches_and_masks(tubes)
        full_logits = torch.tensor([1.0])
        classifier = MagicMock(side_effect=[torch.tensor([-0.5]), torch.tensor([0.5])])

        _, trigger, winner, diag = find_first_crossing_trigger(
            classifier=classifier,
            tubes=tubes,
            patches_per_tube=patches,
            masks_per_tube=masks,
            full_logits=full_logits,
            aggregation="max_logit",
            threshold=0.0,
            min_prefix_length=2,
        )
        assert winner == 3
        assert trigger == 2
        assert diag == {3: {"crossing_frame": 2, "prefix_length": 3}}
        assert classifier.call_count == 2

    def test_full_length_crossing_reuses_full_logits(self) -> None:
        tube = _tube(1, [(0, _det()), (1, _det()), (2, _det())])
        tubes = [tube]
        patches, masks = _make_patches_and_masks(tubes)
        full_logits = torch.tensor([1.0])
        classifier = MagicMock(return_value=torch.tensor([-0.5]))

        _, trigger, winner, diag = find_first_crossing_trigger(
            classifier=classifier,
            tubes=tubes,
            patches_per_tube=patches,
            masks_per_tube=masks,
            full_logits=full_logits,
            aggregation="max_logit",
            threshold=0.0,
            min_prefix_length=2,
        )
        assert winner == 1
        assert trigger == 2
        assert diag == {1: {"crossing_frame": 2, "prefix_length": 3}}
        assert classifier.call_count == 1

    def test_find_first_crossing_unknown_aggregation_raises(self) -> None:
        tubes = [_tube(1, [(0, _det()), (1, _det())])]
        patches, masks = _make_patches_and_masks(tubes)
        with pytest.raises(ValueError, match="aggregation"):
            find_first_crossing_trigger(
                classifier=MagicMock(),
                tubes=tubes,
                patches_per_tube=patches,
                masks_per_tube=masks,
                full_logits=torch.tensor([1.0]),
                aggregation="bogus",
                threshold=0.0,
                min_prefix_length=2,
            )

    def test_logistic_mode_fires_at_first_crossing(self) -> None:
        tube = _tube(4, [(0, _det()), (1, _det()), (2, _det())])
        tubes = [tube]
        patches, masks = _make_patches_and_masks(tubes)
        full_logits = torch.tensor([2.0])
        classifier = MagicMock(return_value=torch.tensor([0.5]))

        cal = LogisticCalibrator(
            features=["logit", "log_len", "mean_conf", "n_tubes"],
            coefficients=np.array([2.0, 0.0, 0.0, 0.0]),
            intercept=0.0,
            sanity_checks=[],
        )

        is_positive, trigger, winner, diag = find_first_crossing_trigger(
            classifier=classifier,
            tubes=tubes,
            patches_per_tube=patches,
            masks_per_tube=masks,
            full_logits=full_logits,
            aggregation="logistic",
            threshold=0.0,
            calibrator=cal,
            logistic_threshold=0.5,
            min_prefix_length=2,
        )
        assert is_positive is True
        assert winner == 4
        assert trigger == 1
        assert diag == {4: {"crossing_frame": 1, "prefix_length": 2}}

    def test_logistic_mode_no_qualifying_tube(self) -> None:
        tube = _tube(1, [(0, _det()), (1, _det()), (2, _det())])
        tubes = [tube]
        patches, masks = _make_patches_and_masks(tubes)
        full_logits = torch.tensor([-2.0])
        classifier = MagicMock()

        cal = LogisticCalibrator(
            features=["logit", "log_len", "mean_conf", "n_tubes"],
            coefficients=np.array([2.0, 0.0, 0.0, 0.0]),
            intercept=0.0,
            sanity_checks=[],
        )

        res = find_first_crossing_trigger(
            classifier=classifier,
            tubes=tubes,
            patches_per_tube=patches,
            masks_per_tube=masks,
            full_logits=full_logits,
            aggregation="logistic",
            threshold=0.0,
            calibrator=cal,
            logistic_threshold=0.5,
            min_prefix_length=2,
        )
        assert res == (False, None, None, {})
        classifier.assert_not_called()

    def test_logistic_mode_requires_calibrator(self) -> None:
        tubes = [_tube(1, [(0, _det()), (1, _det())])]
        patches, masks = _make_patches_and_masks(tubes)
        with pytest.raises(ValueError, match="calibrator"):
            find_first_crossing_trigger(
                classifier=MagicMock(),
                tubes=tubes,
                patches_per_tube=patches,
                masks_per_tube=masks,
                full_logits=torch.tensor([1.0]),
                aggregation="logistic",
                threshold=0.0,
                calibrator=None,
                logistic_threshold=0.5,
                min_prefix_length=2,
            )

    def test_logistic_log_len_affects_prefix_decision(self) -> None:
        """Predicate sees prefix length via log_len, not just the logit."""
        tube = _tube(1, [(0, _det()), (1, _det()), (2, _det()), (3, _det())])
        tubes = [tube]
        patches, masks = _make_patches_and_masks(tubes)
        full_logits = torch.tensor([10.0])
        classifier = MagicMock(return_value=torch.tensor([10.0]))

        cal = LogisticCalibrator(
            features=["logit", "log_len", "mean_conf", "n_tubes"],
            coefficients=np.array([0.0, -2.0, 0.0, 0.0]),
            intercept=0.0,
            sanity_checks=[],
        )

        res = find_first_crossing_trigger(
            classifier=classifier,
            tubes=tubes,
            patches_per_tube=patches,
            masks_per_tube=masks,
            full_logits=full_logits,
            aggregation="logistic",
            threshold=0.0,
            calibrator=cal,
            logistic_threshold=0.5,
            min_prefix_length=2,
        )
        assert res == (False, None, None, {})


# ── build_tubes_for_inference ────────────────────────────────────────────


class TestBuildTubesForInference:
    @staticmethod
    def _fd(idx, dets):
        return FrameDetections(
            frame_idx=idx, frame_id=str(idx), timestamp=None, detections=dets
        )

    @staticmethod
    def _d(cx, cy, w=0.05, h=0.05):
        return Detection(class_id=0, cx=cx, cy=cy, w=w, h=h, confidence=0.9)

    def test_without_merge_keys_matches_legacy(self):
        """No merge keys -> same result as the legacy build + filter path."""
        fds = [self._fd(i, [self._d(0.5, 0.5)]) for i in range(5)]
        legacy = filter_and_interpolate_tubes(
            build_tubes(fds, iou_threshold=0.2, max_misses=2),
            min_tube_length=2,
            min_detected_entries=2,
            interpolate_gaps=True,
        )
        new = build_tubes_for_inference(
            fds,
            iou_threshold=0.2,
            max_misses=2,
            min_tube_length=2,
            min_detected_entries=2,
            interpolate_gaps=True,
            merge_iomin=None,
            merge_prox_factor=None,
            merge_max_gap=None,
        )
        assert len(new) == len(legacy)
        assert [(t.start_frame, t.end_frame, len(t.entries)) for t in new] == [
            (t.start_frame, t.end_frame, len(t.entries)) for t in legacy
        ]

    def test_with_merge_keys_fuses_fragmented_plume(self):
        """A growing plume that build_tubes splits at the IoU jump merges back."""
        fds = [
            self._fd(0, [self._d(0.5, 0.5, 0.02, 0.02)]),
            self._fd(1, [self._d(0.5, 0.5, 0.02, 0.02)]),
            self._fd(2, [self._d(0.5, 0.5, 0.2, 0.2)]),
            self._fd(3, [self._d(0.5, 0.5, 0.2, 0.2)]),
        ]
        tubes = build_tubes_for_inference(
            fds,
            iou_threshold=0.2,
            max_misses=2,
            min_tube_length=2,
            min_detected_entries=2,
            interpolate_gaps=True,
            merge_iomin=0.3,
            merge_prox_factor=1.0,
            merge_max_gap=10,
        )
        assert len(tubes) == 1

    def test_partial_merge_keys_treated_as_no_merge(self):
        """If any merge key is None, merging is skipped (backward-compat guard)."""
        fds = [
            self._fd(0, [self._d(0.5, 0.5, 0.02, 0.02)]),
            self._fd(1, [self._d(0.5, 0.5, 0.02, 0.02)]),
            self._fd(2, [self._d(0.5, 0.5, 0.2, 0.2)]),
            self._fd(3, [self._d(0.5, 0.5, 0.2, 0.2)]),
        ]
        tubes = build_tubes_for_inference(
            fds,
            iou_threshold=0.2,
            max_misses=2,
            min_tube_length=2,
            min_detected_entries=2,
            interpolate_gaps=True,
            merge_iomin=0.3,
            merge_prox_factor=None,
            merge_max_gap=10,
        )
        assert len(tubes) == 2  # would be 1 if merge ran

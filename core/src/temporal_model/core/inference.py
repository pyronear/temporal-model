"""Pure-function helpers used by :class:`BboxTubeTemporalModel.predict`.

Each helper corresponds to one stage of the six-stage pipeline described in
``docs/specs/2026-04-15-temporal-model-protocol-design.md``. Kept separate so
``predict()`` is thin and each stage is unit-testable in isolation.
"""

from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

from .logistic_calibrator import LogisticCalibrator, extract_features
from .model_input import crop_and_resize, expand_bbox, norm_bbox_to_pixel_square
from .protocol import Frame
from .stabilize import union_window
from .tubes import (
    build_tubes,
    merge_colocated_tubes,
)
from .tubes import (
    interpolate_gaps as _interpolate_gaps,
)
from .types import Detection, FrameDetections, Tube


def pad_frames_symmetrically(
    frames: list[Frame],
    *,
    min_length: int,
) -> tuple[list[Frame], list[int]]:
    """Pad ``frames`` up to ``min_length`` by alternating prepend/append.

    Mirrors ``tracking_fsm_baseline.data.pad_sequence``: the first frame is
    prepended, then the last frame is appended, alternating until the
    sequence reaches ``min_length``.

    Returns ``(padded_frames, padded_indices)`` where ``padded_indices`` lists
    slots in ``padded_frames`` that are synthesised duplicates (not real
    captures). Empty inputs and inputs already at or above ``min_length``
    pass through with an empty index list.
    """
    if not frames or len(frames) >= min_length:
        return list(frames), []
    result = list(frames)
    real_positions = list(range(len(frames)))
    prepend = True
    while len(result) < min_length:
        src = frames[0] if prepend else frames[-1]
        if prepend:
            result.insert(0, src)
            real_positions = [p + 1 for p in real_positions]
        else:
            result.append(src)
        prepend = not prepend
    real_set = set(real_positions)
    padded_indices = [i for i in range(len(result)) if i not in real_set]
    return result, padded_indices


def pad_frames_uniform(
    frames: list[Frame],
    *,
    min_length: int,
) -> tuple[list[Frame], list[int]]:
    """Pad ``frames`` up to ``min_length`` by uniform nearest-neighbor upsampling.

    Each of ``min_length`` output slots picks the real frame whose position
    is closest under floor mapping ``i * N // M``. Real frames are repeated
    in-place, spread evenly across the padded sequence rather than clustered
    at the boundaries. Useful as an alternative to
    :func:`pad_frames_symmetrically` when the classifier is sensitive to the
    temporal distribution of duplicates (e.g. transformer attention).

    Returns ``(padded_frames, padded_indices)``. The first occurrence of each
    real source frame is treated as the "real" slot; repeats are padding.
    Empty inputs and inputs already at or above ``min_length`` pass through
    with an empty index list.
    """
    if not frames or len(frames) >= min_length:
        return list(frames), []
    n = len(frames)
    source = [i * n // min_length for i in range(min_length)]
    padded_frames = [frames[i] for i in source]
    seen: set[int] = set()
    padded_indices: list[int] = []
    for slot, src_idx in enumerate(source):
        if src_idx in seen:
            padded_indices.append(slot)
        else:
            seen.add(src_idx)
    return padded_frames, padded_indices


def run_yolo_on_frames(
    yolo_model: Any,
    frames: list[Frame],
    *,
    confidence_threshold: float,
    iou_nms: float,
    image_size: int,
    device: str | torch.device | None = None,
) -> list[FrameDetections]:
    """Run YOLO once over all frames in a single batched call.

    Args:
        yolo_model: An ultralytics ``YOLO`` instance (or any object exposing
            ``predict(list_of_paths, ...)`` with the same return shape).
        frames: Temporally ordered Pyronear :class:`Frame` objects.
        confidence_threshold: Minimum detection confidence.
        iou_nms: IoU threshold for YOLO's internal NMS.
        image_size: Inference resolution passed to YOLO.
        device: Torch device for YOLO inference. ``None`` lets ultralytics
            auto-pick (defaults to CPU).

    Returns:
        One :class:`FrameDetections` per input frame (possibly with zero
        detections), in the same order, with ``frame_idx`` = position.
    """
    if not frames:
        return []

    paths = [str(f.image_path) for f in frames]
    predict_kwargs: dict[str, Any] = dict(
        conf=confidence_threshold,
        iou=iou_nms,
        imgsz=image_size,
        verbose=False,
    )
    if device is not None:
        predict_kwargs["device"] = str(device)
    results = yolo_model.predict(paths, **predict_kwargs)

    out: list[FrameDetections] = []
    for idx, (frame, pred) in enumerate(zip(frames, results, strict=True)):
        detections: list[Detection] = []
        boxes = pred.boxes
        if boxes is not None and len(boxes) > 0:
            xywhn = boxes.xywhn
            confs = boxes.conf
            cls = boxes.cls
            for i in range(len(boxes)):
                row = xywhn[i].tolist()
                detections.append(
                    Detection(
                        class_id=int(cls[i].item()),
                        cx=row[0],
                        cy=row[1],
                        w=row[2],
                        h=row[3],
                        confidence=float(confs[i].item()),
                    )
                )
        out.append(
            FrameDetections(
                frame_idx=idx,
                frame_id=frame.frame_id,
                timestamp=frame.timestamp,
                detections=detections,
            )
        )
    return out


def filter_and_interpolate_tubes(
    tubes: list[Tube],
    *,
    min_tube_length: int,
    min_detected_entries: int,
    interpolate_gaps: bool,
) -> list[Tube]:
    """Filter tubes by length / observation count, then optionally interpolate gaps.

    Args:
        tubes: Candidate tubes (output of
            :func:`~temporal_model.core.tubes.build_tubes`).
        min_tube_length: Keep tubes where
            ``end_frame - start_frame + 1 >= min_tube_length``.
        min_detected_entries: Keep tubes with at least this many non-gap entries.
        interpolate_gaps: If True, fill gap entries in surviving tubes via
            :func:`~temporal_model.core.tubes.interpolate_gaps`.

    Returns:
        Surviving tubes in original order.
    """
    survivors: list[Tube] = []
    for t in tubes:
        length = t.end_frame - t.start_frame + 1
        if length < min_tube_length:
            continue
        n_obs = sum(1 for e in t.entries if e.detection is not None)
        if n_obs < min_detected_entries:
            continue
        if interpolate_gaps:
            _interpolate_gaps(t)
        survivors.append(t)
    return survivors


def crop_tube_patches(
    tube: Tube,
    frames: list[Frame],
    *,
    context_factor: float,
    patch_size: int,
    max_frames: int,
    normalization_mean: list[float],
    normalization_std: list[float],
    stabilize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Crop patches for a single tube, padded/truncated to ``max_frames``.

    Matches ``TubePatchDataset.__getitem__`` exactly: PIL→uint8 array,
    ``expand_bbox → norm_bbox_to_pixel_square → crop_and_resize``, then
    ``to_tensor`` (CHW, float32, [0,1]), then mean/std normalization.
    """
    frame_by_idx = {i: f for i, f in enumerate(frames)}

    n = min(len(tube.entries), max_frames)
    patches = torch.zeros(max_frames, 3, patch_size, patch_size, dtype=torch.float32)
    mask = torch.zeros(max_frames, dtype=torch.bool)
    mean_t = torch.tensor(normalization_mean).view(3, 1, 1)
    std_t = torch.tensor(normalization_std).view(3, 1, 1)

    window = None
    if stabilize:
        observed = [
            (e.detection.cx, e.detection.cy, e.detection.w, e.detection.h)
            for e in tube.entries
            if e.detection is not None and not e.is_gap
        ]
        if not observed:
            observed = [
                (e.detection.cx, e.detection.cy, e.detection.w, e.detection.h)
                for e in tube.entries
                if e.detection is not None
            ]
        window = union_window(observed)

    for slot, entry in enumerate(tube.entries[:n]):
        det = entry.detection
        if det is None:
            # Shouldn't happen post-interpolation; leave zero-padded + mask=False.
            continue
        frame = frame_by_idx[entry.frame_idx]
        image = np.array(Image.open(frame.image_path).convert("RGB"))
        img_h, img_w, _ = image.shape

        box_src = window if stabilize else (det.cx, det.cy, det.w, det.h)
        cx, cy, w, h = expand_bbox(
            box_src[0], box_src[1], box_src[2], box_src[3], context_factor
        )
        box = norm_bbox_to_pixel_square(cx, cy, w, h, img_w, img_h)
        patch_np = crop_and_resize(image, box, patch_size)
        patch_t = to_tensor(Image.fromarray(patch_np))  # CHW float32 [0,1]
        patches[slot] = (patch_t - mean_t) / std_t
        mask[slot] = True

    return patches, mask


def score_tubes(
    classifier: Any,
    *,
    patches_per_tube: list[torch.Tensor],
    masks_per_tube: list[torch.Tensor],
) -> torch.Tensor:
    """Run one batched classifier forward over all tubes.

    Args:
        classifier: A callable ``(patches[N,T,3,H,W], mask[N,T]) -> logits[N]``.
        patches_per_tube: One ``[T, 3, H, W]`` tensor per tube.
        masks_per_tube: One ``[T]`` bool tensor per tube.

    Returns:
        ``Tensor[N]`` of logits (empty tensor if no tubes).
    """
    if not patches_per_tube:
        return torch.zeros(0)
    patches = torch.stack(patches_per_tube, dim=0)
    mask = torch.stack(masks_per_tube, dim=0)
    with torch.no_grad():
        return classifier(patches, mask)


def find_first_crossing_trigger(
    *,
    classifier: Any,
    tubes: list[Tube],
    patches_per_tube: list[torch.Tensor],
    masks_per_tube: list[torch.Tensor],
    full_logits: torch.Tensor,
    aggregation: str = "max_logit",
    threshold: float,
    calibrator: LogisticCalibrator | None = None,
    logistic_threshold: float = 0.5,
    min_prefix_length: int,
) -> tuple[bool, int | None, int | None, dict]:
    """Find the earliest frame at which the aggregation rule would have fired.

    Under the full-tube guard (D2), only tubes whose *full-length*
    decision is positive qualify. For each qualifying tube, prefixes of
    length ``L = min_prefix_length .. len(tube.entries)`` are scored
    serially; the first L whose decision is positive gives that tube's
    first-crossing ``frame_idx = tube.entries[L-1].frame_idx``. The
    sequence-level trigger is the qualifying tube with the earliest
    first-crossing frame (tie-break: smallest ``tube_id``).

    ``is_positive`` is preserved bit-for-bit versus
    :func:`pick_winner_and_trigger`: the sequence is positive iff any
    tube's full-length decision is positive.

    Micro-optimisation: at ``L = len(tube.entries)`` the prefix logit is
    exactly ``full_logits[i]`` — we reuse it rather than re-running the
    classifier.

    Args:
        classifier: Callable ``(patches[1,T,3,H,W], mask[1,T]) -> logits[1]``.
        tubes: Kept tubes, aligned with ``full_logits``.
        patches_per_tube: One ``[max_frames, 3, H, W]`` tensor per tube.
        masks_per_tube: One ``[max_frames]`` bool tensor per tube.
        full_logits: Output of :func:`score_tubes` on the full tubes.
        aggregation: ``"max_logit"`` or ``"logistic"``.
        threshold: Raw logit threshold (``max_logit`` only).
        calibrator: Required when ``aggregation == "logistic"``.
        logistic_threshold: Probability threshold (``logistic`` only).
        min_prefix_length: Smallest prefix length to score. Must equal
            the inference-time ``infer_min_tube_length``.

    Returns:
        Tuple ``(is_positive, trigger_frame_index, trigger_tube_id,
        per_tube_first_crossing)``. The trailing dict maps
        ``tube_id -> {"crossing_frame": int, "prefix_length": int}`` for
        every qualifying tube.

    Raises:
        ValueError: unknown ``aggregation`` or ``"logistic"`` without a
            calibrator.
    """
    if not tubes:
        return False, None, None, {}

    if aggregation == "max_logit":

        def decides_positive(logit: float, _tube_prefix: Tube, _n_tubes: int) -> bool:
            return logit >= threshold
    elif aggregation == "logistic":
        if calibrator is None:
            raise ValueError("aggregation='logistic' requires a fitted calibrator")

        def decides_positive(logit: float, tube_prefix: Tube, n_tubes: int) -> bool:
            tube_dict = {
                "logit": logit,
                "start_frame": tube_prefix.start_frame,
                "end_frame": tube_prefix.end_frame,
                "entries": [
                    {
                        "confidence": (
                            e.detection.confidence if e.detection is not None else None
                        )
                    }
                    for e in tube_prefix.entries
                ],
            }
            features = extract_features(tube_dict, n_tubes=n_tubes)
            return bool(calibrator.predict_proba(features) >= logistic_threshold)
    else:
        raise ValueError(f"unknown aggregation: {aggregation!r}")

    n_tubes = len(tubes)

    qualifying_indices: list[int] = [
        i
        for i, tube in enumerate(tubes)
        if decides_positive(float(full_logits[i].item()), tube, n_tubes)
    ]
    if not qualifying_indices:
        return False, None, None, {}

    per_tube: dict[int, dict] = {}
    for i in qualifying_indices:
        tube = tubes[i]
        full_len = len(tube.entries)
        assert full_len >= min_prefix_length, (
            f"tube {tube.tube_id} len {full_len} < min_prefix_length "
            f"{min_prefix_length}"
        )

        patches_i = patches_per_tube[i]
        mask_i = masks_per_tube[i]

        crossed = False
        for L in range(min_prefix_length, full_len + 1):
            if full_len == L:
                prefix_logit = float(full_logits[i].item())
            else:
                prefix_mask = mask_i.clone()
                prefix_mask[L:] = False
                with torch.no_grad():
                    out = classifier(patches_i.unsqueeze(0), prefix_mask.unsqueeze(0))
                prefix_logit = float(out[0].item())

            prefix_entries = tube.entries[:L]
            prefix_tube = Tube(
                tube_id=tube.tube_id,
                entries=prefix_entries,
                start_frame=prefix_entries[0].frame_idx,
                end_frame=prefix_entries[-1].frame_idx,
            )

            if decides_positive(prefix_logit, prefix_tube, n_tubes):
                per_tube[tube.tube_id] = {
                    "crossing_frame": prefix_entries[-1].frame_idx,
                    "prefix_length": L,
                }
                crossed = True
                break

        assert crossed, (
            f"D2 invariant violated: qualifying tube {tube.tube_id} "
            f"produced no crossing prefix"
        )

    trigger_tube_id = min(
        per_tube,
        key=lambda tid: (per_tube[tid]["crossing_frame"], tid),
    )
    trigger = per_tube[trigger_tube_id]["crossing_frame"]
    return True, trigger, trigger_tube_id, per_tube


def build_tubes_for_inference(
    frame_detections: list[FrameDetections],
    *,
    iou_threshold: float,
    max_misses: int,
    min_tube_length: int,
    min_detected_entries: int,
    interpolate_gaps: bool,
    merge_iomin: float | None,
    merge_prox_factor: float | None,
    merge_max_gap: int | None,
) -> list[Tube]:
    """Run the full inference tube pipeline.

    When all three ``merge_*`` thresholds are provided, the merge pass runs
    between two filter steps (filter-before-merge prevents resurrecting
    sub-threshold noise; the second filter+interpolate prepares the tubes for
    the classifier). If any ``merge_*`` is ``None`` the merge is skipped and
    the pipeline collapses to the legacy ``build_tubes`` -> single
    ``filter_and_interpolate_tubes`` path (bit-for-bit backward compatible).
    """
    tubes = build_tubes(
        frame_detections, iou_threshold=iou_threshold, max_misses=max_misses
    )
    merge_active = (
        merge_iomin is not None
        and merge_prox_factor is not None
        and merge_max_gap is not None
    )
    if merge_active:
        tubes = filter_and_interpolate_tubes(
            tubes,
            min_tube_length=min_tube_length,
            min_detected_entries=min_detected_entries,
            interpolate_gaps=False,
        )
        tubes = merge_colocated_tubes(
            tubes,
            merge_iomin=merge_iomin,
            merge_prox_factor=merge_prox_factor,
            merge_max_gap=merge_max_gap,
        )
    return filter_and_interpolate_tubes(
        tubes,
        min_tube_length=min_tube_length,
        min_detected_entries=min_detected_entries,
        interpolate_gaps=interpolate_gaps,
    )

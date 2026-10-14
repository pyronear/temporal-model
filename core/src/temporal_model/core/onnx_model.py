"""Torch-free runtime for the tube classifier, backed by onnxruntime.

Loads the ``model_onnx.zip`` built by :mod:`temporal_model.core.export_onnx`
and runs the shared :class:`~temporal_model.core.pipeline.TubePipelineModel`
pipeline with the exported classifier. There is no bundled detector: callers
(e.g. an edge device already running its own YOLO) must pass
``frame_detections`` for every frame.

Format 2 packages ship the classifier as two graphs — ``backbone.onnx``
(one patch -> one embedding) and ``head.onnx`` (a tube of embeddings ->
logit) — so the backbone, which is ~all of the cost, runs only on real
(unmasked) slots, and a serving layer that re-scores a sliding window every
round can keep per-frame embeddings across calls (``embedding_cache_size``).
Format 1 packages (one monolithic ``classifier.onnx``) still load, without
those savings.

This module must stay importable with only the ``onnx`` extra installed —
no torch, timm or ultralytics anywhere on its import path.
Design: ``docs/specs/2026-09-09-onnx-export-design.md`` and
``docs/specs/2026-10-14-onnx-embedding-cache-design.md``.
"""

import json
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np
import yaml
from PIL import Image

from .crop import crop_and_resize
from .inference import normalize_patch, tube_crop_boxes
from .logistic_calibrator import LogisticCalibrator
from .pipeline import (
    TubePipelineModel,
    UncalibratedModelError,
    aggregation_of,
    require_calibrated,
)
from .protocol import Frame
from .stage_timer import StageTimer, stage_ctx
from .types import Tube

__all__ = [
    "ONNX_MODEL_FILENAME",
    "OnnxPackage",
    "OnnxTemporalModel",
    "UncalibratedModelError",  # torch-free import path for [onnx]-only installs
    "load_onnx_package",
    "score_split",
]

ONNX_FORMAT_VERSION = 2
SUPPORTED_FORMAT_VERSIONS = (1, 2)
ONNX_MODEL_FILENAME = "model_onnx.zip"
MANIFEST_FILENAME = "manifest.yaml"
CLASSIFIER_ONNX_FILENAME = "classifier.onnx"  # format 1: backbone + head fused
BACKBONE_ONNX_FILENAME = "backbone.onnx"
HEAD_ONNX_FILENAME = "head.onnx"
INPUT_PATCHES = "patches"
INPUT_MASK = "mask"
INPUT_FEATURES = "features"
OUTPUT_FEATURES = "features"
OUTPUT_LOGIT = "logit"
_CPU_PROVIDERS = ["CPUExecutionProvider"]

# (frame_id, pixel crop box): what fully determines one slot's embedding.
CacheKey = tuple[str, tuple[int, int, int, int]]


@dataclass
class OnnxPackage:
    """A loaded ONNX package: sessions, config, calibrator and manifest.

    Format 2 fills ``backbone`` + ``head``; format 1 fills ``classifier``.
    """

    config: dict[str, Any]
    calibrator: LogisticCalibrator | None
    manifest: dict[str, Any]
    backbone: Any = None  # onnxruntime.InferenceSession
    head: Any = None
    classifier: Any = None


def load_onnx_package(
    package_path: Path,
    *,
    allow_uncalibrated: bool = False,
    providers: list[str] | None = None,
) -> OnnxPackage:
    """Load a ``model_onnx.zip`` (read in memory, nothing is extracted).

    Raises:
        FileNotFoundError: if ``package_path`` does not exist.
        KeyError: if the archive is missing expected entries.
        ValueError: if ``format_version`` is unsupported.
        UncalibratedModelError: if the package is uncalibrated and
            ``allow_uncalibrated`` is False.
    """
    import onnxruntime as ort  # noqa: PLC0415  # keep `import temporal_model.core` light

    if not package_path.exists():
        raise FileNotFoundError(f"Archive not found: {package_path}")

    with zipfile.ZipFile(package_path) as zf:
        names = zf.namelist()
        if MANIFEST_FILENAME not in names:
            raise KeyError(f"Archive missing {MANIFEST_FILENAME}")
        manifest = yaml.safe_load(zf.read(MANIFEST_FILENAME))
        version = manifest.get("format_version")
        if version not in SUPPORTED_FORMAT_VERSIONS:
            raise ValueError(
                f"Unsupported format_version {version} "
                f"(expected one of {SUPPORTED_FORMAT_VERSIONS})"
            )
        graph_keys = (
            ("classifier_onnx",) if version == 1 else ("backbone_onnx", "head_onnx")
        )
        config_name = manifest["config"]
        calibrator_name = manifest.get("logistic_calibrator")
        for n in (*(manifest[k] for k in graph_keys), config_name, calibrator_name):
            if n is not None and n not in names:
                raise KeyError(f"Archive missing {n}")
        config = yaml.safe_load(zf.read(config_name))
        calibrator = None
        if calibrator_name is not None:
            calibrator = LogisticCalibrator.from_dict(
                json.loads(zf.read(calibrator_name))
            )
            calibrator.verify_sanity_checks()
        graphs = {k: zf.read(manifest[k]) for k in graph_keys}

    if not allow_uncalibrated:
        require_calibrated(
            calibrator, aggregation_of(config), context="load_onnx_package"
        )

    def session(model_bytes: bytes) -> Any:
        return ort.InferenceSession(
            model_bytes,
            providers=providers if providers is not None else _CPU_PROVIDERS,
        )

    sessions = {k.removesuffix("_onnx"): session(b) for k, b in graphs.items()}
    return OnnxPackage(
        config=config, calibrator=calibrator, manifest=manifest, **sessions
    )


def embed(backbone: Any, patches: np.ndarray) -> np.ndarray:
    """``patches[M,3,P,P] float32 -> features[M,D] float32`` (one run)."""
    (feats,) = backbone.run([OUTPUT_FEATURES], {INPUT_PATCHES: patches})
    return np.asarray(feats, dtype=np.float32)


def score_features(head: Any, features: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """``features[N,T,D] float32, mask[N,T] bool -> logits[N] float32``."""
    (logits,) = head.run([OUTPUT_LOGIT], {INPUT_FEATURES: features, INPUT_MASK: mask})
    return np.asarray(logits, dtype=np.float32)


def score_split(
    backbone: Any, head: Any, patches: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Score tubes with the split graphs, embedding only the unmasked slots.

    Masked slots get a zero embedding: the head drops them from attention
    (key padding), so their content never reaches the logit.
    """
    n, t = mask.shape
    feat_dim = int(backbone.get_outputs()[0].shape[1])
    features = np.zeros((n, t, feat_dim), dtype=np.float32)
    if mask.any():
        features[mask] = embed(backbone, patches[mask])
    return score_features(head, features, mask)


class OnnxTemporalModel(TubePipelineModel):
    """Tube classifier served by onnxruntime; detections must be supplied.

    ``embedding_cache_size`` (format 2 only) keeps the last N per-slot
    embeddings keyed by ``(frame_id, pixel crop box)``: a caller that
    re-submits a sliding window of the same frames only pays the backbone
    for slots it has not embedded yet. Frame ids must therefore be unique
    per image. Not thread-safe: score from one thread at a time.
    """

    def __init__(
        self,
        *,
        config: dict[str, Any],
        calibrator: LogisticCalibrator | None = None,
        backbone: Any = None,
        head: Any = None,
        classifier: Any = None,
        embedding_cache_size: int = 0,
    ) -> None:
        super().__init__(config=config, calibrator=calibrator)
        if (backbone is None or head is None) == (classifier is None):
            raise ValueError(
                "pass either backbone + head (format 2) or classifier (format 1)"
            )
        self._backbone = backbone
        self._head = head
        self._classifier = classifier
        self._cache: OrderedDict[CacheKey, np.ndarray] | None = None
        if embedding_cache_size > 0:
            if backbone is None:
                raise ValueError("embedding cache needs a split (format 2) package")
            self._cache = OrderedDict()
        self._cache_size = embedding_cache_size
        self.cache_hits = 0
        self.cache_misses = 0

    @classmethod
    def from_package(
        cls,
        package_path: Path,
        *,
        allow_uncalibrated: bool = False,
        providers: list[str] | None = None,
        embedding_cache_size: int = 0,
    ) -> Self:
        pkg = load_onnx_package(
            package_path, allow_uncalibrated=allow_uncalibrated, providers=providers
        )
        return cls(
            config=pkg.config,
            calibrator=pkg.calibrator,
            backbone=pkg.backbone,
            head=pkg.head,
            classifier=pkg.classifier,
            embedding_cache_size=embedding_cache_size,
        )

    @property
    def has_split_graphs(self) -> bool:
        """True for a format 2 package (backbone + head, cache-capable)."""
        return self._backbone is not None

    def _score(self, patches: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Score all tubes in one pass (the graphs have a dynamic batch axis)."""
        if self._classifier is not None:
            (logits,) = self._classifier.run(
                [OUTPUT_LOGIT], {INPUT_PATCHES: patches, INPUT_MASK: mask}
            )
            return np.asarray(logits, dtype=np.float32)
        return score_split(self._backbone, self._head, patches, mask)

    def _cached(self, key: CacheKey) -> np.ndarray | None:
        if self._cache is None:
            return None
        feat = self._cache.get(key)
        if feat is not None:
            self._cache.move_to_end(key)
        return feat

    def _remember(self, key: CacheKey, feat: np.ndarray) -> None:
        if self._cache is None:
            return
        self._cache[key] = feat
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    def _crop_and_score(
        self,
        kept: list[Tube],
        frames: list[Frame],
        *,
        timer: StageTimer | None,
        keep_patches: bool,
    ) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        """Crop, embed and score slot by slot, reusing cached embeddings.

        Falls back to the generic path (every patch materialised, every
        slot embedded) for format 1 packages and when the caller needs the
        patches back (trigger search).
        """
        if keep_patches or self._backbone is None:
            return super()._crop_and_score(
                kept, frames, timer=timer, keep_patches=keep_patches
            )
        mi = self._cfg["model_input"]
        max_frames = int(self._cfg["classifier"]["max_frames"])
        patch_size = int(mi["patch_size"])
        mean = np.asarray(mi["normalization"]["mean"], dtype=np.float32).reshape(
            3, 1, 1
        )
        std = np.asarray(mi["normalization"]["std"], dtype=np.float32).reshape(3, 1, 1)
        feat_dim = int(self._backbone.get_outputs()[0].shape[1])

        features = np.zeros((len(kept), max_frames, feat_dim), dtype=np.float32)
        mask = np.zeros((len(kept), max_frames), dtype=bool)
        misses: list[tuple[int, int, CacheKey]] = []
        patches: list[np.ndarray] = []
        decoded: dict[str, np.ndarray] = {}  # one JPEG decode per frame per call
        with stage_ctx(timer, "crop"):
            for i, tube in enumerate(kept):
                slots = tube_crop_boxes(
                    tube,
                    frames,
                    context_factor=mi["context_factor"],
                    max_frames=max_frames,
                    stabilize=mi.get("stabilize", True),
                )
                for slot, item in enumerate(slots):
                    if item is None:
                        continue
                    frame, box = item
                    mask[i, slot] = True
                    key = (frame.frame_id, box)
                    feat = self._cached(key)
                    if feat is not None:
                        features[i, slot] = feat
                        self.cache_hits += 1
                        continue
                    self.cache_misses += 1
                    image = decoded.get(frame.frame_id)
                    if image is None:
                        image = np.array(Image.open(frame.image_path).convert("RGB"))
                        decoded[frame.frame_id] = image
                    patches.append(
                        normalize_patch(
                            crop_and_resize(image, box, patch_size), mean, std
                        )
                    )
                    misses.append((i, slot, key))

        with stage_ctx(timer, "classifier"):
            if misses:
                fresh = embed(self._backbone, np.stack(patches, axis=0))
                for (i, slot, key), feat in zip(misses, fresh, strict=True):
                    features[i, slot] = feat
                    self._remember(key, feat)
            logits = score_features(self._head, features, mask)
        return logits, [], []

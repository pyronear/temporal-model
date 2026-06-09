"""Tests for BboxTubeTemporalModel packaging."""

import json
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import yaml

from temporal_model.core.detector import load_detector
from temporal_model.core.logistic_calibrator import LogisticCalibrator
from temporal_model.core.package import (
    CLASSIFIER_CKPT_FILENAME,
    CONFIG_FILENAME,
    FORMAT_VERSION,
    LOGISTIC_CALIBRATOR_FILENAME,
    MANIFEST_FILENAME,
    YOLO_WEIGHTS_FILENAME,
    UncalibratedModelError,
    build_model_package,
    is_calibrated,
    load_model_package,
    require_calibrated,
)
from temporal_model.core.temporal_classifier import TemporalSmokeClassifier

SAMPLE_CONFIG: dict = {
    "infer": {"confidence_threshold": 0.01, "iou_nms": 0.2, "image_size": 1024},
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
        "max_frames": 20,
        "pretrained": False,
        "global_pool": "token",
        "transformer_num_layers": 1,
        "transformer_num_heads": 6,
        "transformer_ffn_dim": 64,
        "transformer_dropout": 0.0,
    },
    "decision": {
        "aggregation": "max_logit",
        "threshold": 0.42,
        "target_recall": 0.95,
        "trigger_rule": "end_of_winner",
    },
}


@pytest.fixture()
def dummy_yolo_weights(tmp_path: Path) -> Path:
    p = tmp_path / "yolo.pt"
    p.write_bytes(b"fake-yolo")
    return p


@pytest.fixture()
def dummy_classifier_ckpt(tmp_path: Path) -> Path:
    p = tmp_path / "classifier.ckpt"
    p.write_bytes(b"fake-classifier")
    return p


@pytest.fixture()
def built_archive(
    tmp_path: Path, dummy_yolo_weights: Path, dummy_classifier_ckpt: Path
) -> Path:
    out = tmp_path / "model.zip"
    build_model_package(
        yolo_weights_path=dummy_yolo_weights,
        classifier_ckpt_path=dummy_classifier_ckpt,
        config=SAMPLE_CONFIG,
        variant="vit_dinov2_finetune",
        output_path=out,
        allow_uncalibrated=True,
    )
    return out


class TestBuildArchive:
    def test_output_exists(self, built_archive: Path) -> None:
        assert built_archive.exists()

    def test_is_valid_zip(self, built_archive: Path) -> None:
        assert zipfile.is_zipfile(built_archive)

    def test_contains_all_entries(self, built_archive: Path) -> None:
        with zipfile.ZipFile(built_archive, "r") as zf:
            names = set(zf.namelist())
        assert names == {
            MANIFEST_FILENAME,
            YOLO_WEIGHTS_FILENAME,
            CLASSIFIER_CKPT_FILENAME,
            CONFIG_FILENAME,
        }

    def test_yolo_weights_preserved(
        self, built_archive: Path, dummy_yolo_weights: Path
    ) -> None:
        with zipfile.ZipFile(built_archive, "r") as zf:
            assert zf.read(YOLO_WEIGHTS_FILENAME) == dummy_yolo_weights.read_bytes()

    def test_classifier_ckpt_preserved(
        self, built_archive: Path, dummy_classifier_ckpt: Path
    ) -> None:
        with zipfile.ZipFile(built_archive, "r") as zf:
            assert (
                zf.read(CLASSIFIER_CKPT_FILENAME) == dummy_classifier_ckpt.read_bytes()
            )


class TestManifest:
    def test_format_version(self, built_archive: Path) -> None:
        with zipfile.ZipFile(built_archive, "r") as zf:
            manifest = yaml.safe_load(zf.read(MANIFEST_FILENAME))
        assert manifest["format_version"] == FORMAT_VERSION

    def test_variant_recorded(self, built_archive: Path) -> None:
        with zipfile.ZipFile(built_archive, "r") as zf:
            manifest = yaml.safe_load(zf.read(MANIFEST_FILENAME))
        assert manifest["variant"] == "vit_dinov2_finetune"

    def test_file_pointers(self, built_archive: Path) -> None:
        with zipfile.ZipFile(built_archive, "r") as zf:
            manifest = yaml.safe_load(zf.read(MANIFEST_FILENAME))
        assert manifest["yolo_weights"] == YOLO_WEIGHTS_FILENAME
        assert manifest["classifier_checkpoint"] == CLASSIFIER_CKPT_FILENAME
        assert manifest["config"] == CONFIG_FILENAME


class TestConfigRoundTrip:
    def test_config_bytes_match(self, built_archive: Path) -> None:
        with zipfile.ZipFile(built_archive, "r") as zf:
            loaded = yaml.safe_load(zf.read(CONFIG_FILENAME))
        assert loaded == SAMPLE_CONFIG


class TestBuildMissingWeightsRaises:
    def test_missing_yolo(self, tmp_path: Path, dummy_classifier_ckpt: Path) -> None:
        with pytest.raises(FileNotFoundError):
            build_model_package(
                yolo_weights_path=tmp_path / "nope.pt",
                classifier_ckpt_path=dummy_classifier_ckpt,
                config=SAMPLE_CONFIG,
                variant="vit_dinov2_finetune",
                output_path=tmp_path / "out.zip",
            )

    def test_missing_classifier_ckpt(
        self, tmp_path: Path, dummy_yolo_weights: Path
    ) -> None:
        with pytest.raises(FileNotFoundError):
            build_model_package(
                yolo_weights_path=dummy_yolo_weights,
                classifier_ckpt_path=tmp_path / "nope.ckpt",
                config=SAMPLE_CONFIG,
                variant="vit_dinov2_finetune",
                output_path=tmp_path / "out.zip",
            )


# Build a real tiny classifier state_dict so load_model_package can construct
# and populate a TemporalSmokeClassifier from it.
@pytest.fixture()
def real_tiny_classifier_ckpt(tmp_path: Path) -> Path:
    """A Lightning-style ckpt holding a TemporalSmokeClassifier state_dict.

    Uses the smaller vit_small_patch16_224 backbone for speed; the production
    variant is vit_small_patch14_dinov2.lvd142m.
    """
    model = TemporalSmokeClassifier(
        backbone="vit_small_patch16_224",
        pretrained=False,
        global_pool="token",
        transformer_num_layers=1,
        transformer_num_heads=6,
        transformer_ffn_dim=64,
        transformer_dropout=0.0,
        max_frames=20,
    )
    # Lightning ckpt schema: torch.save({"state_dict": {...}, ...})
    state_dict = {f"model.{k}": v for k, v in model.state_dict().items()}
    ckpt_path = tmp_path / "tiny.ckpt"
    torch.save({"state_dict": state_dict}, ckpt_path)
    return ckpt_path


@pytest.fixture()
def real_tiny_config() -> dict:
    cfg = {k: dict(v) if isinstance(v, dict) else v for k, v in SAMPLE_CONFIG.items()}
    cfg["classifier"] = dict(cfg["classifier"])
    return cfg


@pytest.fixture()
def real_tiny_archive(
    tmp_path: Path,
    dummy_yolo_weights: Path,
    real_tiny_classifier_ckpt: Path,
    real_tiny_config: dict,
) -> Path:
    out = tmp_path / "tiny_model.zip"
    build_model_package(
        yolo_weights_path=dummy_yolo_weights,
        classifier_ckpt_path=real_tiny_classifier_ckpt,
        config=real_tiny_config,
        variant="tiny",
        output_path=out,
        allow_uncalibrated=True,
    )
    return out


class TestLoadRoundtrip:
    @patch("temporal_model.core.package.load_yolo")
    def test_config_passthrough(
        self,
        mock_yolo: MagicMock,
        real_tiny_archive: Path,
        tmp_path: Path,
        real_tiny_config: dict,
    ) -> None:
        mock_yolo.return_value = MagicMock(name="FakeYOLO")
        pkg = load_model_package(
            real_tiny_archive, extract_dir=tmp_path / "ext", allow_uncalibrated=True
        )
        assert pkg.config == real_tiny_config

    @patch("temporal_model.core.package.load_yolo")
    def test_yolo_returned(
        self, mock_yolo: MagicMock, real_tiny_archive: Path, tmp_path: Path
    ) -> None:
        sentinel = MagicMock(name="FakeYOLO")
        mock_yolo.return_value = sentinel
        pkg = load_model_package(
            real_tiny_archive, extract_dir=tmp_path / "ext", allow_uncalibrated=True
        )
        assert pkg.yolo_model is sentinel

    @patch("temporal_model.core.package.load_yolo")
    def test_classifier_forward_runs(
        self,
        mock_yolo: MagicMock,
        real_tiny_archive: Path,
        tmp_path: Path,
    ) -> None:
        mock_yolo.return_value = MagicMock(name="FakeYOLO")
        pkg = load_model_package(
            real_tiny_archive, extract_dir=tmp_path / "ext", allow_uncalibrated=True
        )

        patches = torch.zeros(1, 4, 3, 224, 224)
        mask = torch.tensor([[True, True, True, True]])
        with torch.no_grad():
            logit = pkg.classifier(patches, mask)
        assert logit.shape == (1,)


class TestLoadRejectsBadArchive:
    @patch("temporal_model.core.package.load_yolo")
    def test_missing_manifest(self, mock_yolo: MagicMock, tmp_path: Path) -> None:
        bad = tmp_path / "bad.zip"
        with zipfile.ZipFile(bad, "w") as zf:
            zf.writestr(CONFIG_FILENAME, "infer: {}")
        with pytest.raises(KeyError):
            load_model_package(bad, extract_dir=tmp_path / "ext")

    @patch("temporal_model.core.package.load_yolo")
    def test_unsupported_format_version(
        self, mock_yolo: MagicMock, tmp_path: Path
    ) -> None:
        bad = tmp_path / "bad.zip"
        manifest = {
            "format_version": 99,
            "variant": "x",
            "yolo_weights": YOLO_WEIGHTS_FILENAME,
            "classifier_checkpoint": CLASSIFIER_CKPT_FILENAME,
            "config": CONFIG_FILENAME,
        }
        with zipfile.ZipFile(bad, "w") as zf:
            zf.writestr(MANIFEST_FILENAME, yaml.dump(manifest))
            zf.writestr(YOLO_WEIGHTS_FILENAME, b"x")
            zf.writestr(CLASSIFIER_CKPT_FILENAME, b"x")
            zf.writestr(CONFIG_FILENAME, "{}")
        with pytest.raises(ValueError, match="format_version"):
            load_model_package(bad, extract_dir=tmp_path / "ext")


def _make_calibrator() -> LogisticCalibrator:
    """Calibrator whose sanity check is computed so verify_sanity_checks passes."""
    coefs = np.array([0.5, 1.5, 2.5, 0.0])
    intercept = -3.0
    feats = np.array([1.0, 2.0, 0.5, 2.0])
    z = float(feats @ coefs) + intercept
    prob = float(1.0 / (1.0 + np.exp(-z)))
    return LogisticCalibrator(
        features=["logit", "log_len", "mean_conf", "n_tubes"],
        coefficients=coefs,
        intercept=intercept,
        sanity_checks=[{"features": feats.tolist(), "prob": prob}],
    )


class TestCalibratorBundling:
    @patch("temporal_model.core.package.load_yolo")
    def test_package_without_calibrator_has_no_entry(
        self,
        mock_yolo: MagicMock,
        real_tiny_archive: Path,
        tmp_path: Path,
    ) -> None:
        mock_yolo.return_value = MagicMock(name="FakeYOLO")
        with zipfile.ZipFile(real_tiny_archive, "r") as zf:
            assert LOGISTIC_CALIBRATOR_FILENAME not in zf.namelist()
            manifest = yaml.safe_load(zf.read(MANIFEST_FILENAME))
            assert "logistic_calibrator" not in manifest

        pkg = load_model_package(
            real_tiny_archive, extract_dir=tmp_path / "ext", allow_uncalibrated=True
        )
        assert pkg.calibrator is None

    @patch("temporal_model.core.package.load_yolo")
    def test_package_with_calibrator_round_trips(
        self,
        mock_yolo: MagicMock,
        tmp_path: Path,
        dummy_yolo_weights: Path,
        real_tiny_classifier_ckpt: Path,
        real_tiny_config: dict,
    ) -> None:
        mock_yolo.return_value = MagicMock(name="FakeYOLO")
        cal = _make_calibrator()

        out = tmp_path / "m.zip"
        build_model_package(
            yolo_weights_path=dummy_yolo_weights,
            classifier_ckpt_path=real_tiny_classifier_ckpt,
            config=real_tiny_config,
            variant="tiny",
            output_path=out,
            calibrator=cal,
            allow_uncalibrated=True,
        )

        with zipfile.ZipFile(out, "r") as zf:
            names = zf.namelist()
            assert LOGISTIC_CALIBRATOR_FILENAME in names
            manifest = yaml.safe_load(zf.read(MANIFEST_FILENAME))
            assert manifest["logistic_calibrator"] == LOGISTIC_CALIBRATOR_FILENAME

        pkg = load_model_package(
            out, extract_dir=tmp_path / "ext", allow_uncalibrated=True
        )
        assert pkg.calibrator is not None
        assert pkg.calibrator.features == cal.features
        np.testing.assert_allclose(pkg.calibrator.coefficients, cal.coefficients)
        assert pkg.calibrator.intercept == cal.intercept
        pkg.calibrator.verify_sanity_checks()  # no raise

    @patch("temporal_model.core.package.load_yolo")
    def test_load_rejects_tampered_calibrator(
        self,
        mock_yolo: MagicMock,
        tmp_path: Path,
        dummy_yolo_weights: Path,
        real_tiny_classifier_ckpt: Path,
        real_tiny_config: dict,
    ) -> None:
        mock_yolo.return_value = MagicMock(name="FakeYOLO")
        cal = _make_calibrator()

        out = tmp_path / "m.zip"
        build_model_package(
            yolo_weights_path=dummy_yolo_weights,
            classifier_ckpt_path=real_tiny_classifier_ckpt,
            config=real_tiny_config,
            variant="tiny",
            output_path=out,
            calibrator=cal,
            allow_uncalibrated=True,
        )

        # Rewrite the zip with tampered coefficients.
        tampered_path = tmp_path / "tampered.zip"
        with (
            zipfile.ZipFile(out, "r") as src,
            zipfile.ZipFile(tampered_path, "w") as dst,
        ):
            for name in src.namelist():
                data = src.read(name)
                if name == LOGISTIC_CALIBRATOR_FILENAME:
                    payload = json.loads(data)
                    payload["coefficients"] = [2.0 * c for c in payload["coefficients"]]
                    data = json.dumps(payload).encode()
                dst.writestr(name, data)

        with pytest.raises(ValueError, match="sanity check"):
            load_model_package(tampered_path, extract_dir=tmp_path / "ext2")


class TestProvenance:
    def test_model_version_recorded_when_provided(
        self,
        tmp_path: Path,
        dummy_yolo_weights: Path,
        dummy_classifier_ckpt: Path,
    ) -> None:
        out = tmp_path / "m.zip"
        build_model_package(
            yolo_weights_path=dummy_yolo_weights,
            classifier_ckpt_path=dummy_classifier_ckpt,
            config=SAMPLE_CONFIG,
            variant="vit_dinov2_finetune",
            output_path=out,
            model_version="1.4.0",
            allow_uncalibrated=True,
        )
        with zipfile.ZipFile(out, "r") as zf:
            manifest = yaml.safe_load(zf.read(MANIFEST_FILENAME))
        assert manifest["model_version"] == "1.4.0"

    def test_model_version_absent_when_not_provided(self, built_archive: Path) -> None:
        with zipfile.ZipFile(built_archive, "r") as zf:
            manifest = yaml.safe_load(zf.read(MANIFEST_FILENAME))
        assert "model_version" not in manifest

    def test_provenance_detector_matches_source_of_truth(
        self, built_archive: Path
    ) -> None:
        with zipfile.ZipFile(built_archive, "r") as zf:
            manifest = yaml.safe_load(zf.read(MANIFEST_FILENAME))
        assert manifest["provenance"]["detector"] == load_detector().model_dump()

    def test_provenance_backbone_from_config(self, built_archive: Path) -> None:
        with zipfile.ZipFile(built_archive, "r") as zf:
            manifest = yaml.safe_load(zf.read(MANIFEST_FILENAME))
        assert (
            manifest["provenance"]["backbone"]
            == SAMPLE_CONFIG["classifier"]["backbone"]
        )

    def test_provenance_train_git_sha_recorded(
        self,
        tmp_path: Path,
        dummy_yolo_weights: Path,
        dummy_classifier_ckpt: Path,
    ) -> None:
        out = tmp_path / "m.zip"
        build_model_package(
            yolo_weights_path=dummy_yolo_weights,
            classifier_ckpt_path=dummy_classifier_ckpt,
            config=SAMPLE_CONFIG,
            variant="vit_dinov2_finetune",
            output_path=out,
            train_git_sha="abc1234",
            allow_uncalibrated=True,
        )
        with zipfile.ZipFile(out, "r") as zf:
            manifest = yaml.safe_load(zf.read(MANIFEST_FILENAME))
        assert manifest["provenance"]["train_git_sha"] == "abc1234"


class TestIsCalibrated:
    def test_calibrator_and_logistic_is_calibrated(self) -> None:
        assert is_calibrated(_make_calibrator(), "logistic") is True

    def test_calibrator_but_max_logit_is_uncalibrated(self) -> None:
        assert is_calibrated(_make_calibrator(), "max_logit") is False

    def test_no_calibrator_logistic_is_uncalibrated(self) -> None:
        assert is_calibrated(None, "logistic") is False

    def test_no_calibrator_max_logit_is_uncalibrated(self) -> None:
        assert is_calibrated(None, "max_logit") is False


class TestRequireCalibrated:
    def test_passes_when_calibrated(self) -> None:
        require_calibrated(_make_calibrator(), "logistic", context="x")  # no raise

    def test_raises_when_uncalibrated(self) -> None:
        with pytest.raises(UncalibratedModelError, match="not calibrated"):
            require_calibrated(None, "max_logit", context="ctx")

    def test_error_is_a_valueerror(self) -> None:
        assert issubclass(UncalibratedModelError, ValueError)


class TestBuildCalibrationGate:
    def test_build_rejects_uncalibrated_by_default(
        self, tmp_path: Path, dummy_yolo_weights: Path, dummy_classifier_ckpt: Path
    ) -> None:
        # SAMPLE_CONFIG is max_logit and no calibrator -> uncalibrated.
        with pytest.raises(UncalibratedModelError, match="not calibrated"):
            build_model_package(
                yolo_weights_path=dummy_yolo_weights,
                classifier_ckpt_path=dummy_classifier_ckpt,
                config=SAMPLE_CONFIG,
                variant="vit_dinov2_finetune",
                output_path=tmp_path / "out.zip",
            )

    def test_build_rejects_calibrator_with_max_logit(
        self, tmp_path: Path, dummy_yolo_weights: Path, dummy_classifier_ckpt: Path
    ) -> None:
        # Calibrator present but aggregation is max_logit -> still uncalibrated.
        with pytest.raises(UncalibratedModelError):
            build_model_package(
                yolo_weights_path=dummy_yolo_weights,
                classifier_ckpt_path=dummy_classifier_ckpt,
                config=SAMPLE_CONFIG,  # aggregation == "max_logit"
                variant="v",
                output_path=tmp_path / "out.zip",
                calibrator=_make_calibrator(),
            )

    def test_build_allows_uncalibrated_when_opted_in(
        self, tmp_path: Path, dummy_yolo_weights: Path, dummy_classifier_ckpt: Path
    ) -> None:
        out = build_model_package(
            yolo_weights_path=dummy_yolo_weights,
            classifier_ckpt_path=dummy_classifier_ckpt,
            config=SAMPLE_CONFIG,
            variant="v",
            output_path=tmp_path / "out.zip",
            allow_uncalibrated=True,
        )
        assert out.exists()

    def test_build_allows_calibrated_by_default(
        self, tmp_path: Path, dummy_yolo_weights: Path, dummy_classifier_ckpt: Path
    ) -> None:
        cfg = {**SAMPLE_CONFIG}
        cfg["decision"] = {**SAMPLE_CONFIG["decision"], "aggregation": "logistic"}
        out = build_model_package(
            yolo_weights_path=dummy_yolo_weights,
            classifier_ckpt_path=dummy_classifier_ckpt,
            config=cfg,
            variant="v",
            output_path=tmp_path / "out.zip",
            calibrator=_make_calibrator(),
        )
        assert out.exists()


class TestLoadCalibrationGate:
    @patch("temporal_model.core.package.load_yolo")
    def test_load_rejects_uncalibrated_by_default(
        self, mock_yolo: MagicMock, real_tiny_archive: Path, tmp_path: Path
    ) -> None:
        mock_yolo.return_value = MagicMock(name="FakeYOLO")
        # real_tiny_archive is max_logit, no calibrator -> uncalibrated.
        with pytest.raises(UncalibratedModelError, match="not calibrated"):
            load_model_package(real_tiny_archive, extract_dir=tmp_path / "ext")

    @patch("temporal_model.core.package.load_yolo")
    def test_load_allows_uncalibrated_when_opted_in(
        self, mock_yolo: MagicMock, real_tiny_archive: Path, tmp_path: Path
    ) -> None:
        mock_yolo.return_value = MagicMock(name="FakeYOLO")
        pkg = load_model_package(
            real_tiny_archive, extract_dir=tmp_path / "ext", allow_uncalibrated=True
        )
        assert pkg.calibrator is None

    @patch("temporal_model.core.package.load_yolo")
    def test_load_allows_calibrated_by_default(
        self,
        mock_yolo: MagicMock,
        tmp_path: Path,
        dummy_yolo_weights: Path,
        real_tiny_classifier_ckpt: Path,
        real_tiny_config: dict,
    ) -> None:
        mock_yolo.return_value = MagicMock(name="FakeYOLO")
        cfg = dict(real_tiny_config)
        cfg["decision"] = dict(cfg["decision"])
        cfg["decision"]["aggregation"] = "logistic"
        out = tmp_path / "cal.zip"
        build_model_package(
            yolo_weights_path=dummy_yolo_weights,
            classifier_ckpt_path=real_tiny_classifier_ckpt,
            config=cfg,
            variant="tiny",
            output_path=out,
            calibrator=_make_calibrator(),
        )
        pkg = load_model_package(out, extract_dir=tmp_path / "ext")
        assert pkg.calibrator is not None

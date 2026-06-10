import json
import zipfile
from pathlib import Path

from temporal_model.eval.model_config import read_model_config

MANIFEST = """\
detector:
  source: hf:pyronear/yolo11s_nimble-narwhal_v6.0.0
  type: yolo
train_git_sha: 4b4d43ad77c401bab6d01d561b0aa2337f7ee031
variant: vit_dinov2_finetune
yolo_weights: yolo_weights.pt
"""

CONFIG = """\
decision:
  aggregation: logistic
  logistic_threshold: 0.4736
  threshold: 0.8699
  trigger_rule: end_of_winner
infer:
  pad_strategy: symmetric
  pad_to_min_frames: 6
model_input:
  context_factor: 1.5
  stabilize: true
  patch_size: 224
classifier:
  backbone: vit_small_patch14_dinov2.lvd142m
  max_frames: 20
tubes:
  min_tube_length: 4
"""

CALIBRATOR = '{"features": ["logit"], "coefficients": [0.5], "intercept": -5.1}'


def _make_zip(tmp_path: Path, *, with_calibrator: bool = True) -> Path:
    zp = tmp_path / "model.zip"
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr("manifest.yaml", MANIFEST)
        z.writestr("config.yaml", CONFIG)
        if with_calibrator:
            z.writestr("logistic_calibrator.json", CALIBRATOR)
    return zp


def test_read_model_config_merges_members(tmp_path):
    cfg = read_model_config(_make_zip(tmp_path))
    assert cfg["detector"]["source"] == "hf:pyronear/yolo11s_nimble-narwhal_v6.0.0"
    assert cfg["variant"] == "vit_dinov2_finetune"
    assert cfg["train_git_sha"].startswith("4b4d43a")
    assert cfg["decision"]["aggregation"] == "logistic"
    assert cfg["decision"]["threshold"] == 0.8699
    assert cfg["decision"]["logistic_threshold"] == 0.4736
    assert cfg["infer"]["pad_strategy"] == "symmetric"
    assert cfg["infer"]["pad_to_min_frames"] == 6
    assert cfg["model_input"]["stabilize"] is True
    assert cfg["model_input"]["context_factor"] == 1.5
    assert cfg["classifier"]["max_frames"] == 20
    assert cfg["calibrator"]["features"] == ["logit"]


def test_read_model_config_missing_calibrator_is_none(tmp_path):
    cfg = read_model_config(_make_zip(tmp_path, with_calibrator=False))
    assert cfg["calibrator"] is None
    assert cfg["variant"] == "vit_dinov2_finetune"


def test_read_model_config_missing_zip_returns_empty(tmp_path):
    assert read_model_config(tmp_path / "does_not_exist.zip") == {}

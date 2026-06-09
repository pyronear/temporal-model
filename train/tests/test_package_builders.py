"""Tests for the pure config/kwarg builders in package.py."""

from temporal_model.train.package import (
    allow_uncalibrated_for,
    build_config,
    classifier_kwargs,
    tubes_config,
)

PARAMS = {
    "tubes": {
        "iou_threshold": 0.2,
        "max_misses": 2,
        "merge_iomin": 0.3,
        "merge_prox_factor": 1.0,
        "merge_max_gap": 10,
    },
    "build_tubes": {"min_tube_length": 4, "min_detected_entries": 2},
    "model_input": {"context_factor": 1.5, "patch_size": 224},
    "package": {
        "target_recall": 0.95,
        "infer_min_tube_length": 2,
        "infer": {"confidence_threshold": 0.1, "iou_nms": 0.2, "image_size": 1024},
        "aggregation": {"vit_dinov2_finetune": "logistic"},
    },
    "train_vit_dinov2_finetune": {
        "backbone": "vit_small_patch14_dinov2.lvd142m",
        "finetune": True,
        "finetune_last_n_blocks": 1,
        "global_pool": "token",
        "img_size": 224,
        "max_frames": 20,
        "transformer_num_layers": 2,
        "transformer_num_heads": 6,
        "transformer_ffn_dim": 1536,
        "transformer_dropout": 0.1,
    },
}


def test_classifier_kwargs_is_transformer_only() -> None:
    kw = classifier_kwargs(PARAMS["train_vit_dinov2_finetune"])
    assert kw["backbone"] == "vit_small_patch14_dinov2.lvd142m"
    assert kw["pretrained"] is False
    assert kw["global_pool"] == "token"
    assert kw["transformer_num_heads"] == 6
    # No GRU-era keys.
    assert "arch" not in kw and "hidden_dim" not in kw and "bidirectional" not in kw


def test_tubes_config_includes_merge_when_present() -> None:
    cfg = tubes_config(PARAMS)
    assert cfg["min_tube_length"] == 4
    assert cfg["infer_min_tube_length"] == 2
    assert cfg["merge_iomin"] == 0.3 and cfg["merge_max_gap"] == 10


def test_build_config_shape() -> None:
    cfg = build_config(
        PARAMS,
        PARAMS["train_vit_dinov2_finetune"],
        threshold=0.4,
        aggregation="logistic",
        logistic_threshold=0.52,
    )
    assert set(cfg) == {"infer", "tubes", "model_input", "classifier", "decision"}
    assert cfg["decision"]["aggregation"] == "logistic"
    assert cfg["decision"]["threshold"] == 0.4
    assert cfg["decision"]["logistic_threshold"] == 0.52
    assert cfg["decision"]["target_recall"] == 0.95
    assert cfg["model_input"]["normalization"]["mean"] == [0.485, 0.456, 0.406]


def test_build_config_bakes_stabilize_from_param() -> None:
    params = {**PARAMS, "model_input": {**PARAMS["model_input"], "stabilize": True}}
    cfg = build_config(
        params,
        params["train_vit_dinov2_finetune"],
        threshold=0.4,
        aggregation="logistic",
        logistic_threshold=0.52,
    )
    assert cfg["model_input"]["stabilize"] is True


def test_build_config_stabilize_defaults_true_when_absent() -> None:
    # PARAMS["model_input"] has no "stabilize" key.
    cfg = build_config(
        PARAMS,
        PARAMS["train_vit_dinov2_finetune"],
        threshold=0.4,
        aggregation="logistic",
        logistic_threshold=0.52,
    )
    assert cfg["model_input"]["stabilize"] is True


def test_allow_uncalibrated_only_for_non_logistic() -> None:
    # The real opt-out wired into build_model_package in package.py:
    # logistic enforces calibration (no opt-out); everything else opts out.
    assert allow_uncalibrated_for("logistic") is False
    assert allow_uncalibrated_for("max_logit") is True
    assert allow_uncalibrated_for("anything_else") is True

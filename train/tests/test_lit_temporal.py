"""Tests for the temporal classifier Lightning module."""

from unittest.mock import MagicMock

import torch

from temporal_model.train.lit_temporal import LitTemporalClassifier


def _batch(b: int = 2, t: int = 5) -> dict:
    return {
        "patches": torch.randn(b, t, 3, 224, 224),
        "mask": torch.ones(b, t, dtype=torch.bool),
        "label": torch.tensor([1.0, 0.0][:b]),
        "sequence_id": [f"seq_{i}" for i in range(b)],
    }


def _vit_lit(**overrides) -> LitTemporalClassifier:
    """A small ViT-backbone transformer classifier for fast tests.

    Uses vit_small_patch16_224 (feat_dim 384); the production backbone is
    vit_small_patch14_dinov2.lvd142m.
    """
    kwargs = dict(
        backbone="vit_small_patch16_224",
        learning_rate=1e-3,
        weight_decay=1e-2,
        pretrained=False,
        global_pool="token",
        transformer_num_layers=1,
        transformer_num_heads=6,
        transformer_ffn_dim=64,
        transformer_dropout=0.0,
        max_frames=20,
    )
    kwargs.update(overrides)
    return LitTemporalClassifier(**kwargs)


def test_lit_module_training_step_returns_loss_scalar():
    lit = _vit_lit()
    loss = lit.training_step(_batch(), batch_idx=0)
    assert loss.ndim == 0
    assert loss.item() >= 0


def test_lit_module_validation_step_runs_without_error():
    lit = _vit_lit()
    lit.validation_step(_batch(), batch_idx=0)


def test_lit_module_optimizer_only_includes_head_params():
    lit = _vit_lit()
    opt = lit.configure_optimizers()
    head_param_ids = {id(p) for p in lit.model.head.parameters()}
    optim_param_ids = {id(p) for g in opt.param_groups for p in g["params"]}
    assert optim_param_ids == head_param_ids


def test_lit_module_optimizer_uses_two_groups_when_finetune():
    lit = _vit_lit(finetune=True, finetune_last_n_blocks=1, backbone_lr=1e-5)
    opt = lit.configure_optimizers()
    assert len(opt.param_groups) == 2
    lrs = sorted(g["lr"] for g in opt.param_groups)
    assert lrs == [1e-5, 1e-3]

    # Every trainable param must land in exactly one group.
    trainable = {id(p) for p in lit.model.parameters() if p.requires_grad}
    grouped = [id(p) for g in opt.param_groups for p in g["params"]]
    assert set(grouped) == trainable
    assert len(grouped) == len(trainable), "duplicate params across groups"


def test_lit_module_optimizer_stays_single_group_when_frozen():
    lit = _vit_lit()
    opt = lit.configure_optimizers()
    assert len(opt.param_groups) == 1
    assert opt.param_groups[0]["lr"] == 1e-3


def test_lit_temporal_transformer_forward_shape():
    lit = LitTemporalClassifier(
        backbone="vit_small_patch16_224",
        learning_rate=1e-4,
        weight_decay=0.05,
        pretrained=False,
        transformer_num_layers=2,
        transformer_num_heads=6,
        transformer_ffn_dim=1536,
        transformer_dropout=0.0,
        max_frames=20,
        global_pool="token",
    )
    patches = torch.randn(2, 4, 3, 224, 224)
    mask = torch.ones(2, 4, dtype=torch.bool)
    out = lit(patches, mask)
    assert out.shape == (2,)


def test_lit_temporal_cosine_warmup_returns_scheduler_dict(tmp_path):
    lit = _vit_lit(use_cosine_warmup=True, warmup_frac=0.1)
    # configure_optimizers reads self.trainer; attach a minimal stub.
    stub = MagicMock()
    stub.estimated_stepping_batches = 100
    lit.trainer = stub
    out = lit.configure_optimizers()
    assert isinstance(out, dict)
    assert "optimizer" in out
    assert "lr_scheduler" in out
    assert out["lr_scheduler"]["interval"] == "step"

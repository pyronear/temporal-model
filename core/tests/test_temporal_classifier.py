"""Tests for TemporalSmokeClassifier and its components."""

import pytest
import torch

from temporal_model.core.temporal_classifier import (
    TemporalSmokeClassifier,
    TimmBackbone,
    TransformerHead,
)

# A small ViT that takes 224x224 input; the production backbone is
# vit_small_patch14_dinov2.lvd142m, but this in1k variant is faster for tests
# and exercises the same ViT unfreeze path.
TEST_BACKBONE = "vit_small_patch16_224"
TEST_FEAT_DIM = 384


def test_timm_backbone_outputs_features_per_frame():
    bb = TimmBackbone(name=TEST_BACKBONE, pretrained=False, global_pool="token")
    x = torch.randn(2, 3, 224, 224)
    out = bb(x)
    assert out.shape == (2, bb.feat_dim)
    assert bb.feat_dim == TEST_FEAT_DIM


def test_timm_backbone_has_no_trainable_params():
    bb = TimmBackbone(name=TEST_BACKBONE, pretrained=False, global_pool="token")
    trainable = [p for p in bb.parameters() if p.requires_grad]
    assert trainable == []


def test_timm_backbone_stays_in_eval_mode_after_train_call():
    bb = TimmBackbone(name=TEST_BACKBONE, pretrained=False, global_pool="token")
    bb.train()
    assert not bb.backbone.training


def test_timm_backbone_frozen_forward_matches_no_grad_eval():
    torch.manual_seed(0)
    bb = TimmBackbone(
        name=TEST_BACKBONE, pretrained=False, finetune=False, global_pool="token"
    )
    x = torch.randn(2, 3, 224, 224)
    # Put the module into train mode; frozen path must still force eval().
    bb.train()
    assert bb.backbone.training is False
    out_a = bb(x)
    # Independent reference: call the underlying timm model directly in eval/no_grad.
    bb.backbone.eval()
    with torch.no_grad():
        out_b = bb.backbone(x)
    assert torch.allclose(out_a, out_b, atol=1e-6)


def test_timm_backbone_finetune_unsupported_family_raises():
    # Only ViT backbones have an unfreeze rule; any other family should raise
    # the same informative error.
    with pytest.raises(NotImplementedError) as exc:
        TimmBackbone(
            name="resnet18",
            pretrained=False,
            finetune=True,
            finetune_last_n_blocks=1,
        )
    msg = str(exc.value)
    assert "resnet18" in msg
    assert "children" in msg.lower()


def test_timm_backbone_vit_token_pool_returns_cls_embedding():
    bb = TimmBackbone(
        name="vit_small_patch16_224",
        pretrained=False,
        global_pool="token",
    )
    x = torch.randn(2, 3, 224, 224)
    out = bb(x)
    assert out.shape == (2, bb.feat_dim)
    assert bb.feat_dim == 384


def test_timm_backbone_finetune_vit_s16_unfreezes_only_last_block():
    bb = TimmBackbone(
        name="vit_small_patch16_224",
        pretrained=False,
        finetune=True,
        finetune_last_n_blocks=1,
        global_pool="token",
    )
    trainable_names = [n for n, p in bb.named_parameters() if p.requires_grad]
    assert trainable_names, "expected some trainable params"
    # timm's ViT wraps blocks under `blocks.<i>.*`; last block is index 11 for ViT-S.
    assert all(".blocks.11." in n for n in trainable_names), trainable_names


def test_timm_backbone_finetune_vit_s16_n2_unfreezes_last_two_blocks():
    bb = TimmBackbone(
        name="vit_small_patch16_224",
        pretrained=False,
        finetune=True,
        finetune_last_n_blocks=2,
        global_pool="token",
    )
    trainable_names = [n for n, p in bb.named_parameters() if p.requires_grad]
    assert trainable_names
    assert all((".blocks.10." in n) or (".blocks.11." in n) for n in trainable_names), (
        trainable_names
    )


def test_timm_backbone_finetune_train_mode_propagates_vit_s16():
    bb = TimmBackbone(
        name="vit_small_patch16_224",
        pretrained=False,
        finetune=True,
        finetune_last_n_blocks=1,
        global_pool="token",
    )
    bb.train()
    assert bb.backbone.blocks[11].training is True
    assert bb.backbone.blocks[0].training is False


def test_timm_backbone_vit_s14_dinov2_finetune_unfreezes_last_block():
    bb = TimmBackbone(
        name="vit_small_patch14_dinov2.lvd142m",
        pretrained=False,
        finetune=True,
        finetune_last_n_blocks=1,
        global_pool="token",
    )
    trainable_names = [n for n, p in bb.named_parameters() if p.requires_grad]
    assert trainable_names
    assert all(".blocks.11." in n for n in trainable_names), trainable_names


def test_timm_backbone_vit_s14_dinov2_img_size_224_forward():
    # DINOv2 ViT-S/14 was pretrained at 518x518 and would reject 224 input
    # without the img_size override (triggers pos_embed interpolation).
    bb = TimmBackbone(
        name="vit_small_patch14_dinov2.lvd142m",
        pretrained=False,
        global_pool="token",
        img_size=224,
    )
    x = torch.randn(2, 3, 224, 224)
    out = bb(x)
    assert out.shape == (2, 384)


def test_transformer_head_returns_logits_per_batch():
    head = TransformerHead(
        feat_dim=384,
        num_layers=2,
        num_heads=6,
        ffn_dim=1536,
        dropout=0.0,
        max_frames=20,
    )
    feats = torch.randn(3, 20, 384)
    mask = torch.ones(3, 20, dtype=torch.bool)
    logits = head(feats, mask)
    assert logits.shape == (3,)


def test_transformer_head_respects_mask():
    torch.manual_seed(0)
    head = TransformerHead(
        feat_dim=16,
        num_layers=1,
        num_heads=2,
        ffn_dim=32,
        dropout=0.0,
        max_frames=20,
    )
    head.eval()
    real = torch.randn(2, 16)
    a = torch.zeros(20, 16)
    a[:2] = real
    b = a.clone()
    b[2:] = 1e3  # junk in padded positions
    feats = torch.stack([a, b])
    mask = torch.zeros(2, 20, dtype=torch.bool)
    mask[:, :2] = True
    logits = head(feats, mask)
    assert torch.allclose(logits[0], logits[1], atol=1e-4), logits


def test_classifier_transformer_forward_shape_vit_backbone():
    clf = TemporalSmokeClassifier(
        backbone="vit_small_patch16_224",
        pretrained=False,
        transformer_num_layers=2,
        transformer_num_heads=6,
        transformer_ffn_dim=1536,
        transformer_dropout=0.0,
        max_frames=20,
        global_pool="token",
    )
    patches = torch.randn(2, 5, 3, 224, 224)
    mask = torch.ones(2, 5, dtype=torch.bool)
    logits = clf(patches, mask)
    assert logits.shape == (2,)


def test_classifier_transformer_handles_padded_batches():
    clf = TemporalSmokeClassifier(
        backbone="vit_small_patch16_224",
        pretrained=False,
        transformer_num_layers=1,
        transformer_num_heads=6,
        transformer_ffn_dim=384,
        transformer_dropout=0.0,
        max_frames=20,
        global_pool="token",
    )
    patches = torch.randn(3, 20, 3, 224, 224)
    mask = torch.zeros(3, 20, dtype=torch.bool)
    mask[0, :20] = True
    mask[1, :10] = True
    mask[2, :3] = True
    logits = clf(patches, mask)
    assert logits.shape == (3,)


def test_classifier_transformer_frozen_only_head_trainable():
    clf = TemporalSmokeClassifier(
        backbone="vit_small_patch16_224",
        pretrained=False,
        transformer_num_layers=2,
        transformer_num_heads=6,
        transformer_ffn_dim=1536,
        transformer_dropout=0.0,
        max_frames=20,
        global_pool="token",
    )
    trainable = [n for n, p in clf.named_parameters() if p.requires_grad]
    assert all(n.startswith("head.") for n in trainable), trainable


def test_classifier_transformer_finetune_exposes_last_vit_block():
    clf = TemporalSmokeClassifier(
        backbone="vit_small_patch16_224",
        pretrained=False,
        transformer_num_layers=2,
        transformer_num_heads=6,
        transformer_ffn_dim=1536,
        transformer_dropout=0.0,
        max_frames=20,
        global_pool="token",
        finetune=True,
        finetune_last_n_blocks=1,
    )
    trainable = [n for n, p in clf.named_parameters() if p.requires_grad]
    assert any(".blocks.11." in n for n in trainable), trainable
    assert any(n.startswith("head.") for n in trainable)
    assert not any(".blocks.0." in n for n in trainable)

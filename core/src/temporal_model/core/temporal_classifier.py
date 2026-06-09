"""Temporal smoke classifier: timm backbone + transformer temporal head."""

import timm
import torch
from torch import Tensor, nn

__all__ = ["TimmBackbone", "TransformerHead", "TemporalSmokeClassifier"]


class TimmBackbone(nn.Module):
    """Wraps a pretrained timm model as a per-frame feature extractor.

    When ``finetune=False`` (default), all params are frozen and the
    inner model is forced into ``eval()`` mode regardless of the parent
    module's training flag; forward is wrapped in ``torch.no_grad()``.

    When ``finetune=True``, the last ``finetune_last_n_blocks`` blocks
    are unfrozen (family-specific resolution); everything else stays
    frozen, and ``.train()`` propagates normally so BatchNorm on
    unfrozen blocks updates.
    """

    def __init__(
        self,
        name: str,
        pretrained: bool = True,
        finetune: bool = False,
        finetune_last_n_blocks: int = 0,
        global_pool: str = "avg",
        img_size: int | None = None,
    ) -> None:
        super().__init__()
        # DINOv2 ViT-S/14 was pretrained at 518x518 and hardcodes self.img_size;
        # passing img_size lets timm interpolate pos_embed to our 224 input.
        extra = {"img_size": img_size} if img_size is not None else {}
        self.backbone = timm.create_model(
            name,
            pretrained=pretrained,
            num_classes=0,
            global_pool=global_pool,
            **extra,
        )
        self.feat_dim: int = self.backbone.num_features
        self.finetune = finetune
        self.name = name

        if not finetune:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
            return

        # Finetune path: freeze everything first, then unfreeze last N blocks.
        for p in self.backbone.parameters():
            p.requires_grad = False
        self._unfreeze_last_n_blocks(finetune_last_n_blocks)

    def _unfreeze_last_n_blocks(self, n: int) -> None:
        if n <= 0:
            return
        name = self.name
        if name.startswith("vit_"):
            # timm's ViT exposes a `blocks` ModuleList of transformer blocks.
            stages = list(self.backbone.blocks)
        else:
            stage_names = [n_ for n_, _ in self.backbone.named_children()]
            raise NotImplementedError(
                f"finetune=True is not implemented for backbone family "
                f"{name!r}. Top-level children: {stage_names}. Add an "
                f"explicit unfreeze rule in TimmBackbone._unfreeze_last_n_blocks."
            )
        for stage in stages[-n:]:
            for p in stage.parameters():
                p.requires_grad = True

    def train(self, mode: bool = True) -> "TimmBackbone":
        super().train(mode)
        if not self.finetune:
            self.backbone.eval()
        else:
            # Keep frozen sub-modules in eval mode so their BatchNorm
            # running stats stay deterministic.
            for module in self.backbone.modules():
                if not any(p.requires_grad for p in module.parameters(recurse=False)):
                    has_trainable_descendant = any(
                        p.requires_grad for p in module.parameters()
                    )
                    if not has_trainable_descendant:
                        module.eval()
        return self

    def forward(self, x: Tensor) -> Tensor:
        if not self.finetune:
            with torch.no_grad():
                return self.backbone(x)
        return self.backbone(x)


class TransformerHead(nn.Module):
    """Learnable [CLS] + learned positional embeddings + Transformer encoder.

    Returns one logit per tube. Padded positions are masked out of
    attention via ``src_key_padding_mask``.
    """

    def __init__(
        self,
        feat_dim: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        max_frames: int,
    ) -> None:
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, feat_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        # +1 for the prepended CLS position.
        self.pos_embed = nn.Parameter(torch.zeros(1, max_frames + 1, feat_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feat_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(feat_dim, 1)
        self.max_frames = max_frames

    def forward(self, feats: Tensor, mask: Tensor) -> Tensor:
        b, t, _ = feats.shape
        if t > self.max_frames:
            raise ValueError(
                f"TransformerHead received T={t} frames but was configured "
                f"with max_frames={self.max_frames}"
            )
        cls = self.cls_token.expand(b, 1, -1)
        x = torch.cat([cls, feats], dim=1)  # (B, T+1, D)
        x = x + self.pos_embed[:, : t + 1, :]
        # True = pad (ignored) per torch convention; our input mask is True = real.
        cls_real = torch.ones(b, 1, dtype=torch.bool, device=mask.device)
        real_mask = torch.cat([cls_real, mask], dim=1)
        key_padding_mask = ~real_mask  # (B, T+1), True = pad
        out = self.encoder(x, src_key_padding_mask=key_padding_mask)
        cls_out = out[:, 0, :]
        return self.classifier(cls_out).squeeze(-1)


class TemporalSmokeClassifier(nn.Module):
    """Timm backbone applied per-frame plus a transformer temporal head.

    Produces a single binary logit per tube.
    """

    def __init__(
        self,
        backbone: str,
        pretrained: bool = True,
        finetune: bool = False,
        finetune_last_n_blocks: int = 0,
        transformer_num_layers: int = 2,
        transformer_num_heads: int = 6,
        transformer_ffn_dim: int = 1536,
        transformer_dropout: float = 0.1,
        max_frames: int = 20,
        global_pool: str = "avg",
        img_size: int | None = None,
    ) -> None:
        super().__init__()
        self.backbone = TimmBackbone(
            name=backbone,
            pretrained=pretrained,
            finetune=finetune,
            finetune_last_n_blocks=finetune_last_n_blocks,
            global_pool=global_pool,
            img_size=img_size,
        )
        feat_dim = self.backbone.feat_dim
        self.head = TransformerHead(
            feat_dim=feat_dim,
            num_layers=transformer_num_layers,
            num_heads=transformer_num_heads,
            ffn_dim=transformer_ffn_dim,
            dropout=transformer_dropout,
            max_frames=max_frames,
        )

    def forward(self, patches: Tensor, mask: Tensor) -> Tensor:
        b, t, c, h, w = patches.shape
        flat = patches.reshape(b * t, c, h, w)
        feats = self.backbone(flat).reshape(b, t, -1)
        return self.head(feats, mask)

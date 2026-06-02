"""PyTorch Lightning wrapper around TemporalSmokeClassifier."""

import math

import lightning as L
import torch
from torch import Tensor

from temporal_model.core.temporal_classifier import TemporalSmokeClassifier


class LitTemporalClassifier(L.LightningModule):
    """Lightning module: BCE loss, AdamW on head params only.

    Args:
        backbone: timm model name.
        learning_rate: AdamW lr.
        weight_decay: AdamW weight decay.
        pretrained: whether to load pretrained backbone weights.
        finetune: if True, unfreeze backbone blocks and use two param groups.
        finetune_last_n_blocks: number of backbone blocks to unfreeze.
        backbone_lr: learning rate for backbone params (required when finetune=True).
    """

    def __init__(
        self,
        backbone: str,
        learning_rate: float,
        weight_decay: float,
        pretrained: bool = True,
        finetune: bool = False,
        finetune_last_n_blocks: int = 0,
        backbone_lr: float | None = None,
        transformer_num_layers: int = 2,
        transformer_num_heads: int = 6,
        transformer_ffn_dim: int = 1536,
        transformer_dropout: float = 0.1,
        max_frames: int = 20,
        global_pool: str = "avg",
        img_size: int | None = None,
        use_cosine_warmup: bool = False,
        warmup_frac: float = 0.05,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = TemporalSmokeClassifier(
            backbone=backbone,
            pretrained=pretrained,
            finetune=finetune,
            finetune_last_n_blocks=finetune_last_n_blocks,
            transformer_num_layers=transformer_num_layers,
            transformer_num_heads=transformer_num_heads,
            transformer_ffn_dim=transformer_ffn_dim,
            transformer_dropout=transformer_dropout,
            max_frames=max_frames,
            global_pool=global_pool,
            img_size=img_size,
        )
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.finetune = finetune
        self.backbone_lr = backbone_lr
        self.use_cosine_warmup = use_cosine_warmup
        self.warmup_frac = warmup_frac
        self._val_preds: list[float] = []
        self._val_labels: list[float] = []

    def forward(self, patches: Tensor, mask: Tensor) -> Tensor:
        return self.model(patches, mask)

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        logits = self(batch["patches"], batch["mask"])
        loss = self.loss_fn(logits, batch["label"])
        self.log("train/loss", loss, prog_bar=True, batch_size=logits.shape[0])
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        logits = self(batch["patches"], batch["mask"])
        loss = self.loss_fn(logits, batch["label"])
        probs = torch.sigmoid(logits).detach().cpu()
        labels = batch["label"].detach().cpu()
        self._val_preds.extend(probs.tolist())
        self._val_labels.extend(labels.tolist())
        self.log("val/loss", loss, prog_bar=True, batch_size=logits.shape[0])

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return
        probs = torch.tensor(self._val_preds)
        labels = torch.tensor(self._val_labels)
        preds = (probs > 0.5).float()
        tp = ((preds == 1) & (labels == 1)).sum().float()
        fp = ((preds == 1) & (labels == 0)).sum().float()
        fn = ((preds == 0) & (labels == 1)).sum().float()
        tn = ((preds == 0) & (labels == 0)).sum().float()
        acc = (tp + tn) / (tp + tn + fp + fn).clamp(min=1)
        prec = tp / (tp + fp).clamp(min=1)
        rec = tp / (tp + fn).clamp(min=1)
        f1 = 2 * prec * rec / (prec + rec).clamp(min=1e-8)
        self.log("val/accuracy", acc, prog_bar=True)
        self.log("val/precision", prec)
        self.log("val/recall", rec)
        self.log("val/f1", f1, prog_bar=True)
        self._val_preds.clear()
        self._val_labels.clear()

    def configure_optimizers(self):
        if not self.finetune:
            head_params = [p for p in self.model.head.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(
                head_params,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        else:
            if self.backbone_lr is None:
                raise ValueError("backbone_lr must be set when finetune=True")
            backbone_params = [
                p for p in self.model.backbone.parameters() if p.requires_grad
            ]
            head_params = [p for p in self.model.head.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(
                [
                    {
                        "params": head_params,
                        "lr": self.learning_rate,
                        "weight_decay": self.weight_decay,
                    },
                    {
                        "params": backbone_params,
                        "lr": self.backbone_lr,
                        "weight_decay": self.weight_decay,
                    },
                ]
            )

        if not self.use_cosine_warmup:
            return optimizer

        total_steps = int(self.trainer.estimated_stepping_batches)
        warmup_steps = max(1, int(total_steps * self.warmup_frac))

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

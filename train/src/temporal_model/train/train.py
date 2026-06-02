"""Train the temporal smoke classifier (timm backbone + transformer head).

Reads a single named section from ``params.yaml``
(e.g. ``train_vit_dinov2_finetune``) so each DVC stage owns its own params.
"""

import argparse
import sys
from pathlib import Path

import lightning as L
import torch
import yaml
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from torch.utils.data import DataLoader

from temporal_model.train.augment import build_tube_augment
from temporal_model.train.batch_samples import SampleTrainBatchesCallback
from temporal_model.train.dataset import TubePatchDataset
from temporal_model.train.lit_temporal import LitTemporalClassifier
from temporal_model.train.training_plots import (
    find_latest_metrics_csv,
    plot_training_curves,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--val-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--params-path", type=Path, required=True)
    parser.add_argument("--params-key", required=True, help="Key in params.yaml")
    parser.add_argument(
        "--sample-batches",
        type=int,
        default=3,
        help="Dump this many training batches as PNG grids at epoch 0 (0 disables)",
    )
    args = parser.parse_args()

    full_params = yaml.safe_load(args.params_path.read_text())
    cfg = full_params[args.params_key]
    augment_cfg = full_params.get("augment", {"enabled": False})

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(
        f"CUDA available: {torch.cuda.is_available()} | "
        f"device count: {torch.cuda.device_count()} | "
        f"device: {device_name}",
        file=sys.stderr,
        flush=True,
    )

    L.seed_everything(cfg["seed"], workers=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_transform = build_tube_augment(augment_cfg, train=True)
    val_transform = build_tube_augment(augment_cfg, train=False)

    train_ds = TubePatchDataset(
        args.train_dir,
        max_frames=cfg["max_frames"],
        transform=train_transform,
    )
    val_ds = TubePatchDataset(
        args.val_dir,
        max_frames=cfg["max_frames"],
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        persistent_workers=cfg["num_workers"] > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        persistent_workers=cfg["num_workers"] > 0,
    )

    lit = LitTemporalClassifier(
        backbone=cfg["backbone"],
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        pretrained=True,
        finetune=cfg.get("finetune", False),
        finetune_last_n_blocks=cfg.get("finetune_last_n_blocks", 0),
        backbone_lr=cfg.get("backbone_lr"),
        transformer_num_layers=cfg.get("transformer_num_layers", 2),
        transformer_num_heads=cfg.get("transformer_num_heads", 6),
        transformer_ffn_dim=cfg.get("transformer_ffn_dim", 1536),
        transformer_dropout=cfg.get("transformer_dropout", 0.1),
        max_frames=cfg.get("max_frames", 20),
        global_pool=cfg.get("global_pool", "avg"),
        img_size=cfg.get("img_size"),
        use_cosine_warmup=cfg.get("use_cosine_warmup", False),
        warmup_frac=cfg.get("warmup_frac", 0.05),
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=args.output_dir,
            filename="best",
            monitor="val/f1",
            mode="max",
            save_top_k=1,
            save_weights_only=False,
        ),
        EarlyStopping(
            monitor="val/f1", mode="max", patience=cfg["early_stop_patience"]
        ),
    ]
    if args.sample_batches > 0:
        callbacks.append(
            SampleTrainBatchesCallback(
                output_dir=args.output_dir / "batch_samples",
                n_batches=args.sample_batches,
            )
        )
    loggers = [
        CSVLogger(save_dir=args.output_dir, name="csv_logs"),
        TensorBoardLogger(save_dir=args.output_dir, name="tb_logs"),
    ]

    trainer = L.Trainer(
        max_epochs=cfg["max_epochs"],
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=10,
        deterministic=True,
        accelerator="auto",
        devices=1,
    )
    print(
        f"Trainer accelerator flag: {trainer._accelerator_connector._accelerator_flag}",
        file=sys.stderr,
        flush=True,
    )
    trainer.fit(lit, train_loader, val_loader)

    best = args.output_dir / "best.ckpt"
    target = args.output_dir / "best_checkpoint.pt"
    if best.exists():
        if target.exists():
            target.unlink()
        best.rename(target)

    try:
        csv_path = find_latest_metrics_csv(args.output_dir / "csv_logs")
        plot_path = args.output_dir / "plots" / "training_curves.png"
        plot_training_curves(csv_path, plot_path, title=args.params_key)
        print(f"Wrote {plot_path}", file=sys.stderr, flush=True)
    except Exception as exc:
        print(f"Training-curve plot failed: {exc}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()

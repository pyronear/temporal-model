"""Per-run training curve plotting from Lightning CSVLogger metrics."""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator


def find_latest_metrics_csv(csv_log_dir: Path) -> Path:
    version_dirs = []
    for child in csv_log_dir.iterdir():
        if not child.is_dir():
            continue
        match = re.fullmatch(r"version_(\d+)", child.name)
        if match:
            version_dirs.append((int(match.group(1)), child))
    if not version_dirs:
        raise FileNotFoundError(f"No version_* directories under {csv_log_dir}")
    version_dirs.sort(key=lambda pair: pair[0])
    latest_dir = version_dirs[-1][1]
    csv_path = latest_dir / "metrics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No metrics.csv at {csv_path}")
    return csv_path


def aggregate_train_loss_per_epoch(df: pd.DataFrame) -> pd.Series:
    train_rows = df.dropna(subset=["train/loss"])
    return train_rows.groupby("epoch")["train/loss"].mean()


_VAL_COLUMNS = [
    "epoch",
    "val/loss",
    "val/accuracy",
    "val/f1",
    "val/precision",
    "val/recall",
]


def extract_val_metrics_per_epoch(df: pd.DataFrame) -> pd.DataFrame:
    val_rows = df.dropna(subset=["val/f1"])
    return val_rows[_VAL_COLUMNS].reset_index(drop=True)


_SUBPLOT_GRID = [
    ("train/loss", "train"),
    ("val/loss", "val"),
    ("val/accuracy", "val"),
    ("val/precision", "val"),
    ("val/recall", "val"),
    ("val/f1", "val"),
]


def plot_training_curves(csv_path: Path, output_path: Path, title: str) -> None:
    df = pd.read_csv(csv_path)
    train_series = aggregate_train_loss_per_epoch(df)
    val_df = extract_val_metrics_per_epoch(df)
    best_epoch = (
        val_df.loc[val_df["val/f1"].idxmax(), "epoch"] if not val_df.empty else None
    )

    with plt.style.context("seaborn-v0_8-whitegrid"):
        fig, axes = plt.subplots(
            nrows=2, ncols=3, figsize=(12, 7), constrained_layout=True
        )
        fig.suptitle(title, fontsize=14, fontweight="bold")

        flat_axes = axes.flatten()
        for ax, (metric, source) in zip(flat_axes, _SUBPLOT_GRID, strict=True):
            if source == "train":
                xs = list(train_series.index)
                ys = list(train_series.values)
            else:
                xs = list(val_df["epoch"])
                ys = list(val_df[metric])

            if xs:
                ax.plot(xs, ys, marker=".", linewidth=1.8)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "no data yet",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    color="gray",
                )

            if best_epoch is not None:
                ax.axvline(best_epoch, color="gray", linestyle="--", alpha=0.6)

            ax.set_title(metric)
            ax.grid(alpha=0.3)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        for ax in axes[1]:
            ax.set_xlabel("epoch")
        for ax in axes[0]:
            ax.set_xlabel("")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

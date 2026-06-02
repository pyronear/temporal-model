import csv

import pandas as pd

from temporal_model.train.training_plots import (
    aggregate_train_loss_per_epoch,
    extract_val_metrics_per_epoch,
    plot_training_curves,
)


def test_aggregate_train_loss_per_epoch_means_per_epoch():
    df = pd.DataFrame(
        {
            "epoch": [0, 0, 0, 1, 1, 2],
            "step": [0, 1, 2, 3, 4, 5],
            "train/loss": [1.0, 2.0, 3.0, 4.0, 6.0, 8.0],
            "val/f1": [None, None, None, None, None, None],
        }
    )

    result = aggregate_train_loss_per_epoch(df)

    assert list(result.index) == [0, 1, 2]
    assert list(result.values) == [2.0, 5.0, 8.0]


def test_aggregate_train_loss_per_epoch_ignores_val_only_rows():
    df = pd.DataFrame(
        {
            "epoch": [0, 0, 1],
            "step": [0, 1, 2],
            "train/loss": [1.0, None, 3.0],
            "val/f1": [None, 0.9, None],
        }
    )

    result = aggregate_train_loss_per_epoch(df)

    assert list(result.index) == [0, 1]
    assert list(result.values) == [1.0, 3.0]


def test_aggregate_train_loss_per_epoch_returns_empty_when_no_train_rows():
    df = pd.DataFrame(
        {
            "epoch": [0, 1],
            "step": [0, 1],
            "train/loss": [None, None],
            "val/f1": [0.9, 0.95],
        }
    )

    result = aggregate_train_loss_per_epoch(df)

    assert len(result) == 0


def test_extract_val_metrics_per_epoch_one_row_per_epoch():
    df = pd.DataFrame(
        {
            "epoch": [0, 0, 1, 1, 2],
            "step": [0, 1, 2, 3, 4],
            "train/loss": [1.0, 2.0, 3.0, 4.0, None],
            "val/loss": [None, 0.5, None, 0.4, 0.3],
            "val/accuracy": [None, 0.8, None, 0.85, 0.9],
            "val/f1": [None, 0.7, None, 0.75, 0.8],
            "val/precision": [None, 0.6, None, 0.65, 0.7],
            "val/recall": [None, 0.9, None, 0.92, 0.95],
        }
    )

    result = extract_val_metrics_per_epoch(df)

    assert list(result.columns) == [
        "epoch",
        "val/loss",
        "val/accuracy",
        "val/f1",
        "val/precision",
        "val/recall",
    ]
    assert list(result["epoch"]) == [0, 1, 2]
    assert list(result["val/f1"]) == [0.7, 0.75, 0.8]


def test_extract_val_metrics_per_epoch_skips_rows_missing_f1():
    df = pd.DataFrame(
        {
            "epoch": [0, 1],
            "step": [0, 1],
            "train/loss": [1.0, 2.0],
            "val/loss": [None, 0.4],
            "val/accuracy": [None, 0.85],
            "val/f1": [None, 0.75],
            "val/precision": [None, 0.65],
            "val/recall": [None, 0.92],
        }
    )

    result = extract_val_metrics_per_epoch(df)

    assert list(result["epoch"]) == [1]


def _write_fixture_csv(path):
    rows = [
        [
            "epoch",
            "step",
            "train/loss",
            "val/accuracy",
            "val/f1",
            "val/loss",
            "val/precision",
            "val/recall",
        ],
        [0, 0, 0.9, None, None, None, None, None],
        [0, 1, 0.8, None, None, None, None, None],
        [0, 2, None, 0.7, 0.65, 0.5, 0.6, 0.7],
        [1, 3, 0.6, None, None, None, None, None],
        [1, 4, 0.5, None, None, None, None, None],
        [1, 5, None, 0.85, 0.82, 0.3, 0.8, 0.85],
        [2, 6, 0.4, None, None, None, None, None],
        [2, 7, None, 0.9, 0.88, 0.25, 0.85, 0.9],
    ]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(["" if v is None else v for v in row])


def test_plot_training_curves_writes_png(tmp_path):
    csv_path = tmp_path / "metrics.csv"
    _write_fixture_csv(csv_path)
    out_path = tmp_path / "out" / "training_curves.png"

    plot_training_curves(csv_path, out_path, title="test_variant")

    assert out_path.exists()
    assert out_path.stat().st_size > 0

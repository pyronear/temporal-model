"""Evaluate a packaged bbox-tube-temporal model.zip.

Uses the pyrocore TemporalModel protocol. Loads the archive with
BboxTubeTemporalModel.from_archive, iterates
sequences in the given split directory, calls model.load_sequence +
model.predict per sequence, and writes leaderboard-schema metrics
plus PR/ROC curves and per-sequence predictions to --output-dir.

Strict error policy: any per-sequence exception aborts the run.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from temporal_model.core.model import BboxTubeTemporalModel
from temporal_model.core.sequences import (
    get_sorted_frames,
    is_wf_sequence,
    list_sequences,
)
from temporal_model.eval.eval_plots import (
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
)
from temporal_model.eval.protocol_eval import (
    SequenceRecord,
    build_record,
    compute_metrics,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-zip", type=Path, required=True)
    parser.add_argument("--sequences-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-name",
        required=True,
        help="Label embedded in metrics.json (e.g. 'vit_dinov2_finetune-val').",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override device selection (cuda/mps/cpu). Defaults to auto.",
    )
    return parser.parse_args()


def _record_to_json(rec: SequenceRecord) -> dict:
    """Serialise a record for predictions.json.

    Flattens the nested ``details`` into the legacy predictions.json shape
    consumed by ``scripts/analyze_variant.py`` and the logistic-calibrator
    fitter: top-level ``kept_tubes``, ``trigger_tube_id``, ``tube_logits``,
    ``num_tubes_total``, ``num_tubes_kept``, and ``threshold``.
    """
    details = rec.details
    tubes = details.get("tubes", {})
    decision = details.get("decision", {})
    kept = tubes.get("kept", [])
    return {
        "sequence_id": rec.sequence_id,
        "label": rec.label,
        "is_positive": rec.is_positive,
        "trigger_frame_index": rec.trigger_frame_index,
        "score": rec.score if rec.score != float("-inf") else None,
        "num_tubes_kept": len(kept),
        "num_tubes_total": int(tubes.get("num_candidates", 0)),
        "tube_logits": [float(t["logit"]) for t in kept],
        "trigger_tube_id": decision.get("trigger_tube_id"),
        "threshold": (
            float(decision["threshold"]) if "threshold" in decision else None
        ),
        "kept_tubes": kept,
        "ttd_frames": rec.ttd_frames,
    }


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = BboxTubeTemporalModel.from_archive(args.model_zip, device=args.device)

    sequences = list_sequences(args.sequences_dir)
    records: list[SequenceRecord] = []
    dropped: list[dict] = []

    for seq_dir in tqdm(sequences, desc=args.model_name, unit="seq"):
        frame_paths = get_sorted_frames(seq_dir)
        if not frame_paths:
            dropped.append({"sequence_id": seq_dir.name, "reason": "no_images"})
            continue
        label = "smoke" if is_wf_sequence(seq_dir) else "fp"
        frames = model.load_sequence(frame_paths)
        output = model.predict(frames)
        records.append(
            build_record(
                sequence_dir=seq_dir,
                label=label,
                frames=frames,
                output=output,
            )
        )

    metrics = compute_metrics(args.model_name, records)

    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (args.output_dir / "dropped.json").write_text(json.dumps(dropped, indent=2))
    predictions = sorted(
        (_record_to_json(r) for r in records),
        key=lambda p: (
            p["score"] is None,
            -(p["score"] if p["score"] is not None else 0.0),
        ),
    )
    (args.output_dir / "predictions.json").write_text(json.dumps(predictions, indent=2))

    y_true = np.array([1 if r.label == "smoke" else 0 for r in records])
    scores = np.array([r.score for r in records], dtype=float)
    scores_finite = np.clip(scores, np.finfo(float).min, np.finfo(float).max)

    cm_counts = np.array(
        [
            [metrics["tn"], metrics["fp"]],
            [metrics["fn"], metrics["tp"]],
        ],
        dtype=float,
    )
    neg_total = metrics["tn"] + metrics["fp"]
    pos_total = metrics["tp"] + metrics["fn"]
    cm_norm = np.array(
        [
            [
                metrics["tn"] / neg_total if neg_total else 0.0,
                metrics["fp"] / neg_total if neg_total else 0.0,
            ],
            [
                metrics["fn"] / pos_total if pos_total else 0.0,
                metrics["tp"] / pos_total if pos_total else 0.0,
            ],
        ],
        dtype=float,
    )

    plot_confusion_matrix(
        cm_counts,
        args.output_dir / "confusion_matrix.png",
        title=f"{args.model_name} (counts)",
        normalized=False,
    )
    plot_confusion_matrix(
        cm_norm,
        args.output_dir / "confusion_matrix_normalized.png",
        title=f"{args.model_name} (row-normalized)",
        normalized=True,
    )
    plot_pr_curve(
        y_true, scores_finite, args.output_dir / "pr_curve.png", title=args.model_name
    )
    plot_roc_curve(
        y_true, scores_finite, args.output_dir / "roc_curve.png", title=args.model_name
    )

    print(json.dumps(metrics, indent=2))
    print(
        f"[{args.model_name}] kept={len(records)} dropped={len(dropped)} "
        f"P={metrics['precision']} R={metrics['recall']} F1={metrics['f1']}"
    )


if __name__ == "__main__":
    main()

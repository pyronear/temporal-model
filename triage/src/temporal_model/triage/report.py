"""Writers for the eval-viewer contract + the triage worklists.

Layout (read by viewer/ with DATA_ROOT=../triage — see viewer/lib/paths.ts):
``data/08_reporting/pyro-annotator/vit_dinov2_finetune/{results.json, details/,
sequences/, model_config.json, dropped.json}`` plus the triage-specific
``unlabeled.json`` / ``review.json`` worklists. Row/detail shapes mirror
``eval/src/temporal_model/eval/evaluate.py`` and
``core/.../details_schema.py``; rows add ``triage_score``/``triage_bucket``.

The ``unlabeled.json`` ``bulk_payload`` is a ready-to-send body for
``POST /api/v1/annotations/sequences/bulk`` — written to disk ONLY. triage never
transmits it; applying it is a separate, deliberate human step.
"""

from __future__ import annotations

import json
from pathlib import Path

from temporal_model.triage.score import ScoredSequence

SOURCE = "pyro-annotator"
MODEL_DIR = "vit_dinov2_finetune"  # viewer/lib/paths.ts MODEL_NAME


def _result_row(s: ScoredSequence) -> dict:
    kept = s.details.get("tubes", {}).get("kept", [])
    decision = "keep" if s.bucket == "review" else "discard"
    return {
        "key": s.key,
        "source": SOURCE,
        "label": s.meta.label,  # "unknown" — unannotated backlog
        "decision": decision,
        "outcome": "n/a",  # no ground truth to score against
        "triage_score": s.score,
        "triage_bucket": s.bucket,
        "score": max((t["logit"] for t in kept), default=None),
        "probability": s.score,
        "num_tubes_kept": len(kept),
        "trigger_frame_index": s.trigger_frame_index,
        "organization_name": s.meta.organization_name,
        "camera_name": s.meta.camera_name,
        "started_at": s.meta.started_at,
    }


def _sequence_view(s: ScoredSequence) -> dict:
    return {
        "key": s.key,
        "source": SOURCE,
        "label": s.meta.label,
        "organization_name": s.meta.organization_name,
        "camera_name": s.meta.camera_name,
        "started_at": s.meta.started_at,
        "frames": [p.as_posix() for p in s.frame_paths],
    }


def write_triage_report(
    output_dir: Path,
    scored: list[ScoredSequence],
    *,
    dropped: list[dict],
    threshold: float,
    model_config: dict,
) -> None:
    out = output_dir / SOURCE / MODEL_DIR
    (out / "details").mkdir(parents=True, exist_ok=True)
    (out / "sequences").mkdir(parents=True, exist_ok=True)

    rows = [_result_row(s) for s in scored]
    (out / "results.json").write_text(json.dumps(rows, indent=2))
    (out / "model_config.json").write_text(
        json.dumps({**model_config, "threshold": threshold}, indent=2, default=str)
    )
    (out / "dropped.json").write_text(json.dumps(dropped, indent=2))
    for s in scored:
        (out / "details" / f"{s.key}.json").write_text(
            json.dumps(s.details, indent=2, default=str)
        )
        (out / "sequences" / f"{s.key}.json").write_text(
            json.dumps(_sequence_view(s), indent=2)
        )

    low = sorted((s for s in scored if s.bucket == "unlabeled"), key=lambda s: s.score)
    high = sorted(
        (s for s in scored if s.bucket == "review"),
        key=lambda s: s.score,
        reverse=True,
    )
    low_ids = [s.sequence_id for s in low]
    (out / "unlabeled.json").write_text(
        json.dumps(
            {
                "threshold": threshold,
                "count": len(low),
                "sequence_ids": low_ids,
                "items": [
                    {"sequence_id": s.sequence_id, "key": s.key, "score": s.score}
                    for s in low
                ],
                # Ready-to-send body for POST /api/v1/annotations/sequences/bulk.
                # WRITTEN TO DISK ONLY — triage never sends it.
                "bulk_payload": {
                    "sequence_ids": low_ids,
                    "false_positive_type": "unlabeled",
                    "is_unsure": False,
                    "force": False,
                },
            },
            indent=2,
        )
    )
    (out / "review.json").write_text(
        json.dumps(
            {
                "threshold": threshold,
                "count": len(high),
                "sequence_ids": [s.sequence_id for s in high],
                "items": [
                    {"sequence_id": s.sequence_id, "key": s.key, "score": s.score}
                    for s in high
                ],
            },
            indent=2,
        )
    )

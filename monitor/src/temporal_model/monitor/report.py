"""Writers for the eval-viewer reporting contract, one tree per organization.

Layout (read by viewer/ with DATA_ROOT=../monitor — see viewer/lib/paths.ts):
``data/08_reporting/<org_slug>/vit_dinov2_finetune/{results.json, details/,
sequences/, model_config.json, dropped.json}``. Shapes mirror
``eval/src/temporal_model/eval/evaluate.py`` rows and
``core/src/temporal_model/core/details_schema.py`` details; results rows add
the monitor-only provenance columns (recorded_probability, replay_matches,
temporal_*_version), which the viewer treats as optional.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from temporal_model.monitor.geometry import tube_stabilized_window
from temporal_model.monitor.store import SequenceMeta, slugify

MODEL_DIR = "vit_dinov2_finetune"  # viewer/lib/paths.ts MODEL_NAME


def decision_from_output(is_smoke: bool) -> str:
    return "keep" if is_smoke else "discard"


def compute_outcome(decision: str, label: str) -> str:
    """Copy of eval's outcomes.compute_outcome (same strings, same n/a rule)."""
    if label == "smoke":
        return "kept-smoke" if decision == "keep" else "discarded-smoke"
    if label == "fp":
        return "kept-fp" if decision == "keep" else "discarded-fp"
    return "n/a"


def reshape_details(api_details: dict[str, Any]) -> dict[str, Any]:
    """Verbose /predict ``details`` -> eval ``BboxTubeDetails`` shape.

    Differences bridged: the API nests tube counters under preprocessing and
    flattens tubes to a list; trigger fields exist only on releases with
    compute_trigger (absent -> None); stabilized_window is never in the API
    response (derived here).
    """
    pre = api_details["preprocessing"]
    dec = api_details["decision"]
    kept = [
        {
            "tube_id": t["tube_id"],
            "start_frame": t["start_frame"],
            "end_frame": t["end_frame"],
            "logit": t["logit"],
            "probability": t.get("probability"),
            "first_crossing_frame": t.get("first_crossing_frame"),
            "entries": t["entries"],
            "stabilized_window": tube_stabilized_window(t["entries"]),
        }
        for t in api_details["tubes"]
    ]
    return {
        "preprocessing": {
            "num_frames_input": pre["num_frames_input"],
            "num_truncated": pre["num_truncated"],
            "padded_frame_indices": pre["padded_frame_indices"],
        },
        "tubes": {
            "num_candidates": pre.get("num_tube_candidates", 0),
            "num_outside_roi": pre.get("num_tubes_outside_roi", 0),
            "kept": kept,
        },
        "decision": {
            "aggregation": dec["aggregation"],
            "threshold": dec["threshold"],
            "trigger_tube_id": dec.get("trigger_tube_id"),
        },
    }


def result_row(
    *,
    meta: SequenceMeta,
    response: dict[str, Any],
    details: dict[str, Any],
    replay_matches: bool | None,
) -> dict[str, Any]:
    """One results.json row: eval columns + monitor provenance extras."""
    kept = details["tubes"]["kept"]
    decision = decision_from_output(response["is_smoke"])
    return {
        "key": meta.key,
        # must equal the reporting tree's <org_slug> dir — the viewer filters
        # rows by string equality with the directory-derived source name
        "source": slugify(meta.organization_name),
        "label": meta.label,
        "decision": decision,
        "outcome": compute_outcome(decision, meta.label),
        "score": max(t["logit"] for t in kept) if kept else None,
        "probability": response.get("probability"),
        "num_tubes_kept": len(kept),
        "trigger_frame_index": response.get("trigger_frame_index"),
        "organization_name": meta.organization_name,
        "camera_name": meta.camera_name,
        "started_at": meta.started_at,
        "recorded_probability": meta.temporal_model_score,
        "replay_matches": replay_matches,
        "temporal_model_version": meta.temporal_model_version,
        "temporal_api_version": meta.temporal_api_version,
    }


@dataclass
class OrgReport:
    """Accumulates one organization's rows/details/views before writing."""

    org_slug: str
    rows: list[dict] = field(default_factory=list)
    details_by_key: dict[str, dict] = field(default_factory=dict)
    views_by_key: dict[str, dict] = field(default_factory=dict)
    model_config: dict | None = None
    dropped: list[dict] = field(default_factory=list)

    def add(self, *, row: dict, details: dict, view: dict, model_config: dict) -> None:
        self.rows.append(row)
        self.details_by_key[row["key"]] = details
        self.views_by_key[row["key"]] = view
        if self.model_config is None:
            self.model_config = model_config

    def drop(self, key: str, reason: str) -> None:
        # eval's dropped.json field name is sequence_id; keep it for tooling parity
        self.dropped.append({"sequence_id": key, "reason": reason})


def write_report(output_dir: Path, report: OrgReport) -> None:
    out = output_dir / report.org_slug / MODEL_DIR
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(report.rows, indent=2))
    (out / "model_config.json").write_text(
        json.dumps(report.model_config or {}, indent=2)
    )
    (out / "dropped.json").write_text(json.dumps(report.dropped, indent=2))
    details_dir = out / "details"
    details_dir.mkdir(exist_ok=True)
    for key, details in report.details_by_key.items():
        (details_dir / f"{key}.json").write_text(json.dumps(details, indent=2))
    sequences_dir = out / "sequences"
    sequences_dir.mkdir(exist_ok=True)
    for key, view in report.views_by_key.items():
        (sequences_dir / f"{key}.json").write_text(json.dumps(view, indent=2))

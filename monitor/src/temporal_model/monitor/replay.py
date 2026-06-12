"""Replay stored sequences through their pinned api release, write reports.

Flow: group sequences by recorded temporal_api_version -> one compose stack
per group (image tag == version; model.zip is baked into the image) -> verify
/health model_version matches each sequence's recorded one -> upload frames
under their original bucket_keys -> POST /predict?verbose=true&
compute_trigger=true (older releases ignore the unknown params; the trigger
fields are then simply absent) -> compare the replayed probability to the
recorded score -> write one eval-viewer tree per organization.
"""

from __future__ import annotations

import logging
import re
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import requests

from temporal_model.monitor import reconstruct
from temporal_model.monitor.report import (
    MODEL_DIR,
    OrgReport,
    reshape_details,
    result_row,
    write_report,
)
from temporal_model.monitor.stack import API_URL, BUCKET, ReplayStack, StackError
from temporal_model.monitor.store import SequenceMeta, iter_metas, slugify

logger = logging.getLogger(__name__)

# Production and replay run on different hardware; identical inputs reproduce
# to ~1e-6 (observed cross-CPU float noise), while any behavioral difference
# (other frames, other tubes) moves the probability by >=1e-2.
SCORE_TOLERANCE = 1e-5
STORE_REL = "data/01_raw/sequences"  # viewer frame paths are relative to monitor/
# Docker image tag charset (no slashes, colons or @ — see run_replay's guard).
_TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class _StackFactory(Protocol):
    def __call__(
        self, compose_file: Path, version: str, image: str | None = None
    ) -> Any: ...


def _default_predict(frames: list[str], roi_xyxyn: list[float] | None) -> dict:
    body: dict[str, Any] = {"bucket": BUCKET, "frames": frames}
    if roi_xyxyn is not None:
        body["roi_xyxyn"] = roi_xyxyn
    resp = requests.post(
        f"{API_URL}/predict",
        params={"verbose": "true", "compute_trigger": "true"},
        json=body,
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json()


SOURCE_SLUG = "alert-api"


def _org_report(reports: dict[str, OrgReport], _meta: SequenceMeta) -> OrgReport:
    if SOURCE_SLUG not in reports:
        reports[SOURCE_SLUG] = OrgReport(org_slug=SOURCE_SLUG)
    return reports[SOURCE_SLUG]


def _files_by_key(
    seq_dir: Path, meta: SequenceMeta, kept: list[str]
) -> dict[str, Path]:
    """First stored file per kept bucket_key (several detections may share one)."""
    files: dict[str, Path] = {}
    for f in meta.frames:
        if f.bucket_key in kept and f.bucket_key not in files:
            files[f.bucket_key] = seq_dir / f.file
    return files


def run_replay(
    *,
    store_dir: Path,
    output_dir: Path,
    compose_file: Path,
    stack_factory: _StackFactory = ReplayStack,
    predict: Callable[[list[str], list[float] | None], dict] = _default_predict,
    trigger_image: str | None = None,
) -> dict[str, int]:
    reports: dict[str, OrgReport] = {}
    groups: dict[str, list[tuple[Path, SequenceMeta]]] = {}
    replayed = mismatched = window_drift = dropped = 0

    for seq_dir, meta in iter_metas(store_dir):
        if not meta.temporal_api_version:
            _org_report(reports, meta).drop(meta.key, "no_temporal_version")
            dropped += 1
            continue
        # The version becomes a Docker image tag; a malformed value (slashes,
        # "@sha256:", registry prefixes) must never redirect the pull.
        if not _TAG_RE.match(meta.temporal_api_version):
            logger.warning(
                "%s: %r is not a valid image tag", meta.key, meta.temporal_api_version
            )
            _org_report(reports, meta).drop(meta.key, "invalid_api_version")
            dropped += 1
            continue
        groups.setdefault(meta.temporal_api_version, []).append((seq_dir, meta))

    for version in sorted(groups):
        items = groups[version]
        logger.info(
            "replaying %d sequence(s) against api version %s", len(items), version
        )
        stack = stack_factory(compose_file, version)
        try:
            stack.up()
        except subprocess.CalledProcessError:
            logger.error("could not start image for api version %s", version)
            for _, meta in items:
                _org_report(reports, meta).drop(meta.key, "image_pull_failed")
                dropped += 1
            continue
        try:
            try:
                health = stack.wait_healthy()
            except StackError:
                logger.exception("api for version %s never became healthy", version)
                for _, meta in items:
                    _org_report(reports, meta).drop(meta.key, "stack_unhealthy")
                    dropped += 1
                continue
            for seq_dir, meta in items:
                outcome = _replay_one(stack, health, seq_dir, meta, reports, predict)
                if outcome == "ok":
                    replayed += 1
                elif outcome == "mismatch":
                    replayed += 1
                    mismatched += 1
                elif outcome == "window_drift":
                    replayed += 1
                    mismatched += 1
                    window_drift += 1
                else:
                    dropped += 1
        finally:
            stack.down()

    trigger_enriched = 0
    if trigger_image is not None and any(r.rows for r in reports.values()):
        trigger_enriched, _ = _enrich_triggers(
            reports, store_dir, compose_file, stack_factory, predict, trigger_image
        )

    for report in reports.values():
        write_report(output_dir, report)
    summary = {
        "replayed": replayed,
        "mismatched": mismatched,
        "window_drift": window_drift,
        "dropped": dropped,
        "trigger_enriched": trigger_enriched,
    }
    logger.info(
        "replay done: %(replayed)d replayed (%(mismatched)d score mismatches, "
        "%(window_drift)d window drift), %(dropped)d dropped, "
        "%(trigger_enriched)d trigger-enriched",
        summary,
    )
    return summary


def _enrich_triggers(
    reports: dict[str, OrgReport],
    store_dir: Path,
    compose_file: Path,
    stack_factory: _StackFactory,
    predict: Callable[[list[str], list[float] | None], dict],
    trigger_image: str,
) -> tuple[int, int]:
    """Second pass with a newer serving build: fill the trigger fields.

    The pinned replay stays authoritative for tubes/probability; the trigger
    search just needs serving code >= the compute_trigger feature. Fields are
    merged ONLY when the enrichment probability reproduces the pinned replay's
    (same window, same model.zip) — otherwise the sequence keeps no trigger.
    Returns (enriched, skipped).
    """
    # Build a lookup from meta.key -> (seq_dir, meta) for all sequences in the store.
    meta_by_key: dict[str, tuple[Path, SequenceMeta]] = {
        meta.key: (seq_dir, meta) for seq_dir, meta in iter_metas(store_dir)
    }

    # There is only one report (alert-api).
    report = next(iter(reports.values()))
    rows = report.rows
    if not rows:
        return 0, 0

    enriched = skipped = 0
    # Use a placeholder version for labeling; image is what matters.
    stack = stack_factory(compose_file, "trigger-enrichment", image=trigger_image)
    try:
        stack.up()
        try:
            stack.wait_healthy()
        except StackError:
            logger.exception("trigger-enrichment stack never became healthy")
            return 0, len(rows)
        for row in rows:
            key = row["key"]
            entry = meta_by_key.get(key)
            if entry is None:
                logger.warning(
                    "trigger enrichment: meta not found for %s, skipping", key
                )
                skipped += 1
                continue
            seq_dir, meta = entry
            _, kept, roi = reconstruct.frames_and_roi(meta.frames)
            files = _files_by_key(seq_dir, meta, kept)
            try:
                stack.upload_frames(files)
                resp = predict(kept, roi)
            except Exception:  # noqa: BLE001
                logger.warning(
                    "trigger enrichment: predict failed for %s, skipping", key
                )
                skipped += 1
                continue
            enr_prob = resp.get("probability")
            pinned_prob = row.get("replayed_probability")
            if (
                enr_prob is None
                or pinned_prob is None
                or abs(enr_prob - pinned_prob) > SCORE_TOLERANCE
            ):
                logger.warning(
                    "trigger enrichment: probability mismatch for %s "
                    "(pinned=%.6f, enrichment=%.6f), skipping trigger fields",
                    key,
                    pinned_prob if pinned_prob is not None else float("nan"),
                    enr_prob if enr_prob is not None else float("nan"),
                )
                skipped += 1
                continue
            # Merge trigger fields into the row and its details.
            row["trigger_frame_index"] = resp.get("trigger_frame_index")
            enr = reshape_details(resp["details"])
            details = report.details_by_key[key]
            details["decision"]["trigger_tube_id"] = enr["decision"]["trigger_tube_id"]
            # Build tube_id -> first_crossing_frame from enrichment kept tubes.
            fcf_by_id = {
                t["tube_id"]: t.get("first_crossing_frame")
                for t in enr["tubes"]["kept"]
            }
            for tube in details["tubes"]["kept"]:
                if tube["tube_id"] in fcf_by_id:
                    tube["first_crossing_frame"] = fcf_by_id[tube["tube_id"]]
            # Record which image was used for enrichment (once per report).
            if (
                report.model_config is not None
                and "trigger_image" not in report.model_config
            ):  # noqa: E501
                report.model_config["trigger_image"] = trigger_image
            enriched += 1
    except subprocess.CalledProcessError:
        logger.error("could not start trigger-enrichment stack (%s)", trigger_image)
        return 0, len(rows)
    finally:
        stack.down()

    logger.info("trigger enrichment: %d enriched, %d skipped", enriched, skipped)
    return enriched, skipped


def _find_matching_window(
    stack: Any,
    seq_dir: Path,
    meta: SequenceMeta,
    recorded: float,
    predict: Callable[[list[str], list[float] | None], dict],
) -> int | None:
    """Distinct-frame count n whose window reproduces the recorded score.

    Production scored "the last <=10 of the first n distinct frames" for some
    unknown n (it stops scoring once a sequence validates, while detections
    keep arriving). Probe candidates ascending and stop at the first match;
    None when no window matches (genuine drift, not window drift). The ROI is
    recomputed per candidate from that window's detections, mirroring the
    production call.
    """
    _, all_keys, _ = reconstruct.frames_and_roi(meta.frames, last_n=None)
    # n == len(all_keys) is the main replay's window, already known mismatched.
    for n in range(reconstruct.MIN_FRAMES, len(all_keys)):
        first_n = set(all_keys[:n])
        subset = [f for f in meta.frames if f.bucket_key in first_n]
        _, kept, roi = reconstruct.frames_and_roi(subset)
        files = _files_by_key(seq_dir, meta, kept)
        try:
            stack.upload_frames(files)
            probability = predict(kept, roi).get("probability")
        except Exception:  # noqa: BLE001 — a failed probe ends the search, not the run
            logger.exception("window probe failed for %s at n=%d", meta.key, n)
            return None
        if probability is not None and abs(recorded - probability) <= SCORE_TOLERANCE:
            return n
    return None


def _replay_one(
    stack: Any,
    health: dict,
    seq_dir: Path,
    meta: SequenceMeta,
    reports: dict[str, OrgReport],
    predict: Callable[[list[str], list[float] | None], dict],
) -> str:
    report = _org_report(reports, meta)
    # version and score are written by the same alert-api UPDATE, but import
    # copies them independently — guard against a version without a score,
    # which result_row could not compare against the threshold.
    if meta.temporal_model_score is None:
        report.drop(meta.key, "no_recorded_score")
        return "no_recorded_score"
    if health.get("model_version") != meta.temporal_model_version:
        report.drop(meta.key, "model_version_mismatch")
        return "model_version_mismatch"
    total, kept, roi = reconstruct.frames_and_roi(meta.frames)
    if total < reconstruct.MIN_FRAMES:
        report.drop(meta.key, "too_few_frames")
        return "too_few_frames"
    files = _files_by_key(seq_dir, meta, kept)
    if len(files) < len(kept) or not all(p.is_file() for p in files.values()):
        report.drop(meta.key, "no_images")
        return "no_images"
    try:
        stack.upload_frames(files)
        response = predict(kept, roi)
    except Exception:  # noqa: BLE001 — one bad sequence must not stop the run
        logger.exception("predict failed for %s", meta.key)
        report.drop(meta.key, "predict_failed")
        return "predict_failed"

    details = reshape_details(response["details"])
    matches = _score_matches(meta.temporal_model_score, response.get("probability"))
    matched_window_frames = None
    if matches is False:
        matched_window_frames = _find_matching_window(
            stack, seq_dir, meta, meta.temporal_model_score, predict
        )
        if matched_window_frames is not None:
            logger.info(
                "%s: recorded score reproduced at the first-%d-frame window "
                "(window drift, not model drift)",
                meta.key,
                matched_window_frames,
            )
    org = slugify(meta.organization_name)
    cam = slugify(meta.camera_name)
    # The frames the model actually saw, in request order, as paths relative
    # to monitor/ (the viewer resolves them against DATA_ROOT).
    frames_rel = []
    for key in kept:
        rel = files[key].relative_to(seq_dir)  # e.g. images/detection_100.jpg
        frames_rel.append(
            f"{STORE_REL}/{org}/{cam}/seq_{meta.sequence_id}/{rel.as_posix()}"
        )
    view = {
        "key": meta.key,
        # source matches the single reporting tree under "alert-api"
        "source": SOURCE_SLUG,
        "label": meta.label,
        "organization_name": meta.organization_name,
        "camera_name": meta.camera_name,
        "started_at": meta.started_at,
        "frames": frames_rel,
    }
    report.add(
        row=result_row(
            meta=meta,
            response=response,
            details=details,
            replay_matches=matches,
            matched_window_frames=matched_window_frames,
        ),
        details=details,
        view=view,
        model_config={
            "variant": MODEL_DIR,
            "decision": details["decision"],
            "model_version": health.get("model_version"),
            "api_version": health.get("api_version"),
        },
    )
    if matches is None or matches:
        return "ok"
    return "window_drift" if matched_window_frames is not None else "mismatch"


def _score_matches(recorded: float | None, replayed: float | None) -> bool | None:
    if recorded is None:
        return None
    if replayed is None:
        return False
    return abs(recorded - replayed) <= SCORE_TOLERANCE

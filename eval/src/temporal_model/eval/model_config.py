"""Read a packaged model.zip's metadata into one plain dict for the viewer.

Merges manifest.yaml (detector provenance), config.yaml (decision/infer/
model_input/tubes/classifier), and logistic_calibrator.json. Tolerant: a missing
or unreadable zip returns {} (e.g. test runs that monkeypatch model loading); a
missing member is omitted or set to None.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import yaml


def _read_member(z: zipfile.ZipFile, name: str):
    """Parse a zip member by extension (yaml/json), or None if absent."""
    if name not in z.namelist():
        return None
    raw = z.read(name).decode()
    if name.endswith((".yaml", ".yml")):
        return yaml.safe_load(raw)
    return json.loads(raw)


def read_model_config(model_zip: Path) -> dict:
    """Merged, JSON-serializable view of a packaged model's config.

    Returns {} when the zip is missing/unreadable. Keys: detector, variant,
    train_git_sha (from manifest.yaml); decision, infer, model_input, tubes,
    classifier (from config.yaml); calibrator (logistic_calibrator.json or None).
    """
    model_zip = Path(model_zip)
    if not model_zip.exists():
        return {}
    try:
        with zipfile.ZipFile(model_zip) as z:
            manifest = _read_member(z, "manifest.yaml") or {}
            config = _read_member(z, "config.yaml") or {}
            calibrator = _read_member(z, "logistic_calibrator.json")
    except (zipfile.BadZipFile, OSError):
        return {}
    return {
        "detector": manifest.get("detector"),
        "variant": manifest.get("variant"),
        "train_git_sha": manifest.get("train_git_sha"),
        "decision": config.get("decision"),
        "infer": config.get("infer"),
        "model_input": config.get("model_input"),
        "tubes": config.get("tubes"),
        "classifier": config.get("classifier"),
        "calibrator": calibrator,
    }

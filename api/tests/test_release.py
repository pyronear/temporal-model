"""Tests for the release CLI (HuggingFace calls mocked)."""

import zipfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from temporal_model.api import release


def _make_zip(path: Path, manifest: dict) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("manifest.yaml", yaml.dump(manifest))
        zf.writestr("classifier.ckpt", b"weights")
    return path


def test_read_model_version_present(tmp_path: Path) -> None:
    z = _make_zip(tmp_path / "m.zip", {"variant": "vit", "model_version": "1.2.0"})
    assert release.read_model_version(z) == "1.2.0"


def test_read_model_version_absent(tmp_path: Path) -> None:
    z = _make_zip(tmp_path / "m.zip", {"variant": "vit"})
    assert release.read_model_version(z) is None


def test_stamp_sets_version_and_preserves_entries(tmp_path: Path) -> None:
    z = _make_zip(tmp_path / "m.zip", {"variant": "vit"})
    release.stamp_model_version(z, "0.2.0")
    with zipfile.ZipFile(z) as zf:
        names = set(zf.namelist())
        manifest = yaml.safe_load(zf.read("manifest.yaml"))
        assert zf.read("classifier.ckpt") == b"weights"
    assert names == {"manifest.yaml", "classifier.ckpt"}
    assert manifest["model_version"] == "0.2.0"
    assert manifest["variant"] == "vit"  # other fields preserved


def test_fetch_asserts_matching_version(tmp_path: Path) -> None:
    src = _make_zip(tmp_path / "hf.zip", {"variant": "vit", "model_version": "0.2.0"})
    out = tmp_path / "out" / "model.zip"
    result = release.fetch(
        "0.2.0", out, repo="org/r", _downloader=lambda **kw: str(src)
    )
    assert result == out and out.read_bytes() == src.read_bytes()


def test_fetch_passes_correct_revision(tmp_path: Path) -> None:
    src = _make_zip(tmp_path / "hf.zip", {"model_version": "0.2.0"})
    calls = {}

    def fake_dl(**kw):
        calls.update(kw)
        return str(src)

    release.fetch("0.2.0", tmp_path / "o.zip", repo="org/r", _downloader=fake_dl)
    assert calls["repo_id"] == "org/r"
    assert calls["filename"] == "model.zip"
    assert calls["revision"] == "v0.2.0"


def test_fetch_version_mismatch_raises(tmp_path: Path) -> None:
    src = _make_zip(tmp_path / "hf.zip", {"model_version": "9.9.9"})
    with pytest.raises(ValueError, match="model_version"):
        release.fetch(
            "0.2.0",
            tmp_path / "o.zip",
            repo="org/r",
            _downloader=lambda **kw: str(src),
        )


def test_render_model_card_substitutes_version() -> None:
    card = release.render_model_card("1.4.2")
    assert release.VERSION_PLACEHOLDER not in card
    assert "v1.4.2" in card  # revision pin in the usage snippet
    assert card.startswith("---")  # HF frontmatter preserved


def test_publish_stamps_uploads_zip_and_card_and_tags(tmp_path: Path) -> None:
    z = _make_zip(tmp_path / "m.zip", {"variant": "vit"})  # no model_version yet
    api = MagicMock()
    # capture the version stamped into the uploaded model.zip (a temp copy)
    zip_version = {}

    def capture(**kw):
        if kw["path_in_repo"] == "model.zip":
            zip_version["v"] = release.read_model_version(Path(kw["path_or_fileobj"]))

    api.upload_file.side_effect = capture
    release.publish("0.3.0", z, repo="org/r", api=api)

    # the caller's file is NOT mutated...
    assert release.read_model_version(z) is None
    # ...but the uploaded copy was stamped
    assert zip_version["v"] == "0.3.0"

    # both the model and the rendered card are uploaded
    calls = {c.kwargs["path_in_repo"]: c.kwargs for c in api.upload_file.call_args_list}
    assert set(calls) == {"model.zip", "README.md"}
    assert calls["model.zip"]["repo_id"] == "org/r"
    card = calls["README.md"]["path_or_fileobj"].decode()
    assert "0.3.0" in card and release.VERSION_PLACEHOLDER not in card
    # tagged v0.3.0
    tag = api.create_tag.call_args.kwargs
    assert tag["tag"] == "v0.3.0" and tag["repo_id"] == "org/r"

"""Release CLI: publish/fetch the packaged model.zip to/from HuggingFace.

The packaged model.zip is released on a HuggingFace model repo, one git
revision/tag ``v<version>`` per release. ``publish`` stamps the version into the
manifest and uploads + tags; ``fetch`` downloads a revision and asserts the
version. Runs identically locally (maintainer HF token for publish) and in CI
(public repo, no token for fetch).

The torch-free ``model_onnx.zip`` (see ``temporal_model.core.export_onnx``)
travels with the same release: ``publish --onnx-file`` uploads it under the
same tag and ``fetch --onnx`` downloads it instead of ``model.zip``.
"""

import argparse
import hashlib
import shutil
import tempfile
import zipfile
from importlib.resources import files
from pathlib import Path

import yaml
from huggingface_hub import HfApi, hf_hub_download

RELEASE_REPO = "pyronear/temporal-model"
MODEL_FILENAME = "model.zip"
ONNX_MODEL_FILENAME = "model_onnx.zip"
MANIFEST_FILENAME = "manifest.yaml"
CARD_TEMPLATE = "model_card.md"
CARD_FILENAME = "README.md"
VERSION_PLACEHOLDER = "{{VERSION}}"


def _tag(version: str) -> str:
    return f"v{version}"


def render_model_card(version: str) -> str:
    """Render the HF model card, substituting the release version.

    The template ships with this package so ``publish`` always has it,
    regardless of the working directory.
    """
    template = (files("temporal_model.api") / CARD_TEMPLATE).read_text(encoding="utf-8")
    return template.replace(VERSION_PLACEHOLDER, version)


def read_model_version(zip_path: Path) -> str | None:
    """Return ``manifest.model_version`` from a model.zip (None if absent)."""
    with zipfile.ZipFile(zip_path) as zf:
        manifest = yaml.safe_load(zf.read(MANIFEST_FILENAME))
    return manifest.get("model_version")


def _update_manifest(zip_path: Path, **fields) -> None:
    """Merge ``fields`` into the zip's manifest (rewrites the archive in place)."""
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        blobs = {n: zf.read(n) for n in names}
    manifest = yaml.safe_load(blobs[MANIFEST_FILENAME])
    manifest.update(fields)
    blobs[MANIFEST_FILENAME] = yaml.dump(manifest, default_flow_style=False).encode()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for n in names:
            zf.writestr(n, blobs[n])


def stamp_model_version(zip_path: Path, version: str) -> None:
    """Set ``manifest.model_version = version`` inside the zip (rewrites archive)."""
    _update_manifest(zip_path, model_version=version)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_onnx_source_sha256(onnx_zip: Path) -> str | None:
    """Return ``manifest.source.sha256`` of a ``model_onnx.zip`` (None if absent)."""
    with zipfile.ZipFile(onnx_zip) as zf:
        manifest = yaml.safe_load(zf.read(MANIFEST_FILENAME))
    return (manifest.get("source") or {}).get("sha256")


def verify_onnx_source(onnx_zip: Path, model_zip: Path) -> None:
    """Refuse to pair an ONNX archive with a ``model.zip`` it was not exported from.

    Raises:
        ValueError: if the ONNX manifest's source hash differs from ``model_zip``'s.
    """
    recorded = read_onnx_source_sha256(onnx_zip)
    actual = _sha256(model_zip)
    if recorded != actual:
        raise ValueError(
            f"{onnx_zip.name} was exported from a model.zip with sha256 "
            f"{recorded!r}, but {model_zip.name} has {actual!r}"
        )


def stamp_onnx_source(onnx_zip: Path, model_zip: Path) -> None:
    """Point ``manifest.source`` of ``onnx_zip`` at ``model_zip`` (name + SHA-256).

    Stamping the version rewrites ``model.zip``, so the checksum recorded at
    export time no longer identifies the released archive; re-stamp it from
    the staged copy that actually gets uploaded.
    """
    _update_manifest(
        onnx_zip,
        source={"package": model_zip.name, "sha256": _sha256(model_zip)},
    )


def fetch(
    version: str,
    output_path: Path,
    *,
    filename: str = MODEL_FILENAME,
    repo: str = RELEASE_REPO,
    _downloader=hf_hub_download,
) -> Path:
    """Download ``filename`` at HF revision ``v<version>``, assert version, write it.

    ``filename`` is ``model.zip`` (default) or ``model_onnx.zip``; both carry
    a ``manifest.yaml`` with ``model_version``.

    Raises:
        ValueError: if the downloaded manifest's ``model_version`` != ``version``.
    """
    downloaded = Path(
        _downloader(repo_id=repo, filename=filename, revision=_tag(version))
    )
    actual = read_model_version(downloaded)
    if actual != version:
        raise ValueError(
            f"model_version mismatch: {repo}@{_tag(version)} has {actual!r}, "
            f"expected {version!r}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(downloaded, output_path)
    return output_path


def publish(
    version: str,
    file_path: Path,
    *,
    onnx_path: Path | None = None,
    repo: str = RELEASE_REPO,
    api: HfApi | None = None,
) -> None:
    """Stamp the version into the manifest, upload ``model.zip`` (and
    ``model_onnx.zip`` when ``onnx_path`` is given) + the rendered model card,
    then tag ``v<version>``.

    The caller's files are **not** modified: the version is stamped into a
    temporary copy, which is what gets uploaded. The model card (README.md) is
    rendered with this version so the repo always advertises the latest release.
    Versions are immutable — if the ``v<version>`` tag already exists,
    ``create_tag`` raises (no silent overwrite).

    Raises:
        ValueError: if ``onnx_path`` was not exported from ``file_path``.
    """
    hf = api or HfApi()
    if onnx_path is not None:
        verify_onnx_source(onnx_path, file_path)
    with tempfile.TemporaryDirectory() as td:
        staged_model = Path(td) / MODEL_FILENAME
        shutil.copyfile(file_path, staged_model)
        stamp_model_version(staged_model, version)
        uploads = [staged_model]
        if onnx_path is not None:
            staged_onnx = Path(td) / ONNX_MODEL_FILENAME
            shutil.copyfile(onnx_path, staged_onnx)
            stamp_model_version(staged_onnx, version)
            stamp_onnx_source(staged_onnx, staged_model)
            uploads.append(staged_onnx)
        for staged in uploads:
            hf.upload_file(
                path_or_fileobj=str(staged),
                path_in_repo=staged.name,
                repo_id=repo,
                repo_type="model",
            )
    hf.upload_file(
        path_or_fileobj=render_model_card(version).encode("utf-8"),
        path_in_repo=CARD_FILENAME,
        repo_id=repo,
        repo_type="model",
    )
    hf.create_tag(repo_id=repo, tag=_tag(version), repo_type="model")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=RELEASE_REPO)
    sub = parser.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="download model.zip for a version from HF")
    f.add_argument("--version", required=True)
    f.add_argument("--output", type=Path, required=True)
    f.add_argument(
        "--onnx",
        action="store_true",
        help=f"download {ONNX_MODEL_FILENAME} instead of {MODEL_FILENAME}",
    )

    p = sub.add_parser("publish", help="stamp + upload + tag a model.zip to HF")
    p.add_argument("--version", required=True)
    p.add_argument("--file", type=Path, required=True)
    p.add_argument(
        "--onnx-file",
        type=Path,
        help=f"also upload this {ONNX_MODEL_FILENAME} under the same tag",
    )

    args = parser.parse_args()
    if args.cmd == "fetch":
        filename = ONNX_MODEL_FILENAME if args.onnx else MODEL_FILENAME
        out = fetch(args.version, args.output, filename=filename, repo=args.repo)
        print(f"fetched {args.repo}@{_tag(args.version)}/{filename} -> {out}")
    else:
        publish(args.version, args.file, onnx_path=args.onnx_file, repo=args.repo)
        print(f"published {args.file} -> {args.repo}@{_tag(args.version)}")


if __name__ == "__main__":
    main()

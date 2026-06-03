"""Release CLI: publish/fetch the packaged model.zip to/from HuggingFace.

The packaged model.zip is released on a HuggingFace model repo, one git
revision/tag ``v<version>`` per release. ``publish`` stamps the version into the
manifest and uploads + tags; ``fetch`` downloads a revision and asserts the
version. Runs identically locally (maintainer HF token for publish) and in CI
(public repo, no token for fetch).
"""

import argparse
import shutil
import tempfile
import zipfile
from importlib.resources import files
from pathlib import Path

import yaml
from huggingface_hub import HfApi, hf_hub_download

RELEASE_REPO = "pyronear/temporal-model"
MODEL_FILENAME = "model.zip"
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


def stamp_model_version(zip_path: Path, version: str) -> None:
    """Set ``manifest.model_version = version`` inside the zip (rewrites archive)."""
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        blobs = {n: zf.read(n) for n in names}
    manifest = yaml.safe_load(blobs[MANIFEST_FILENAME])
    manifest["model_version"] = version
    blobs[MANIFEST_FILENAME] = yaml.dump(manifest, default_flow_style=False).encode()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for n in names:
            zf.writestr(n, blobs[n])


def fetch(
    version: str,
    output_path: Path,
    *,
    repo: str = RELEASE_REPO,
    _downloader=hf_hub_download,
) -> Path:
    """Download ``model.zip`` at HF revision ``v<version>``, assert version, write it.

    Raises:
        ValueError: if the downloaded manifest's ``model_version`` != ``version``.
    """
    downloaded = Path(
        _downloader(repo_id=repo, filename=MODEL_FILENAME, revision=_tag(version))
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
    repo: str = RELEASE_REPO,
    api: HfApi | None = None,
) -> None:
    """Stamp the version into the manifest, upload ``model.zip`` + the rendered
    model card, then tag ``v<version>``.

    The caller's ``file_path`` is **not** modified: the version is stamped into a
    temporary copy, which is what gets uploaded. The model card (README.md) is
    rendered with this version so the repo always advertises the latest release.
    Versions are immutable — if the ``v<version>`` tag already exists,
    ``create_tag`` raises (no silent overwrite).
    """
    hf = api or HfApi()
    with tempfile.TemporaryDirectory() as td:
        staged = Path(td) / MODEL_FILENAME
        shutil.copyfile(file_path, staged)
        stamp_model_version(staged, version)
        hf.upload_file(
            path_or_fileobj=str(staged),
            path_in_repo=MODEL_FILENAME,
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

    p = sub.add_parser("publish", help="stamp + upload + tag a model.zip to HF")
    p.add_argument("--version", required=True)
    p.add_argument("--file", type=Path, required=True)

    args = parser.parse_args()
    if args.cmd == "fetch":
        out = fetch(args.version, args.output, repo=args.repo)
        print(f"fetched {args.repo}@{_tag(args.version)} -> {out}")
    else:
        publish(args.version, args.file, repo=args.repo)
        print(f"published {args.file} -> {args.repo}@{_tag(args.version)}")


if __name__ == "__main__":
    main()

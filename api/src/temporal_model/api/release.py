"""Release CLI: publish/fetch the packaged model.zip to/from HuggingFace.

The packaged model.zip is released on a HuggingFace model repo, one git
revision/tag ``v<version>`` per release. ``publish`` stamps the version into the
manifest and uploads + tags; ``fetch`` downloads a revision and asserts the
version. Runs identically locally (maintainer HF token for publish) and in CI
(public repo, no token for fetch).
"""

import argparse
import shutil
import zipfile
from pathlib import Path

import yaml
from huggingface_hub import HfApi, hf_hub_download

RELEASE_REPO = "pyronear/temporal-model"
MODEL_FILENAME = "model.zip"
MANIFEST_FILENAME = "manifest.yaml"


def _tag(version: str) -> str:
    return f"v{version}"


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
    """Stamp the version into the manifest, upload ``model.zip``, tag ``v<version>``."""
    stamp_model_version(file_path, version)
    hf = api or HfApi()
    hf.upload_file(
        path_or_fileobj=str(file_path),
        path_in_repo=MODEL_FILENAME,
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

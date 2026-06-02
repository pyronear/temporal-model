"""S3 frame fetching.

Downloads frame objects (by key) into a per-request temp directory, preserving
each key's basename and the requested order. Credentials and bucket come from
settings / the boto3 chain — never from the request.
"""

from pathlib import Path
from typing import Any

import boto3
import botocore.exceptions as botoexc

from .errors import FrameNotFound, S3Unavailable
from .settings import Settings

_NOT_FOUND_CODES = {"404", "NoSuchKey", "NoSuchBucket"}


def make_s3_client(settings: Settings) -> Any:
    """Build a boto3 S3 client. ``endpoint_url`` empty → real AWS."""
    return boto3.client(
        "s3",
        region_name=settings.s3_region or None,
        endpoint_url=settings.s3_endpoint_url or None,
    )


def fetch_frames(s3_client, bucket: str, keys: list[str], dest_dir: Path) -> list[Path]:
    """Download ``keys`` from ``bucket`` into ``dest_dir``, in order.

    Each key is written to ``dest_dir/<NNNN>/<basename>`` so basenames are
    preserved exactly and never collide. Returns local paths in request order.
    """
    paths: list[Path] = []
    for i, key in enumerate(keys):
        basename = key.rsplit("/", 1)[-1]
        subdir = dest_dir / f"{i:04d}"
        subdir.mkdir(parents=True, exist_ok=True)
        dst = subdir / basename
        try:
            s3_client.download_file(bucket, key, str(dst))
        except botoexc.ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code", "")
            if error_code in _NOT_FOUND_CODES:
                raise FrameNotFound(f"frame not found: {key}") from exc
            raise S3Unavailable(f"S3 error fetching {key}: {error_code}") from exc
        except botoexc.EndpointConnectionError as exc:
            raise S3Unavailable(f"S3 endpoint unreachable: {exc}") from exc
        paths.append(dst)
    return paths

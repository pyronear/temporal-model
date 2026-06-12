"""Docker compose lifecycle for the pinned-release replay stack.

One ReplayStack per api-version group: ``up()`` starts
``pyronear/temporal-model-api:<version>`` (model.zip baked into the image) +
a throwaway MinIO; ``upload_frames`` puts a sequence's frames under their
ORIGINAL bucket_keys so the request body is byte-identical to production's;
``down()`` removes containers and the MinIO volume.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import boto3
import requests

IMAGE_REPO = "pyronear/temporal-model-api"
COMPOSE_PROJECT = "temporal-monitor-replay"
API_URL = "http://localhost:18000"
MINIO_URL = "http://localhost:19000"
BUCKET = "frames"
_MINIO_CREDS = {
    "aws_access_key_id": "minioadmin",
    "aws_secret_access_key": "minioadmin",
}


class StackError(RuntimeError):
    pass


class ReplayStack:
    def __init__(
        self, compose_file: Path, version: str, image: str | None = None
    ) -> None:
        self.compose_file = compose_file
        self.version = version
        self.image = image or f"{IMAGE_REPO}:{self.version}"

    def _compose(self, *args: str) -> None:
        subprocess.run(
            [
                "docker",
                "compose",
                "-f",
                str(self.compose_file),
                "-p",
                COMPOSE_PROJECT,
                *args,
            ],
            env={**os.environ, "MONITOR_API_IMAGE": self.image},
            check=True,
        )

    def up(self) -> None:
        """Start the stack; raises CalledProcessError on pull/start failure."""
        self._compose("up", "-d")

    def down(self) -> None:
        self._compose("down", "-v")

    def _fetch_health(self) -> dict:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        resp.raise_for_status()
        return resp.json()

    def wait_healthy(self, timeout_s: float = 300, poll_s: float = 2) -> dict:
        """Poll /health until model_loaded; returns the health payload."""
        deadline = time.monotonic() + timeout_s
        last: str = "no response"
        while time.monotonic() < deadline:
            try:
                health = self._fetch_health()
            except Exception as exc:  # noqa: BLE001 — keep polling until deadline
                last = repr(exc)
            else:
                if health.get("model_loaded"):
                    return health
                last = str(health)
            time.sleep(poll_s)
        raise StackError(f"api at {API_URL} never became healthy: {last}")

    def _s3_client(self):
        return boto3.client("s3", endpoint_url=MINIO_URL, **_MINIO_CREDS)

    def upload_frames(self, files_by_key: dict[str, Path]) -> None:
        """Upload local frame files under their original S3 bucket_keys."""
        client = self._s3_client()
        for key, path in files_by_key.items():
            client.upload_file(str(path), BUCKET, key)

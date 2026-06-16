"""Read-only pyro-annotator HTTP client.

INVARIANT: the only non-GET request this client issues is the login POST that
mints a bearer token (it writes no annotation data). There is deliberately no
post/patch/put/delete method here — triage never mutates annotator state.
Pagination is the annotator's ``page``/``size`` scheme returning
``{"items", "page", "pages", "total"}``.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import requests

PAGE_SIZE = 100  # annotator enforces size <= 100
TIMEOUT_S = 60


@dataclass(frozen=True)
class AnnotatorApiConfig:
    url: str
    login: str
    password: str

    @classmethod
    def from_env(cls) -> AnnotatorApiConfig:
        try:
            return cls(
                url=os.environ["ANNOTATOR_API_URL"].rstrip("/"),
                login=os.environ["ANNOTATOR_API_LOGIN"],
                password=os.environ["ANNOTATOR_API_PASSWORD"],
            )
        except KeyError as exc:
            raise SystemExit(
                f"missing env var {exc.args[0]} (see triage/.envrc.example)"
            ) from exc


class AnnotatorApiClient:
    def __init__(
        self, config: AnnotatorApiConfig, session: requests.Session | None = None
    ) -> None:
        self.config = config
        self.session = session if session is not None else requests.Session()
        self.token: str | None = None

    def login(self) -> None:
        """POST /auth/login -> bearer token. The only non-GET call triage makes."""
        resp = self.session.post(
            f"{self.config.url}/api/v1/auth/login",
            json={"username": self.config.login, "password": self.config.password},
            timeout=TIMEOUT_S,
        )
        resp.raise_for_status()
        self.token = resp.json()["access_token"]

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        if self.token is None:
            raise RuntimeError("call login() first")
        resp = self.session.get(
            f"{self.config.url}{path}",
            headers={"Authorization": f"Bearer {self.token}"},
            params=params,
            timeout=TIMEOUT_S,
        )
        resp.raise_for_status()
        return resp.json()

    def iter_unannotated_sequences(
        self, *, page_size: int = PAGE_SIZE, limit: int | None = None
    ) -> Iterator[dict]:
        """Yield sequences with no annotation (has_annotation=false), newest first.

        Stops early once ``limit`` sequences have been yielded — and does not
        fetch further pages — so ``--limit`` is cheap for small test pulls.
        """
        yielded = 0
        page = 1
        while True:
            payload = self._get(
                "/api/v1/sequences/",
                {
                    "has_annotation": False,
                    "page": page,
                    "size": min(page_size, PAGE_SIZE),
                    "order_by": "created_at",
                    "order_direction": "desc",
                },
            )
            items = payload.get("items", [])
            for item in items:
                yield item
                yielded += 1
                if limit is not None and yielded >= limit:
                    return
            if page >= payload.get("pages", page) or not items:
                return
            page += 1

    def list_detections(self, sequence_id: int) -> list[dict]:
        """All detections (frames) of a sequence, oldest first."""
        rows: list[dict] = []
        page = 1
        while True:
            payload = self._get(
                "/api/v1/detections/",
                {
                    "sequence_id": sequence_id,
                    "page": page,
                    "size": PAGE_SIZE,
                    "order_by": "recorded_at",
                    "order_direction": "asc",
                },
            )
            items = payload.get("items", [])
            rows.extend(items)
            if page >= payload.get("pages", page) or not items:
                return rows
            page += 1

    def detection_image_url(self, detection_id: int) -> str:
        """Signed, time-limited download URL for a detection's image."""
        return self._get(f"/api/v1/detections/{detection_id}/url")["url"]

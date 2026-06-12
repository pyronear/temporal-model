"""Thin alert-api HTTP client: login, sequences, detections, cameras, orgs.

Pagination: the sequences/detections list endpoints page at 100 rows; helpers
loop with ``offset`` until a short page. ``/cameras/`` and ``/organizations/``
return everything in one response (no limit param server-side). Detections are
fetched COMPLETELY and oldest first — production ROI reconstruction uses every
detection of a sequence (pyro-api ``validation._sequence_frames_and_roi`` runs
an unbounded ``fetch_all``), so a truncated import would change the replayed
call.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import requests

PAGE_SIZE = 100  # detections endpoint enforces le=100; elsewhere just our batch size
TIMEOUT_S = 60


@dataclass(frozen=True)
class AlertApiConfig:
    url: str
    login: str
    password: str

    @classmethod
    def from_env(cls) -> AlertApiConfig:
        try:
            return cls(
                url=os.environ["ALERT_API_URL"].rstrip("/"),
                login=os.environ["ALERT_API_LOGIN"],
                password=os.environ["ALERT_API_PASSWORD"],
            )
        except KeyError as exc:
            raise SystemExit(
                f"missing env var {exc.args[0]} (see monitor/.envrc.example)"
            ) from exc


class AlertApiClient:
    def __init__(
        self, config: AlertApiConfig, session: requests.Session | None = None
    ) -> None:
        self.config = config
        self.session = session if session is not None else requests.Session()
        self.token: str | None = None

    def login(self) -> None:
        resp = self.session.post(
            f"{self.config.url}/api/v1/login/creds",
            data={"username": self.config.login, "password": self.config.password},
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

    def _get_paginated(self, path: str, params: dict[str, Any]) -> list[dict]:
        rows: list[dict] = []
        offset = 0
        while True:
            page = self._get(path, {**params, "limit": PAGE_SIZE, "offset": offset})
            rows.extend(page)
            if len(page) < PAGE_SIZE:
                return rows
            offset += PAGE_SIZE

    def list_sequences_for_date(self, day: str) -> list[dict]:
        """All sequences started on ``day`` (YYYY-MM-DD).

        The server scopes results to the authenticated account's organization.
        """
        return self._get_paginated("/api/v1/sequences/all/fromdate", {"from_date": day})

    def list_sequence_detections(self, sequence_id: int) -> list[dict]:
        """ALL detections of a sequence, oldest first (see module docstring)."""
        return self._get_paginated(
            f"/api/v1/sequences/{sequence_id}/detections", {"desc": False}
        )

    def list_cameras(self) -> list[dict]:
        return self._get("/api/v1/cameras/", {"include_non_trustable": True})

    def list_organizations(self) -> list[dict]:
        return self._get("/api/v1/organizations/")

    def get_sequence(self, sequence_id: int) -> dict | None:
        """One sequence by id (admin: any org), or None when it doesn't exist."""
        try:
            return self._get(f"/api/v1/sequences/{sequence_id}")
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                return None
            raise

import pytest

from temporal_model.triage.annotator_api import AnnotatorApiClient, AnnotatorApiConfig

CONFIG = AnnotatorApiConfig(
    url="https://annotator.test", login="arthur", password="secret"
)


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise AssertionError(f"HTTP {self.status_code}")


class RecordingSession:
    """Records every HTTP call so we can assert the client is read-only."""

    def __init__(self, pages):
        self._pages = pages  # list of payloads returned by successive GETs
        self.calls = []  # (method, url) tuples

    def post(self, url, **kw):
        self.calls.append(("POST", url))
        return FakeResponse({"access_token": "tok", "token_type": "bearer"})

    def get(self, url, **kw):
        self.calls.append(("GET", url))
        return FakeResponse(self._pages.pop(0))


def test_login_then_paginate_unannotated():
    pages = [
        {"items": [{"id": 1}, {"id": 2}], "page": 1, "pages": 2, "total": 3},
        {"items": [{"id": 3}], "page": 2, "pages": 2, "total": 3},
    ]
    session = RecordingSession(pages)
    client = AnnotatorApiClient(CONFIG, session=session)
    client.login()
    seqs = list(client.iter_unannotated_sequences(page_size=2))
    assert [s["id"] for s in seqs] == [1, 2, 3]


def test_limit_caps_results_and_stops_paging():
    pages = [{"items": [{"id": 1}, {"id": 2}], "page": 1, "pages": 9, "total": 99}]
    session = RecordingSession(pages)
    client = AnnotatorApiClient(CONFIG, session=session)
    client.login()
    seqs = list(client.iter_unannotated_sequences(page_size=2, limit=2))
    assert [s["id"] for s in seqs] == [1, 2]
    # only the first page was fetched (1 login POST + 1 GET)
    assert session.calls == [
        ("POST", "https://annotator.test/api/v1/auth/login"),
        ("GET", "https://annotator.test/api/v1/sequences/"),
    ]


def test_client_is_read_only_only_login_posts():
    """The client must never expose a mutating verb, and the only POST is login."""
    session = RecordingSession([])
    client = AnnotatorApiClient(CONFIG, session=session)
    client.login()
    # No write methods exist on the client.
    for verb in ("post", "patch", "put", "delete"):
        assert not hasattr(client, verb)
    # The single POST it makes is the auth login (writes no annotation data).
    assert session.calls == [("POST", "https://annotator.test/api/v1/auth/login")]


def test_get_requires_login():
    client = AnnotatorApiClient(CONFIG, session=RecordingSession([]))
    with pytest.raises(RuntimeError):
        client.list_detections(1)

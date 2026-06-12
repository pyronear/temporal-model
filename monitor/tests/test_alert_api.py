import pytest
import requests

from temporal_model.monitor.alert_api import AlertApiClient, AlertApiConfig


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            exc = requests.HTTPError(response=self)
            raise exc

    def json(self):
        return self.payload


class FakeSession:
    """Records requests; pops queued responses in FIFO order."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, data=None, timeout=None):
        self.calls.append(("POST", url, data))
        return self.responses.pop(0)

    def get(self, url, headers=None, params=None, timeout=None):
        self.calls.append(("GET", url, params))
        return self.responses.pop(0)


CONFIG = AlertApiConfig(url="https://api.test", login="user", password="pw")


def make_client(responses):
    # Each item is either a plain payload (200 OK) or a FakeResponse already built.
    wrapped = [r if isinstance(r, FakeResponse) else FakeResponse(r) for r in responses]
    session = FakeSession(wrapped)
    client = AlertApiClient(CONFIG, session=session)
    return client, session


def test_login_posts_form_credentials():
    client, session = make_client([{"access_token": "tok"}])
    client.login()
    method, url, data = session.calls[0]
    assert (method, url) == ("POST", "https://api.test/api/v1/login/creds")
    assert data == {"username": "user", "password": "pw"}
    assert client.token == "tok"


def test_list_sequences_for_date_paginates_until_short_page():
    page1 = [{"id": i} for i in range(100)]
    page2 = [{"id": 100}]
    client, session = make_client([{"access_token": "t"}, page1, page2])
    client.login()
    seqs = client.list_sequences_for_date("2026-06-11")
    assert len(seqs) == 101
    # two GET calls with increasing offset
    gets = [c for c in session.calls if c[0] == "GET"]
    assert gets[0][2]["from_date"] == "2026-06-11"
    assert gets[0][2]["offset"] == 0
    assert gets[1][2]["offset"] == 100


def test_list_sequence_detections_paginates_ascending():
    page1 = [{"id": i} for i in range(100)]
    page2 = [{"id": 100}]
    client, session = make_client([{"access_token": "t"}, page1, page2])
    client.login()
    dets = client.list_sequence_detections(42)
    assert len(dets) == 101
    gets = [c for c in session.calls if c[0] == "GET"]
    assert gets[0][1].endswith("/api/v1/sequences/42/detections")
    assert gets[0][2] == {"limit": 100, "offset": 0, "desc": False}
    assert gets[1][2]["offset"] == 100


def test_pagination_stops_on_empty_page():
    page1 = [{"id": i} for i in range(100)]
    client, _ = make_client([{"access_token": "t"}, page1, []])
    client.login()
    assert len(client.list_sequence_detections(42)) == 100


def test_requests_require_login():
    client, _ = make_client([])
    with pytest.raises(RuntimeError, match="login"):
        client.list_cameras()


def test_list_cameras_and_organizations():
    client, session = make_client(
        [{"access_token": "t"}, [{"id": 1, "name": "cam"}], [{"id": 11, "name": "org"}]]
    )
    client.login()
    assert client.list_cameras() == [{"id": 1, "name": "cam"}]
    assert client.list_organizations() == [{"id": 11, "name": "org"}]
    gets = [c for c in session.calls if c[0] == "GET"]
    assert gets[0][1].endswith("/api/v1/cameras/")
    assert gets[0][2] == {"include_non_trustable": True}


def test_get_sequence_returns_payload_on_200():
    payload = {"id": 99, "started_at": "2026-06-01T10:00:00"}
    client, session = make_client([{"access_token": "t"}, payload])
    client.login()
    result = client.get_sequence(99)
    assert result == payload
    gets = [c for c in session.calls if c[0] == "GET"]
    assert gets[0][1].endswith("/api/v1/sequences/99")


def test_get_sequence_returns_none_on_404():
    client, session = make_client(
        [{"access_token": "t"}, FakeResponse(None, status_code=404)]
    )
    client.login()
    assert client.get_sequence(999) is None


def test_get_sequence_reraises_on_500():
    client, session = make_client(
        [{"access_token": "t"}, FakeResponse(None, status_code=500)]
    )
    client.login()
    with pytest.raises(requests.HTTPError):
        client.get_sequence(1)

import pytest
from fastapi.security import HTTPAuthorizationCredentials

from temporal_model.api.auth import require_token
from temporal_model.api.errors import Unauthorized
from temporal_model.api.settings import settings


def _creds(token):
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)


def test_auth_disabled_when_token_unset(monkeypatch):
    monkeypatch.setattr(settings, "token", None)
    # No exception, returns None, even with no credentials.
    assert require_token(None) is None


def test_correct_token_passes(monkeypatch):
    monkeypatch.setattr(settings, "token", "s3cr3t")
    assert require_token(_creds("s3cr3t")) is None


def test_wrong_token_raises(monkeypatch):
    monkeypatch.setattr(settings, "token", "s3cr3t")
    with pytest.raises(Unauthorized):
        require_token(_creds("nope"))


def test_missing_credentials_raise_when_token_set(monkeypatch):
    monkeypatch.setattr(settings, "token", "s3cr3t")
    with pytest.raises(Unauthorized):
        require_token(None)


def test_non_ascii_token_fails_closed_not_500(monkeypatch):
    # Bearer tokens are ASCII (RFC 6750). A misconfigured non-ASCII token can't
    # authenticate over HTTP anyway — the header arrives latin-1-decoded while
    # the env var is UTF-8, so they never match. The contract we guarantee is
    # fail-closed: a clean 401, never a 500. (compare_digest raises TypeError on
    # non-ASCII *str*; comparing bytes turns that into a normal mismatch.)
    monkeypatch.setattr(settings, "token", "wörd-123")
    wire_credential = "wörd-123".encode().decode("latin-1")  # what arrives
    with pytest.raises(Unauthorized):  # not TypeError
        require_token(_creds(wire_credential))

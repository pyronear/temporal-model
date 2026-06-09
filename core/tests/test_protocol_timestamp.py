"""Pins the unified, anchored timestamp parser."""

from datetime import datetime

from temporal_model.core.protocol import parse_timestamp


def test_parses_pyronear_suffix():
    ts = parse_timestamp("adf_site_999_2023-05-23T17-18-31")
    assert ts == datetime(2023, 5, 23, 17, 18, 31)


def test_returns_none_when_no_timestamp():
    assert parse_timestamp("no_timestamp_here") is None


def test_anchored_rejects_trailing_suffix_after_timestamp():
    # A timestamp NOT at the end of the id must not match (anchored `$`).
    assert parse_timestamp("2023-05-23T17-18-31_extra") is None


def test_returns_none_on_invalid_calendar_value():
    assert parse_timestamp("seq_2023-13-45T99-99-99") is None

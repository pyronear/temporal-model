from pathlib import Path
from unittest.mock import patch

import pytest

from temporal_model.monitor.stack import (
    API_URL,
    BUCKET,
    ReplayStack,
    StackError,
)


def test_compose_commands_pin_the_image_tag():
    stack = ReplayStack(Path("dc.yml"), version="0.3.1")
    with patch("temporal_model.monitor.stack.subprocess.run") as run:
        stack.up()
        stack.down()
    up_call, down_call = run.call_args_list
    assert up_call.args[0] == [
        "docker",
        "compose",
        "-f",
        "dc.yml",
        "-p",
        "temporal-monitor-replay",
        "up",
        "-d",
    ]
    assert (
        up_call.kwargs["env"]["MONITOR_API_IMAGE"]
        == "pyronear/temporal-model-api:0.3.1"
    )
    assert up_call.kwargs["check"] is True
    assert down_call.args[0][-2:] == ["down", "-v"]


def test_wait_healthy_polls_until_model_loaded():
    stack = ReplayStack(Path("dc.yml"), version="0.3.1")
    responses = iter(
        [
            ConnectionError("not up yet"),
            {"status": "ok", "model_loaded": False},
            {"status": "ok", "model_loaded": True, "model_version": "0.1.0"},
        ]
    )

    def fake_health():
        item = next(responses)
        if isinstance(item, Exception):
            raise item
        return item

    with patch.object(stack, "_fetch_health", side_effect=fake_health):
        health = stack.wait_healthy(timeout_s=5, poll_s=0)
    assert health["model_version"] == "0.1.0"


def test_wait_healthy_times_out():
    stack = ReplayStack(Path("dc.yml"), version="0.3.1")
    with (
        patch.object(stack, "_fetch_health", side_effect=ConnectionError),
        pytest.raises(StackError, match="health"),
    ):
        stack.wait_healthy(timeout_s=0.05, poll_s=0.01)


def test_upload_frames_puts_each_key_once(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"a")
    (tmp_path / "b.jpg").write_bytes(b"b")
    stack = ReplayStack(Path("dc.yml"), version="0.3.1")
    with patch.object(stack, "_s3_client") as make_client:
        stack.upload_frames(
            {"cam/k1.jpg": tmp_path / "a.jpg", "cam/k2.jpg": tmp_path / "b.jpg"}
        )
    uploaded = [c.args for c in make_client.return_value.upload_file.call_args_list]
    assert sorted(u[2] for u in uploaded) == ["cam/k1.jpg", "cam/k2.jpg"]
    assert all(u[1] == BUCKET for u in uploaded)


def test_api_url_uses_offset_port():
    assert API_URL == "http://localhost:18000"

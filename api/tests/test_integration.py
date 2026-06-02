"""End-to-end test gated on a real model package.

Set TEMPORAL_API_TEST_MODEL_PATH to a real model.zip to run it; otherwise it is
skipped. This is the only test that loads the actual core model + weights.
"""

import os
from pathlib import Path

import pytest

from temporal_model.api.model_runner import ModelRunner

MODEL_PATH = os.environ.get("TEMPORAL_API_TEST_MODEL_PATH")


@pytest.mark.skipif(not MODEL_PATH, reason="no real model.zip provided")
def test_real_model_loads_and_reports_metadata():
    assert MODEL_PATH is not None  # guaranteed by skipif; narrows for type-checkers
    runner = ModelRunner.load(Path(MODEL_PATH), device="cpu")
    assert runner.name
    assert isinstance(runner.calibrated, bool)

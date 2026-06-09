"""run_core threads allow_uncalibrated into BboxTubeTemporalModel.from_package."""

from pathlib import Path
from unittest.mock import patch

from temporal_model.benchmark import run_core as rc


@patch.object(rc.BboxTubeTemporalModel, "from_package")
def test_run_core_defaults_to_strict(mock_from_package, tmp_path: Path) -> None:
    # iter_sequences returns nothing -> run_core raises SystemExit after load,
    # but we only care that from_package was called strict (allow_uncalibrated=False).
    with patch.object(rc, "iter_sequences", return_value=iter(())):
        try:
            rc.run_core(tmp_path, tmp_path / "m.zip", device="cpu")
        except SystemExit:
            pass
    _, kwargs = mock_from_package.call_args
    assert kwargs.get("allow_uncalibrated", False) is False


@patch.object(rc.BboxTubeTemporalModel, "from_package")
def test_run_core_forwards_opt_in(mock_from_package, tmp_path: Path) -> None:
    with patch.object(rc, "iter_sequences", return_value=iter(())):
        try:
            rc.run_core(
                tmp_path, tmp_path / "m.zip", device="cpu", allow_uncalibrated=True
            )
        except SystemExit:
            pass
    _, kwargs = mock_from_package.call_args
    assert kwargs.get("allow_uncalibrated") is True

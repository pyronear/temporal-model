from temporal_model.triage import cli


def test_pull_command_invokes_pull(monkeypatch, tmp_path):
    calls = {}
    fake_client = object()  # _build_client (which logs in) is covered in Task 2

    monkeypatch.setattr(cli, "_build_client", lambda: fake_client)
    monkeypatch.setattr(
        cli,
        "pull_unannotated",
        lambda client, store_dir, **kw: calls.update(
            client=client, store=store_dir, kw=kw
        )
        or {"pulled": 3, "skipped": 0},
    )
    cli.main(["pull", "--store", str(tmp_path), "--limit", "3"])
    assert calls["client"] is fake_client
    assert calls["kw"]["limit"] == 3
    assert calls["kw"]["processing_stage"] == "ready_to_annotate"
    assert calls["store"] == tmp_path


def test_score_command_threshold_flag_overrides(monkeypatch, tmp_path):
    captured = {}

    monkeypatch.setattr(cli, "_load_model", lambda model_zip, device: object())
    monkeypatch.setattr(
        cli,
        "score_sequences",
        lambda model, store, *, threshold: captured.update(threshold=threshold)
        or ([], []),
    )
    monkeypatch.setattr(cli, "read_model_config", lambda model_zip: {})
    monkeypatch.setattr(
        cli,
        "write_triage_report",
        lambda *a, **k: captured.update(wrote=True),
    )
    cli.main(
        [
            "score",
            "--store",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--model-zip",
            str(tmp_path / "model.zip"),
            "--threshold",
            "0.5",
        ]
    )
    assert captured["threshold"] == 0.5
    assert captured["wrote"] is True

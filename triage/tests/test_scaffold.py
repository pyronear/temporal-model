from temporal_model.triage import __doc__ as pkg_doc


def test_package_importable():
    assert "triage" in pkg_doc.lower()

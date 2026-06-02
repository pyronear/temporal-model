import temporal_model.core
from temporal_model.eval import evaluate


def test_eval_imports():
    assert callable(evaluate.main)


def test_core_dependency_importable():
    assert temporal_model.core is not None

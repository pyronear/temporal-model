import temporal_model.core
from temporal_model.train import train


def test_train_imports():
    assert callable(train.main)


def test_core_dependency_importable():
    assert temporal_model.core is not None

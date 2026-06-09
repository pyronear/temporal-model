import temporal_model.core
from temporal_model.core import (
    crop,
    inference,
    model,
    tubes,
    types,
)


def test_core_subpackage_imports():
    assert temporal_model.core is not None
    assert all(mod is not None for mod in (types, tubes, crop, inference, model))

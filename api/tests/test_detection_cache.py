from temporal_model.api.detection_cache import DetectionCache


def test_put_get_contains():
    c = DetectionCache(capacity=4)
    c.put("a", 1)
    assert "a" in c
    assert c.get("a") == 1
    assert len(c) == 1


def test_evicts_least_recently_used():
    c = DetectionCache(capacity=2)
    c.put("a", 1)
    c.put("b", 2)
    c.put("c", 3)  # evicts "a"
    assert "a" not in c
    assert "b" in c and "c" in c
    assert len(c) == 2


def test_get_marks_recently_used():
    c = DetectionCache(capacity=2)
    c.put("a", 1)
    c.put("b", 2)
    c.get("a")      # "a" now most-recently-used
    c.put("c", 3)   # evicts "b", not "a"
    assert "a" in c
    assert "b" not in c
    assert "c" in c


def test_capacity_zero_disables():
    c = DetectionCache(capacity=0)
    c.put("a", 1)
    assert "a" not in c
    assert len(c) == 0

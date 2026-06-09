"""A small size-bounded LRU for per-frame detections.

Keyed by ``frame_id``. Values are stored opaquely (the caller decides the type).
``capacity <= 0`` disables the cache: nothing is stored, so every lookup misses.
"""

from collections import OrderedDict
from typing import Any


class DetectionCache:
    """Least-recently-used cache with a fixed maximum entry count."""

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._data: OrderedDict[str, Any] = OrderedDict()

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)

    def get(self, key: str) -> Any:
        """Return the value for ``key`` and mark it most-recently-used."""
        self._data.move_to_end(key)
        return self._data[key]

    def put(self, key: str, value: Any) -> None:
        """Insert/update ``key``; evict the LRU entry if over capacity."""
        if self._capacity <= 0:
            return
        self._data[key] = value
        self._data.move_to_end(key)
        while len(self._data) > self._capacity:
            self._data.popitem(last=False)

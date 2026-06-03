"""Package-time fitter for :class:`LogisticCalibrator`.

Imports sklearn at module top. Only the package stage imports this module; the
runtime inference path stays sklearn-free.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from temporal_model.core.logistic_calibrator import (
    FEATURE_NAMES,
    LogisticCalibrator,
    extract_features,
)

_EMPTY_FEATURES = np.zeros(len(FEATURE_NAMES), dtype=float)
_N_SANITY_CHECKS = 3
_PARITY_ATOL = 1e-6


def _features_for_record(record: dict) -> np.ndarray:
    kept = record["kept_tubes"]
    if not kept:
        return _EMPTY_FEATURES.copy()
    best = max(kept, key=lambda t: t["logit"])
    return extract_features(best, n_tubes=len(kept))


def fit(records: list[dict]) -> LogisticCalibrator:
    """Fit a :class:`LogisticCalibrator` on ``predictions``-style records.

    Each record carries ``label`` (``"smoke"``/``"fp"``) and ``kept_tubes``.
    Empty ``kept_tubes`` yield an all-zero feature row.

    Raises:
        ValueError: if the record set contains a single class.
        RuntimeError: if the numpy / sklearn parity check fails.
    """
    X = np.stack([_features_for_record(r) for r in records])
    y = np.array([1 if r["label"] == "smoke" else 0 for r in records], dtype=int)

    model = LogisticRegression(max_iter=1000, C=1.0)
    model.fit(X, y)

    coefficients = np.asarray(model.coef_[0], dtype=float)
    intercept = float(model.intercept_[0])

    cal = LogisticCalibrator(
        features=list(FEATURE_NAMES),
        coefficients=coefficients,
        intercept=intercept,
        sanity_checks=[],
    )

    sklearn_probs = model.predict_proba(X)[:, 1]
    numpy_probs = cal.predict_proba_batch(X)
    if not np.allclose(sklearn_probs, numpy_probs, atol=_PARITY_ATOL):
        max_diff = float(np.max(np.abs(sklearn_probs - numpy_probs)))
        raise RuntimeError(
            "sklearn / numpy logistic-regression parity failed "
            f"(max |Δp|={max_diff:.3e} > atol={_PARITY_ATOL:.3e})"
        )

    order = np.argsort(sklearn_probs)
    pick = order[np.linspace(0, len(order) - 1, num=_N_SANITY_CHECKS, dtype=int)]
    sanity_checks = [
        {"features": [float(v) for v in X[i]], "prob": float(sklearn_probs[i])}
        for i in pick
    ]

    return LogisticCalibrator(
        features=cal.features,
        coefficients=cal.coefficients,
        intercept=cal.intercept,
        sanity_checks=sanity_checks,
    )


__all__ = ["fit"]

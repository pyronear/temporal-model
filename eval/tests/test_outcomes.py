import pandas as pd

from temporal_model.eval.outcomes import (
    apply_threshold,
    compute_outcome,
    decision_from_output,
    filter_results,
    max_probability,
    performance_summary,
)


def test_decision_from_output():
    assert decision_from_output(True) == "keep"
    assert decision_from_output(False) == "discard"


def test_compute_outcome_matrix():
    assert compute_outcome("keep", "smoke") == "kept-smoke"
    assert compute_outcome("discard", "smoke") == "discarded-smoke"
    assert compute_outcome("keep", "fp") == "kept-fp"
    assert compute_outcome("discard", "fp") == "discarded-fp"
    assert compute_outcome("keep", "unknown") == "n/a"


def test_max_probability():
    details = {"tubes": {"kept": [{"probability": 0.2}, {"probability": 0.8}]}}
    assert max_probability(details) == 0.8
    assert max_probability({"tubes": {"kept": []}}) is None
    assert max_probability(None) is None


def test_filter_results_errors_only():
    df = pd.DataFrame(
        {"outcome": ["kept-smoke", "kept-fp", "discarded-smoke", "discarded-fp"]}
    )
    out = filter_results(df, errors_only=True)
    assert set(out["outcome"]) == {"kept-fp", "discarded-smoke"}


def test_performance_summary_counts_and_rates():
    df = pd.DataFrame(
        {
            "label": ["smoke", "smoke", "fp", "fp"],
            "outcome": ["kept-smoke", "discarded-smoke", "discarded-fp", "kept-fp"],
        }
    )
    s = performance_summary(df)
    assert s["n_smoke"] == 2 and s["n_fp"] == 2
    assert s["recall"] == 0.5
    assert s["specificity"] == 0.5
    assert s["precision"] == 0.5


def test_apply_threshold_redecides_and_recomputes_outcome():
    df = pd.DataFrame(
        {
            "label": ["smoke", "smoke", "fp", "fp", "smoke"],
            "probability": [0.9, 0.2, 0.8, 0.1, None],
            "decision": ["keep", "keep", "keep", "discard", "keep"],
            "outcome": [
                "kept-smoke",
                "kept-smoke",
                "kept-fp",
                "discarded-fp",
                "kept-smoke",
            ],
            "score": [5.0, 1.0, 4.0, 0.5, 2.0],
        }
    )
    out = apply_threshold(df, 0.5)
    assert list(out["decision"]) == ["keep", "discard", "keep", "discard", "discard"]
    assert list(out["outcome"]) == [
        "kept-smoke",
        "discarded-smoke",
        "kept-fp",
        "discarded-fp",
        "discarded-smoke",  # probability None -> discard
    ]
    # raising the threshold flips the 0.8 fp from kept-fp to discarded-fp
    out2 = apply_threshold(df, 0.85)
    assert out2.loc[2, "decision"] == "discard"
    assert out2.loc[2, "outcome"] == "discarded-fp"
    # untouched columns preserved; input not mutated
    assert list(out["score"]) == [5.0, 1.0, 4.0, 0.5, 2.0]
    assert list(df["decision"]) == ["keep", "keep", "keep", "discard", "keep"]

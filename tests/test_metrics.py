from __future__ import annotations

import pytest

from text2sql.metrics import Prediction, majority_vote, percentile, summarise


def make(correct: bool, difficulty: str = "simple", **kwargs) -> Prediction:
    return Prediction(
        question_id="q", db_id="shop", question="?", gold_sql="SELECT 1",
        pred_sql="SELECT 1", correct=correct, difficulty=difficulty, **kwargs
    )


def test_summary_counts_and_breakdowns():
    preds = [
        make(True, "simple"), make(False, "simple"),
        make(True, "moderate"), make(False, "challenging"),
    ]
    summary = summarise(preds)
    assert summary["n"] == 4
    assert summary["correct"] == 2
    assert summary["ex"] == 50.0
    assert summary["by_difficulty"]["simple"]["ex"] == 50.0
    assert summary["by_difficulty"]["moderate"]["ex"] == 100.0
    assert summary["by_difficulty"]["challenging"]["ex"] == 0.0


def test_summary_of_nothing_does_not_divide_by_zero():
    assert summarise([])["ex"] == 0.0


def test_latency_and_token_rollups():
    preds = [
        make(True, latency_ms=100.0, input_tokens=1000, output_tokens=50),
        make(True, latency_ms=300.0, input_tokens=2000, output_tokens=150),
    ]
    summary = summarise(preds)
    assert summary["latency_ms"]["p50"] == 200.0
    assert summary["tokens"]["input_total"] == 3000
    assert summary["tokens"]["output_mean"] == 100.0


def test_percentile():
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert percentile(values, 50) == 5.5
    assert percentile(values, 100) == 10
    assert percentile([42], 95) == 42
    with pytest.raises(ValueError):
        percentile([], 50)


def test_majority_vote_picks_the_most_common_result_set():
    # Two differently-written but equivalent queries outvote one odd one out.
    candidates = [
        ("SELECT a FROM t", ((1,), (2,))),
        ("SELECT t.a FROM t AS t", ((2,), (1,))),   # same rows, other order
        ("SELECT b FROM t", ((9,),)),
    ]
    assert majority_vote(candidates) in {"SELECT a FROM t", "SELECT t.a FROM t AS t"}


def test_majority_vote_ignores_queries_that_did_not_run():
    candidates = [("BROKEN", None), ("SELECT 1", ((1,),))]
    assert majority_vote(candidates) == "SELECT 1"


def test_majority_vote_with_nothing_executable():
    assert majority_vote([("BROKEN", None)]) == "BROKEN"

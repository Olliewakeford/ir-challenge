"""Tests for irchallenge/metrics.py against hand-computed values."""
import math

from irchallenge.metrics import (
    average_precision,
    evaluate,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_recall_at_k():
    ranked = ["d1", "d2", "d3", "d4"]
    relevant = {"d2", "d4", "d5"}  # d5 is never retrieved
    assert recall_at_k(ranked, relevant, k=4) == 2 / 3
    assert recall_at_k(ranked, relevant, k=1) == 0.0


def test_recall_at_k_no_relevant_docs_is_zero():
    assert recall_at_k(["d1"], set(), k=10) == 0.0


def test_precision_at_k():
    ranked = ["d1", "d2", "d3"]
    relevant = {"d1", "d3"}
    assert precision_at_k(ranked, relevant, k=3) == 2 / 3
    assert precision_at_k(ranked, relevant, k=1) == 1.0


def test_mrr_at_k_first_hit_position():
    ranked = ["d1", "d2", "d3"]
    assert mrr_at_k(ranked, {"d1"}, k=3) == 1.0
    assert mrr_at_k(ranked, {"d2"}, k=3) == 1 / 2
    assert mrr_at_k(ranked, {"d3"}, k=3) == 1 / 3
    assert mrr_at_k(ranked, {"dX"}, k=3) == 0.0


def test_ndcg_at_k_matches_hand_computed_value():
    # relevant at position 1 and 3 (1-indexed): DCG = 1/log2(2) + 1/log2(4)
    ranked = ["d1", "d2", "d3"]
    relevant = {"d1", "d3"}
    dcg = 1.0 / math.log2(2) + 1.0 / math.log2(4)
    idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)  # ideal: both relevant docs first
    expected = dcg / idcg
    assert ndcg_at_k(ranked, relevant, k=3) == expected


def test_ndcg_at_k_perfect_ranking_is_one():
    ranked = ["d1", "d2"]
    relevant = {"d1", "d2"}
    assert ndcg_at_k(ranked, relevant, k=2) == 1.0


def test_ndcg_at_k_no_relevant_docs_is_zero():
    assert ndcg_at_k(["d1"], set(), k=10) == 0.0


def test_average_precision():
    # relevant at rank 1 and rank 3: AP = (1/1 + 2/3) / 2
    ranked = ["d1", "d2", "d3"]
    relevant = {"d1", "d3"}
    expected = (1 / 1 + 2 / 3) / 2
    assert average_precision(ranked, relevant) == expected


def test_evaluate_aggregates_per_query_metrics():
    submission = {"q1": ["d1", "d2"], "q2": ["d3", "d4"]}
    qrels = {"q1": ["d1"], "q2": ["d5"]}  # q2 has no hits at all
    result = evaluate(submission, qrels, ks=[1], verbose=False)
    assert result["overall"]["num_queries"] == 2
    # q1 hits at rank 1, q2 never hits -> mean Recall@1 = (1 + 0) / 2
    assert result["overall"]["Recall@1"] == 0.5
    assert result["per_query"]["q1"]["Recall@1"] == 1.0
    assert result["per_query"]["q2"]["Recall@1"] == 0.0

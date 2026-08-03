"""Tests for irchallenge/filters.py: the year filter.

Disabled in the submitted pipeline (see README "what didn't work": it drops
NDCG@10 from 0.6151 to 0.3782 on this dataset). Still covered here since it's
pure and still used for reproducing that finding.
"""
import pandas as pd

from irchallenge.config import PipelineConfig
from irchallenge.filters import apply_year_filter


def _df(rows):
    return pd.DataFrame(rows)


def test_removes_candidates_published_after_the_query():
    queries_df = _df([{"doc_id": "q1", "year": 2020}])
    corpus_df = _df([
        {"doc_id": "d1", "year": 2018},  # kept: before query
        {"doc_id": "d2", "year": 2020},  # kept: same year
        {"doc_id": "d3", "year": 2022},  # removed: after query
    ])
    config = PipelineConfig(apply_year_filter=True)
    result = apply_year_filter({"q1": ["d1", "d2", "d3"]}, queries_df, corpus_df, config)
    assert result["q1"] == ["d1", "d2"]


def test_disabled_filter_is_a_no_op():
    queries_df = _df([{"doc_id": "q1", "year": 2020}])
    corpus_df = _df([{"doc_id": "d1", "year": 2099}])
    config = PipelineConfig(apply_year_filter=False)
    ranked_lists = {"q1": ["d1"]}
    result = apply_year_filter(ranked_lists, queries_df, corpus_df, config)
    assert result == ranked_lists
    assert result is ranked_lists  # returns the same object unmodified, no filtering attempted


def test_unknown_candidate_year_defaults_to_zero_and_is_kept():
    # a candidate missing from corpus_years defaults to year 0, which is
    # always <= the query year, so it's kept rather than dropped.
    queries_df = _df([{"doc_id": "q1", "year": 2020}])
    corpus_df = _df([{"doc_id": "d1", "year": 2018}])  # "d2" is absent entirely
    config = PipelineConfig(apply_year_filter=True)
    result = apply_year_filter({"q1": ["d1", "d2"]}, queries_df, corpus_df, config)
    assert result["q1"] == ["d1", "d2"]

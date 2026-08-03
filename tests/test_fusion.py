"""Tests for irchallenge/fusion.py.

weighted_rrf_fuse is the shared implementation that replaced four
near-identical copies of this loop across scripts/run_refusion_v4.py,
scripts/run_domain_boost_tune.py, scripts/run_new_signals.py, and
scripts/generate_submission_v4.py. Same four properties as fusion_lib,
since it's the same algorithm.
"""
from irchallenge.config import PipelineConfig
from irchallenge.fusion import fuse_rrf, weighted_rrf_fuse


def test_weighted_rrf_fuse_matches_hand_computed_scores():
    # k=0: docX = 2.0/1 (sigA rank0) + 1.0/2 (sigB rank1) = 2.5
    #      docY = 2.0/2 (sigA rank1) + 1.0/1 (sigB rank0) = 2.0
    signals = {
        "sigA": {"q1": ["docX", "docY"]},
        "sigB": {"q1": ["docY", "docX"]},
    }
    result = weighted_rrf_fuse(signals, {"sigA": 2.0, "sigB": 1.0}, k=0, top_n=10)
    assert result["q1"] == ["docX", "docY"]


def test_weighted_rrf_fuse_weights_reorder_results():
    signals = {
        "sigA": {"q1": ["docA", "docB"]},
        "sigB": {"q1": ["docB", "docA"]},
    }
    a_favoured = weighted_rrf_fuse(signals, {"sigA": 5.0, "sigB": 0.1}, k=0, top_n=10)
    b_favoured = weighted_rrf_fuse(signals, {"sigA": 0.1, "sigB": 5.0}, k=0, top_n=10)
    assert a_favoured["q1"][0] == "docA"
    assert b_favoured["q1"][0] == "docB"
    assert a_favoured["q1"] != b_favoured["q1"]


def test_weighted_rrf_fuse_zero_weight_drops_signal_entirely():
    signals = {
        "good": {"q1": ["d1", "d2"]},
        "noise": {"q1": ["d3", "d4"]},
    }
    result = weighted_rrf_fuse(signals, {"good": 1.0, "noise": 0.0}, k=0, top_n=100)
    assert result["q1"] == ["d1", "d2"]
    assert "d3" not in result["q1"]


def test_weighted_rrf_fuse_missing_weight_key_is_zero():
    # unlike fuse_rrf(config) below, weighted_rrf_fuse's own default for an
    # unlisted signal is 0.0, matching the four scripts it replaced.
    signals = {"a": {"q1": ["d1"]}, "b": {"q1": ["d2"]}}
    result = weighted_rrf_fuse(signals, {"a": 1.0}, k=0, top_n=100)
    assert result["q1"] == ["d1"]


def test_weighted_rrf_fuse_ties_are_deterministic():
    signals = {
        "sigA": {"q1": ["d1", "d2", "d3"]},
        "sigB": {"q1": ["d3", "d2", "d1"]},
    }
    result = weighted_rrf_fuse(signals, {"sigA": 1.0, "sigB": 1.0}, k=1, top_n=10)
    assert result["q1"] == ["d1", "d3", "d2"]
    assert weighted_rrf_fuse(signals, {"sigA": 1.0, "sigB": 1.0}, k=1, top_n=10)["q1"] == result["q1"]


def test_fuse_rrf_uses_config_weights_with_default_one():
    # fuse_rrf(all_retrievers, config) is the pipeline's own Stage 1 fusion:
    # unlike weighted_rrf_fuse, a signal missing from config.rrf_weights
    # defaults to weight 1.0, not 0.0.
    all_retrievers = {"a": {"q1": ["d1", "d2"]}, "b": {"q1": ["d2", "d1"]}}
    config = PipelineConfig(rrf_k=0, rrf_weights={"a": 1.0}, fusion_top_k=10)  # "b" unset -> defaults to 1.0
    result = fuse_rrf(all_retrievers, config)
    # d1 = 1.0/1 (a rank0) + 1.0/2 (b rank1, default weight 1.0) = 1.5
    # d2 = 1.0/2 (a rank1) + 1.0/1 (b rank0, default weight 1.0) = 1.5 -> tie, a's d1 inserted first
    assert result["q1"] == ["d1", "d2"]

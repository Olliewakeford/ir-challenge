"""Tests for scripts/fusion_lib.py: weighted Reciprocal Rank Fusion.

fusion_lib.rrf() scores each document as sum(weight / (k + rank + 1)) over
every signal that ranks it, then sorts descending. These tests check that
against hand-computed numbers, rather than re-deriving the formula and
comparing a function to itself.
"""
from fusion_lib import combsum, nested_rrf, rank_scores, rrf


def test_rrf_matches_hand_computed_scores():
    # sigA ranks docX first, sigB ranks docY first, weight sigA=2x sigB=1.
    # k=0 so the formula is just weight / (rank + 1):
    #   docX = 2.0/1 (sigA rank0) + 1.0/2 (sigB rank1) = 2.0 + 0.5 = 2.5
    #   docY = 2.0/2 (sigA rank1) + 1.0/1 (sigB rank0) = 1.0 + 1.0 = 2.0
    # docX should come out on top.
    signals = {
        "sigA": {"q1": ["docX", "docY"]},
        "sigB": {"q1": ["docY", "docX"]},
    }
    result = rrf(signals, weights={"sigA": 2.0, "sigB": 1.0}, k=0, top_n=10)
    assert result["q1"] == ["docX", "docY"]


def test_rrf_matches_hand_computed_scores_second_fixture():
    # Single signal, k=6 (the value used in the submitted pipeline):
    #   d1 rank0 -> 1/(6+0+1) = 1/7
    #   d2 rank1 -> 1/(6+1+1) = 1/8
    # 1/7 > 1/8, so d1 stays ahead of d2 with a single signal at any k.
    signals = {"only": {"q1": ["d1", "d2"]}}
    result = rrf(signals, weights={"only": 1.0}, k=6, top_n=10)
    assert result["q1"] == ["d1", "d2"]


def test_weights_reorder_results():
    # Two signals disagree on which of two docs comes first. With equal
    # weights they tie (see test_ties below); skewing the weight toward
    # whichever signal ranks docB first should flip the fused order.
    signals = {
        "sigA": {"q1": ["docA", "docB"]},  # sigA wants docA on top
        "sigB": {"q1": ["docB", "docA"]},  # sigB wants docB on top
    }
    equal_weights = rrf(signals, weights={"sigA": 1.0, "sigB": 1.0}, k=0, top_n=10)
    assert equal_weights["q1"][0] == "docA"  # tie broken by insertion order (sigA first)

    skewed = rrf(signals, weights={"sigA": 0.1, "sigB": 5.0}, k=0, top_n=10)
    assert skewed["q1"][0] == "docB"
    assert skewed["q1"] != equal_weights["q1"]


def test_zero_weight_drops_signal_entirely():
    # A zero-weighted signal must not just be deprioritised: its candidates
    # should never appear in the fused output at all, even if there aren't
    # enough other candidates to push them out on score alone.
    signals = {
        "good": {"q1": ["d1", "d2"]},
        "noise": {"q1": ["d3", "d4"]},  # completely disjoint doc ids
    }
    result = rrf(signals, weights={"good": 1.0, "noise": 0.0}, k=0, top_n=100)
    assert result["q1"] == ["d1", "d2"]
    assert "d3" not in result["q1"]
    assert "d4" not in result["q1"]


def test_missing_weight_key_defaults_to_zero():
    # A signal simply absent from the weights dict behaves the same as an
    # explicit weight of 0.0 (the pipeline's own fuse_rrf uses this to mean
    # "signal not in this run" for signals with no cached ranked list).
    signals = {"a": {"q1": ["d1"]}, "b": {"q1": ["d2"]}}
    result = rrf(signals, weights={"a": 1.0}, k=0, top_n=100)
    assert result["q1"] == ["d1"]


def test_ties_are_broken_deterministically():
    # Symmetric fixture: two signals rank the same three docs in opposite
    # order, so d1 and d3 end up with an exact score tie (each is first in
    # one signal, last in the other). A stable sort must break the tie by
    # insertion order into the running score dict, which follows the
    # dict-iteration order of `signals` and then rank within each list:
    # d1 is inserted first (sigA rank 0), then d2, then d3 — so among ties,
    # d1 must precede d3.
    signals = {
        "sigA": {"q1": ["d1", "d2", "d3"]},
        "sigB": {"q1": ["d3", "d2", "d1"]},
    }
    result = rrf(signals, weights={"sigA": 1.0, "sigB": 1.0}, k=1, top_n=10)
    assert result["q1"] == ["d1", "d3", "d2"]

    # Re-running must give the exact same order — no reliance on dict/set
    # iteration randomness anywhere in the implementation.
    again = rrf(signals, weights={"sigA": 1.0, "sigB": 1.0}, k=1, top_n=10)
    assert again["q1"] == result["q1"]


def test_rrf_respects_top_n():
    signals = {"a": {"q1": ["d1", "d2", "d3", "d4"]}}
    result = rrf(signals, weights={"a": 1.0}, k=0, top_n=2)
    assert result["q1"] == ["d1", "d2"]


def test_rrf_default_weight_is_one():
    signals = {"a": {"q1": ["d1", "d2"]}, "b": {"q1": ["d2", "d1"]}}
    with_weights = rrf(signals, weights={"a": 1.0, "b": 1.0}, k=0, top_n=10)
    without_weights = rrf(signals, k=0, top_n=10)  # weights=None -> defaults to 1.0 each
    assert with_weights == without_weights


def test_rank_scores_formula():
    scores = rank_scores(["d1", "d2", "d3"], k=1)
    assert scores == {"d1": 1 / 2, "d2": 1 / 3, "d3": 1 / 4}


def test_combsum_weighted_sum():
    # CombSUM is a plain weighted sum of pre-computed scores, no rank involved.
    score_lists = {
        "sigA": {"q1": {"d1": 0.8, "d2": 0.2}},
        "sigB": {"q1": {"d1": 0.1, "d2": 0.9}},
    }
    result = combsum(score_lists, weights={"sigA": 2.0, "sigB": 1.0}, top_n=10)
    # d1 = 2.0*0.8 + 1.0*0.1 = 1.7 ; d2 = 2.0*0.2 + 1.0*0.9 = 1.3
    assert result["q1"] == ["d1", "d2"]


def test_combsum_zero_weight_drops_signal():
    score_lists = {
        "good": {"q1": {"d1": 0.5}},
        "noise": {"q1": {"d2": 0.9}},
    }
    result = combsum(score_lists, weights={"good": 1.0, "noise": 0.0}, top_n=10)
    assert result["q1"] == ["d1"]


def test_nested_rrf_combines_inner_and_outer_signals():
    # tight_a and tight_b agree strongly that d1 beats d2; loose disagrees.
    # With the inner consensus weighted 3x in the outer fusion, it should
    # still win: d1 = 0.3333 (from loose) + 1.5 (3x inner) = 1.8333, versus
    # d2 = 0.5 (from loose) + 1.0 (3x inner) = 1.5.
    signals = {
        "tight_a": {"q1": ["d1", "d2"]},
        "tight_b": {"q1": ["d1", "d2"]},
        "loose": {"q1": ["d2", "d1"]},
    }
    result = nested_rrf(
        signals,
        inner_names=["tight_a", "tight_b"],
        outer_names=["loose"],
        outer_weights={"loose": 1.0, "__inner__": 3.0},
        k_inner=1, k_outer=1,
        top_n=10,
    )
    assert result["q1"] == ["d1", "d2"]

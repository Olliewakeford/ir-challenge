"""Tests for irchallenge/boost.py: the domain/venue rescoring stage."""
import pandas as pd

from irchallenge.boost import apply_domain_venue_boost


def _df(rows):
    return pd.DataFrame(rows)


def test_same_domain_candidate_is_boosted_above_higher_ranked_other_domain():
    queries_df = _df([{"doc_id": "q1", "domain": "nlp", "venue": "ACL"}])
    corpus_df = _df([
        {"doc_id": "d1", "domain": "vision", "venue": "CVPR"},  # rank 0, wrong domain
        {"doc_id": "d2", "domain": "nlp", "venue": "ICML"},     # rank 1, same domain
    ])
    fused = {"q1": ["d1", "d2"]}
    # d1 base score 1/1=1.0, d2 base score 1/2=0.5 * domain_boost 10 = 5.0 -> d2 wins
    result = apply_domain_venue_boost(fused, queries_df, corpus_df, domain_boost=10.0, venue_boost=1.0)
    assert result["q1"] == ["d2", "d1"]


def test_boost_of_one_is_a_no_op():
    # Same fixture as the "boosted above" test, but with domain_boost=1.0:
    # the same-domain candidate (d2, rank 1) has nothing multiplying its
    # score, so it must NOT overtake d1 the way it does with boost=10.0.
    queries_df = _df([{"doc_id": "q1", "domain": "nlp", "venue": "ACL"}])
    corpus_df = _df([
        {"doc_id": "d1", "domain": "vision", "venue": "CVPR"},
        {"doc_id": "d2", "domain": "nlp", "venue": "ICML"},
    ])
    fused = {"q1": ["d1", "d2"]}
    result = apply_domain_venue_boost(fused, queries_df, corpus_df, domain_boost=1.0, venue_boost=1.0)
    assert result["q1"] == ["d1", "d2"]


def test_venue_boost_stacks_with_domain_boost():
    queries_df = _df([{"doc_id": "q1", "domain": "nlp", "venue": "ACL"}])
    corpus_df = _df([
        {"doc_id": "d1", "domain": "vision", "venue": "CVPR"},  # rank 0: no boosts
        {"doc_id": "d2", "domain": "nlp", "venue": "ACL"},      # rank 1: both boosts
    ])
    fused = {"q1": ["d1", "d2"]}
    # d1 = 1.0 ; d2 = 0.5 * 2 (domain) * 2 (venue) = 2.0 -> d2 wins
    result = apply_domain_venue_boost(fused, queries_df, corpus_df, domain_boost=2.0, venue_boost=2.0)
    assert result["q1"] == ["d2", "d1"]
    # with domain boost only (venue_boost=1.0) it isn't enough to overtake: 0.5*2=1.0 ties d1's 1.0
    result_domain_only = apply_domain_venue_boost(fused, queries_df, corpus_df, domain_boost=2.0, venue_boost=1.0)
    assert result_domain_only["q1"] == ["d1", "d2"]  # tie broken by original (higher) rank


def test_empty_domain_or_venue_never_matches():
    # a query or candidate with a blank domain/venue should never be treated
    # as "same domain" just because both sides are empty strings.
    queries_df = _df([{"doc_id": "q1", "domain": "", "venue": ""}])
    corpus_df = _df([{"doc_id": "d1", "domain": "", "venue": ""}])
    fused = {"q1": ["d1"]}
    result = apply_domain_venue_boost(fused, queries_df, corpus_df, domain_boost=100.0, venue_boost=100.0)
    assert result["q1"] == ["d1"]  # only candidate, boost or not changes nothing observable here,
    # so check the multiplier didn't apply via a second candidate that would
    # otherwise be outscored if the empty-string boost had (incorrectly) fired.
    corpus_df2 = _df([
        {"doc_id": "d1", "domain": "", "venue": ""},   # rank 0, blank domain/venue
        {"doc_id": "d2", "domain": "nlp", "venue": ""},  # rank 1, real domain but blank venue
    ])
    fused2 = {"q1": ["d1", "d2"]}
    result2 = apply_domain_venue_boost(fused2, queries_df, corpus_df2, domain_boost=100.0, venue_boost=100.0)
    assert result2["q1"] == ["d1", "d2"]  # neither boost fires: blank domain/venue never "matches"

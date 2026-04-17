#!/usr/bin/env python3
"""
Shared fusion library: weighted RRF, nested RRF, CombSUM with Platt-calibrated scores.

Functions:
  rrf(ranked_lists, weights=None, k=60, top_n=300) -> {qid: [doc_ids]}
  nested_rrf(signals, inner_names, outer_names, k_inner=1, k_outer=2, top_n=300)
  combsum(score_lists, weights=None, top_n=300)
"""
from collections import defaultdict
from typing import Optional


def rrf(ranked_lists_per_signal: dict, weights: Optional[dict] = None,
        k: int = 60, top_n: int = 300) -> dict:
    """
    ranked_lists_per_signal: {signal_name: {qid: [doc_ids ranked]}}
    weights: {signal_name: float} (default: 1.0 each)

    Returns {qid: [top_n doc_ids after weighted RRF]}.
    """
    if weights is None:
        weights = {name: 1.0 for name in ranked_lists_per_signal}

    qids = set()
    for s in ranked_lists_per_signal.values():
        qids |= set(s.keys())

    out = {}
    for qid in qids:
        score = defaultdict(float)
        for name, per_q in ranked_lists_per_signal.items():
            if qid not in per_q:
                continue
            w = weights.get(name, 0.0)
            if w == 0:
                continue
            for rank, doc_id in enumerate(per_q[qid]):
                score[doc_id] += w / (k + rank + 1)
        ranked = sorted(score.items(), key=lambda x: -x[1])
        out[qid] = [d for d, _ in ranked[:top_n]]
    return out


def nested_rrf(signals_per_qid: dict,
               inner_names: list[str],
               outer_names: list[str],
               inner_weights: Optional[dict] = None,
               outer_weights: Optional[dict] = None,
               k_inner: int = 1,
               k_outer: int = 2,
               top_n: int = 300,
               inner_top_n: int = 300) -> dict:
    """
    Nested RRF:
      - Compute `inner` RRF over `inner_names` with k=k_inner (tight consensus).
      - Include `inner` as a special signal named '__inner__' in the outer RRF,
        along with `outer_names`, with k=k_outer (loose fusion).

    inner_weights / outer_weights: per-signal weights, default 1.0
    """
    inner = rrf(
        {n: signals_per_qid[n] for n in inner_names if n in signals_per_qid},
        weights=inner_weights,
        k=k_inner, top_n=inner_top_n,
    )
    outer_signals = {n: signals_per_qid[n] for n in outer_names if n in signals_per_qid}
    outer_signals["__inner__"] = inner
    ow = dict(outer_weights) if outer_weights else {n: 1.0 for n in outer_names}
    ow.setdefault("__inner__", 1.0)
    return rrf(outer_signals, weights=ow, k=k_outer, top_n=top_n)


def rank_scores(ranked_list: list, k: int = 60) -> dict:
    """Rank->score using 1/(k+rank+1). For use in learned feature vectors."""
    return {d: 1.0 / (k + i + 1) for i, d in enumerate(ranked_list)}


def combsum(score_lists_per_signal: dict, weights: Optional[dict] = None,
            top_n: int = 300) -> dict:
    """
    score_lists_per_signal: {signal: {qid: {doc_id: score}}}
    CombSUM: final_score = sum_signal weight * score
    """
    if weights is None:
        weights = {name: 1.0 for name in score_lists_per_signal}
    qids = set()
    for s in score_lists_per_signal.values():
        qids |= set(s.keys())
    out = {}
    for qid in qids:
        agg = defaultdict(float)
        for name, per_q in score_lists_per_signal.items():
            if qid not in per_q:
                continue
            w = weights.get(name, 0.0)
            if w == 0:
                continue
            for doc_id, s in per_q[qid].items():
                agg[doc_id] += w * s
        ranked = sorted(agg.items(), key=lambda x: -x[1])
        out[qid] = [d for d, _ in ranked[:top_n]]
    return out

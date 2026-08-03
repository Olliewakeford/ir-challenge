"""Weighted Reciprocal Rank Fusion across retrieval signals.

Two entry points, differing only in how they source weights and how they treat
a zero weight. `weighted_rrf_fuse` takes weights explicitly, defaults a missing
signal to 0.0 and skips it entirely; it is what the tuning scripts
(run_refusion_v4.py, run_domain_boost_tune.py, run_new_signals.py,
generate_submission_v4.py) call. `fuse_rrf` and `fuse_rrf_with_scores` read a
PipelineConfig, default a missing signal to 1.0 and do not skip zero-weighted
ones, which is what the pipeline's own Stage 1 expects.

That difference is load-bearing, so the two behaviours are kept distinct. The
scoring loop itself is shared via `_rrf_scores`.
"""
from collections import defaultdict
from itertools import product


def _rrf_scores(all_retrievers: dict, weight_of, k: int, skip_zero: bool) -> dict:
    """
    Accumulate weighted RRF contributions across every signal.

    weight_of: callable mapping a signal name to its weight.
    skip_zero: when True, a zero-weighted signal contributes nothing at all.
        When False it still contributes 0.0 to each doc it ranked, which keeps
        those docs present in the score map even though they gain nothing.

    Returns {qid: {doc_id: score}}.
    """
    all_qids = set()
    for rl in all_retrievers.values():
        all_qids.update(rl.keys())

    per_query = {}
    for qid in all_qids:
        scores = defaultdict(float)
        for name, ranked_lists in all_retrievers.items():
            w = weight_of(name)
            if skip_zero and w == 0:
                continue
            for rank, doc_id in enumerate(ranked_lists.get(qid, [])):
                scores[doc_id] += w / (k + rank + 1)
        per_query[qid] = scores
    return per_query


def _top_docs(scores: dict, limit: int) -> list:
    """Docs sorted by descending score, truncated to `limit`."""
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]


def weighted_rrf_fuse(all_retrievers: dict, weights: dict, k: int, top_n: int = 300) -> dict:
    """
    all_retrievers: {signal_name: {qid: [doc_ids ranked]}}
    weights: {signal_name: float}, missing/zero-weighted signals are skipped.

    Returns {qid: [top_n doc_ids after weighted RRF]}.
    """
    per_query = _rrf_scores(
        all_retrievers, lambda name: weights.get(name, 0.0), k, skip_zero=True
    )
    return {
        qid: [doc_id for doc_id, _ in _top_docs(scores, top_n)]
        for qid, scores in per_query.items()
    }


def fuse_rrf(all_retrievers: dict, config) -> dict:
    """Weighted RRF fusion driven by a PipelineConfig (default weight 1.0)."""
    print(f"\n  [RRF] Fusing {len(all_retrievers)} retrievers with k={config.rrf_k}")
    print(f"  Weights: {config.rrf_weights}")

    fused, _ = fuse_rrf_with_scores(all_retrievers, config)

    print(f"  [RRF] Fused -> {len(fused)} queries, top-{config.fusion_top_k} each")
    return fused


def fuse_rrf_with_scores(all_retrievers: dict, config) -> tuple:
    """RRF fusion that also returns per-doc scores for downstream use."""
    per_query = _rrf_scores(
        all_retrievers,
        lambda name: config.rrf_weights.get(name, 1.0),
        config.rrf_k,
        skip_zero=False,
    )

    fused = {}
    fused_scores = {}  # {qid: {doc_id: score}}
    for qid, scores in per_query.items():
        top = _top_docs(scores, config.fusion_top_k)
        fused[qid] = [doc_id for doc_id, _ in top]
        fused_scores[qid] = dict(top)

    return fused, fused_scores


def tune_rrf_weights(all_retrievers, qrels, queries_df, corpus_df):
    """Grid search over RRF weights using public qrels."""
    from .metrics import evaluate

    print("\n" + "=" * 68)
    print("TUNING RRF WEIGHTS")
    print("=" * 68)

    query_years = dict(zip(queries_df["doc_id"], queries_df["year"]))
    corpus_years = dict(zip(corpus_df["doc_id"], corpus_df["year"]))

    best_score = 0
    best_params = {}

    # Coarse grid
    weight_grid = {
        "specter2": [1.0, 1.5, 2.0, 2.5],
        "scincl": [0.5, 1.0, 1.5],
        "minilm": [0.3, 0.5, 0.7, 1.0],
        "bm25_ta": [0.5, 1.0, 1.5],
        "bm25_ft": [0.3, 0.5, 0.8, 1.0],
        "bm25_sections": [0.5, 0.8, 1.0, 1.2],
    }
    k_values = [40, 50, 60, 80]

    # Only include retrievers we actually have
    active_retrievers = {k: v for k, v in all_retrievers.items() if v}
    active_grid = {k: v for k, v in weight_grid.items() if k in active_retrievers}

    # Generate all combinations
    keys = sorted(active_grid.keys())
    all_combos = list(product(*[active_grid[k] for k in keys]))
    total = len(all_combos) * len(k_values)
    print(f"  Testing {total} combinations ({len(all_combos)} weight combos x {len(k_values)} k values)...")

    for k_val in k_values:
        for combo in all_combos:
            weights = dict(zip(keys, combo))

            # Quick RRF
            all_qids = set()
            for rl in active_retrievers.values():
                all_qids.update(rl.keys())

            fused = {}
            for qid in all_qids:
                scores = defaultdict(float)
                for name, ranked_lists in active_retrievers.items():
                    w = weights.get(name, 1.0)
                    for rank, doc_id in enumerate(ranked_lists.get(qid, [])):
                        scores[doc_id] += w / (k_val + rank + 1)
                sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                # Apply year filter inline
                q_year = query_years.get(qid, 9999)
                kept = [(d, s) for d, s in sorted_docs if corpus_years.get(d, 0) <= q_year]
                fused[qid] = [d for d, _ in kept[:100]]

            # Evaluate
            result = evaluate(fused, qrels, verbose=False)
            ndcg10 = result["overall"]["NDCG@10"]

            if ndcg10 > best_score:
                best_score = ndcg10
                best_params = {"weights": weights, "k": k_val, "ndcg10": ndcg10}

    print(f"\n  Best NDCG@10: {best_score:.4f}")
    print(f"  Best k: {best_params['k']}")
    print(f"  Best weights: {best_params['weights']}")
    return best_params

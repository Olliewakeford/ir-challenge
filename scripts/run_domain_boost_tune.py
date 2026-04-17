#!/usr/bin/env python3
"""Fine-tune domain boost parameter — this is our biggest gain yet."""
import sys
import time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import (
    DATA_DIR, RESULTS_DIR,
    load_corpus, load_queries, load_qrels, load_ranked_lists, save_ranked_lists,
    evaluate,
)


def fuse_rrf(all_retrievers, weights, k_val):
    all_qids = set()
    for rl in all_retrievers.values():
        all_qids.update(rl.keys())
    fused = {}
    for qid in all_qids:
        scores = defaultdict(float)
        for name, ranked_lists in all_retrievers.items():
            w = weights.get(name, 0.0)
            if w == 0:
                continue
            for rank, doc_id in enumerate(ranked_lists.get(qid, [])):
                scores[doc_id] += w / (k_val + rank + 1)
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        fused[qid] = [d for d, _ in sorted_docs[:300]]
    return fused


def apply_domain_boost(fused_results, query_domains, corpus_domains, boost_factor):
    """Boost candidates from the same domain as the query."""
    boosted = {}
    for qid, ranked in fused_results.items():
        q_domain = query_domains.get(qid, "")
        scored = []
        for rank, doc_id in enumerate(ranked):
            base_score = 1.0 / (rank + 1)
            c_domain = corpus_domains.get(doc_id, "")
            if c_domain == q_domain and q_domain:
                base_score *= boost_factor
            scored.append((doc_id, base_score))
        scored.sort(key=lambda x: x[1], reverse=True)
        boosted[qid] = [d for d, _ in scored[:300]]
    return boosted


def apply_venue_boost(fused_results, query_venues, corpus_venues, boost_factor):
    """Boost candidates from the same venue as the query."""
    boosted = {}
    for qid, ranked in fused_results.items():
        q_venue = query_venues.get(qid, "")
        scored = []
        for rank, doc_id in enumerate(ranked):
            base_score = 1.0 / (rank + 1)
            c_venue = corpus_venues.get(doc_id, "")
            if c_venue == q_venue and q_venue:
                base_score *= boost_factor
            scored.append((doc_id, base_score))
        scored.sort(key=lambda x: x[1], reverse=True)
        boosted[qid] = [d for d, _ in scored[:300]]
    return boosted


def apply_combined_boost(fused_results, query_domains, corpus_domains,
                         query_venues, corpus_venues, domain_boost, venue_boost):
    """Apply both domain and venue boost."""
    boosted = {}
    for qid, ranked in fused_results.items():
        q_domain = query_domains.get(qid, "")
        q_venue = query_venues.get(qid, "")
        scored = []
        for rank, doc_id in enumerate(ranked):
            base_score = 1.0 / (rank + 1)
            c_domain = corpus_domains.get(doc_id, "")
            c_venue = corpus_venues.get(doc_id, "")
            if c_domain == q_domain and q_domain:
                base_score *= domain_boost
            if c_venue == q_venue and q_venue:
                base_score *= venue_boost
            scored.append((doc_id, base_score))
        scored.sort(key=lambda x: x[1], reverse=True)
        boosted[qid] = [d for d, _ in scored[:300]]
    return boosted


def main():
    start = time.time()

    corpus_df = load_corpus(DATA_DIR / "corpus.parquet")
    queries_df = load_queries(DATA_DIR / "queries.parquet")
    qrels = load_qrels(DATA_DIR / "qrels.json")

    query_domains = dict(zip(queries_df["doc_id"], queries_df["domain"]))
    corpus_domains = dict(zip(corpus_df["doc_id"], corpus_df["domain"]))
    query_venues = dict(zip(queries_df["doc_id"], queries_df["venue"]))
    corpus_venues = dict(zip(corpus_df["doc_id"], corpus_df["venue"]))

    # Load v4 config
    import json
    config = json.load(open(RESULTS_DIR / "v4_config.json"))
    v4_weights = config["weights"]
    v4_k = config["k"]

    retrievers = {}
    for name in v4_weights:
        path = RESULTS_DIR / f"stage1_{name}_public.json"
        if path.exists():
            retrievers[name] = load_ranked_lists(path)

    fused_v4 = fuse_rrf(retrievers, v4_weights, v4_k)
    r = evaluate(fused_v4, qrels, verbose=False)
    print(f"V4 baseline: NDCG@10={r['overall']['NDCG@10']:.4f}")

    # Fine-tune domain boost
    print("\n=== Domain boost fine-tuning ===")
    best_score = 0
    best_boost = 1.0
    for boost in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0]:
        boosted = apply_domain_boost(fused_v4, query_domains, corpus_domains, boost)
        r = evaluate(boosted, qrels, verbose=False)
        ndcg = r['overall']['NDCG@10']
        recall = r['overall']['Recall@100']
        marker = " ↑" if ndcg > best_score else ""
        print(f"  boost={boost:5.1f}: NDCG@10={ndcg:.4f} Recall@100={recall:.4f}{marker}")
        if ndcg > best_score:
            best_score = ndcg
            best_boost = boost

    print(f"\n  Best domain boost: {best_boost} → NDCG@10={best_score:.4f}")

    # Fine-tune around best boost
    print("\n=== Finer grid around best ===")
    for boost in [best_boost * f for f in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]]:
        boost = round(boost, 1)
        boosted = apply_domain_boost(fused_v4, query_domains, corpus_domains, boost)
        r = evaluate(boosted, qrels, verbose=False)
        ndcg = r['overall']['NDCG@10']
        if ndcg > best_score:
            best_score = ndcg
            best_boost = boost
            print(f"  boost={boost:.1f}: NDCG@10={ndcg:.4f} ↑")

    # Test venue boost on top of domain boost
    print(f"\n=== Venue boost on top of domain boost={best_boost} ===")
    best_combined = best_score
    best_venue_boost = 1.0
    for venue_b in [1.0, 1.2, 1.5, 2.0, 3.0, 5.0]:
        boosted = apply_combined_boost(fused_v4, query_domains, corpus_domains,
                                        query_venues, corpus_venues, best_boost, venue_b)
        r = evaluate(boosted, qrels, verbose=False)
        ndcg = r['overall']['NDCG@10']
        marker = " ↑" if ndcg > best_combined else ""
        print(f"  domain={best_boost} venue={venue_b}: NDCG@10={ndcg:.4f}{marker}")
        if ndcg > best_combined:
            best_combined = ndcg
            best_venue_boost = venue_b

    # Test on DIFFERENT fusion base (v2 without TF-IDF)
    print("\n=== Also test domain boost on v2 fusion ===")
    v2_weights = {
        "bm25_ft": 2.5, "bm25_sections": 0.2, "bm25_ta": 0.15,
        "citation_ctx": 0.15, "minilm": 1.65, "scincl": 0.7, "specter2": 0.3,
    }
    v2_retrievers = {k: v for k, v in retrievers.items() if k in v2_weights}
    fused_v2 = fuse_rrf(v2_retrievers, v2_weights, 10)
    for boost in [3.0, 5.0, 8.0, 10.0]:
        boosted = apply_domain_boost(fused_v2, query_domains, corpus_domains, boost)
        r = evaluate(boosted, qrels, verbose=False)
        ndcg = r['overall']['NDCG@10']
        print(f"  v2+domain_boost={boost}: NDCG@10={ndcg:.4f}")

    # Save best result
    final_boost = best_boost
    final_venue = best_venue_boost
    if best_combined > best_score:
        boosted = apply_combined_boost(fused_v4, query_domains, corpus_domains,
                                        query_venues, corpus_venues, final_boost, final_venue)
        print(f"\n  Using combined boost: domain={final_boost} venue={final_venue}")
    else:
        boosted = apply_domain_boost(fused_v4, query_domains, corpus_domains, final_boost)
        final_venue = 1.0
        print(f"\n  Using domain-only boost: {final_boost}")

    r = evaluate(boosted, qrels, verbose=False)
    print(f"  Final: NDCG@10={r['overall']['NDCG@10']:.4f}  "
          f"Recall@100={r['overall']['Recall@100']:.4f}  MAP={r['overall']['MAP']:.4f}")

    save_ranked_lists(boosted, RESULTS_DIR / "stage2_domain_boosted_public.json")

    # Save config
    boost_config = {
        "base": "v4",
        "domain_boost": final_boost,
        "venue_boost": final_venue,
        "ndcg10": r['overall']['NDCG@10'],
    }
    with open(RESULTS_DIR / "domain_boost_config.json", "w") as f:
        json.dump(boost_config, f, indent=2)

    print(f"\n  Time: {time.time()-start:.1f}s")


if __name__ == "__main__":
    main()

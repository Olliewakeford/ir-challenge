#!/usr/bin/env python3
"""
Refusion v4: Add TF-IDF full-text signal and do coordinate descent over all 8 weights.
Starting from TF-IDF w=1.5 k=8 → 0.6023 baseline.
"""
import sys
import time
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import (
    DATA_DIR, RESULTS_DIR,
    load_qrels, load_ranked_lists, save_ranked_lists,
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


def main():
    start = time.time()
    qrels = load_qrels(DATA_DIR / "qrels.json")

    # Load all retrievers including TF-IDF
    retrievers = {}
    for name in ["specter2", "scincl", "minilm", "bm25_ta", "bm25_ft",
                  "bm25_sections", "citation_ctx", "tfidf_ft"]:
        path = RESULTS_DIR / f"stage1_{name}_public.json"
        if path.exists():
            retrievers[name] = load_ranked_lists(path)
    print(f"Loaded {len(retrievers)} retrievers: {list(retrievers.keys())}")

    # Starting point from individual tuning
    best_weights = {
        "bm25_ft": 2.5, "bm25_sections": 0.2, "bm25_ta": 0.15,
        "citation_ctx": 0.15, "minilm": 1.65, "scincl": 0.7, "specter2": 0.3,
        "tfidf_ft": 1.5,
    }
    best_k = 8

    # Phase 1: Grid search over k and tfidf_ft weight
    print("\n=== Phase 1: k and TF-IDF weight grid ===")
    best_score = 0
    for k in [6, 7, 8, 9, 10, 12]:
        for tfidf_w in [0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5]:
            w = dict(best_weights)
            w["tfidf_ft"] = tfidf_w
            fused = fuse_rrf(retrievers, w, k)
            r = evaluate(fused, qrels, verbose=False)
            ndcg = r['overall']['NDCG@10']
            if ndcg > best_score:
                best_score = ndcg
                best_weights["tfidf_ft"] = tfidf_w
                best_k = k
                print(f"  k={k} tfidf={tfidf_w}: NDCG@10={ndcg:.4f} ↑")

    print(f"\n  Phase 1 best: NDCG@10={best_score:.4f} tfidf={best_weights['tfidf_ft']} k={best_k}")

    # Phase 2: Coordinate descent on all weights
    print("\n=== Phase 2: Coordinate descent (all 8 weights) ===")
    weight_ranges = {
        "bm25_ft": [1.5, 2.0, 2.5, 3.0, 3.5],
        "minilm": [1.0, 1.3, 1.5, 1.65, 1.8, 2.0, 2.5],
        "scincl": [0.3, 0.5, 0.7, 0.9, 1.2],
        "specter2": [0.1, 0.2, 0.3, 0.5, 0.7],
        "bm25_ta": [0.0, 0.1, 0.15, 0.2, 0.3],
        "bm25_sections": [0.0, 0.1, 0.2, 0.3, 0.5],
        "citation_ctx": [0.0, 0.1, 0.15, 0.2, 0.3],
        "tfidf_ft": [0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0],
    }

    for iteration in range(3):
        improved = False
        for param_name, values in weight_ranges.items():
            old_val = best_weights[param_name]
            for val in values:
                if val == old_val:
                    continue
                w = dict(best_weights)
                w[param_name] = val
                fused = fuse_rrf(retrievers, w, best_k)
                r = evaluate(fused, qrels, verbose=False)
                ndcg = r['overall']['NDCG@10']
                if ndcg > best_score:
                    best_score = ndcg
                    best_weights[param_name] = val
                    improved = True
                    print(f"  iter={iteration} {param_name}={val}: NDCG@10={ndcg:.4f} ↑")

        # Also try k
        for k in [5, 6, 7, 8, 9, 10, 12, 15]:
            if k == best_k:
                continue
            fused = fuse_rrf(retrievers, best_weights, k)
            r = evaluate(fused, qrels, verbose=False)
            ndcg = r['overall']['NDCG@10']
            if ndcg > best_score:
                best_score = ndcg
                best_k = k
                improved = True
                print(f"  iter={iteration} k={k}: NDCG@10={ndcg:.4f} ↑")

        if not improved:
            print(f"  Converged at iteration {iteration}")
            break

    print(f"\n=== FINAL RESULTS ===")
    print(f"  Best NDCG@10: {best_score:.4f}")
    print(f"  Best k: {best_k}")
    print(f"  Best weights:")
    for name, w in sorted(best_weights.items(), key=lambda x: x[1], reverse=True):
        print(f"    {name}: {w}")

    # Generate and save final fused results
    fused_final = fuse_rrf(retrievers, best_weights, best_k)
    save_ranked_lists(fused_final, RESULTS_DIR / "stage1_fused_v4_public.json")

    r = evaluate(fused_final, qrels, verbose=False)
    print(f"\n  Final: NDCG@10={r['overall']['NDCG@10']:.4f}  "
          f"Recall@100={r['overall']['Recall@100']:.4f}  MAP={r['overall']['MAP']:.4f}")

    # Save config for held-out submission
    config = {"weights": best_weights, "k": best_k, "ndcg10": best_score}
    import json
    with open(RESULTS_DIR / "v4_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"\n  Config saved to v4_config.json")

    print(f"\n  Time: {time.time()-start:.1f}s")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
K-NN Graph Neighbor Expansion — after initial retrieval, pull in
embedding neighbors of top-scoring docs to boost recall.

The idea: if doc A is highly ranked, docs that are embedding-neighbors
of A are likely also relevant. This is Graph-based Adaptive Reranking (GAR)
from MacAvaney et al. (CIKM 2022).

With only 20K docs, building a k-NN graph is trivial (<1 min).
"""
import sys
import json
import time
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import (
    DATA_DIR, RESULTS_DIR, EMB_DIR,
    load_queries, load_qrels, load_ranked_lists, save_ranked_lists,
    load_embeddings, evaluate,
)
from tqdm.auto import tqdm


def build_knn_graph(embeddings, ids, k=20):
    """Build k-NN graph from embeddings using cosine similarity."""
    print(f"  Building k-NN graph (k={k}) from {len(ids)} embeddings...")

    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normed = embeddings / norms

    # Compute similarities in batches to avoid OOM
    batch_size = 500
    knn_graph = {}  # doc_id -> [neighbor_doc_ids]

    for i in tqdm(range(0, len(ids), batch_size), desc="k-NN graph"):
        batch = normed[i:i+batch_size]
        sims = batch @ normed.T  # (batch_size, N)

        for j in range(len(batch)):
            idx = i + j
            # Get top-k+1 (exclude self)
            top_idx = np.argsort(-sims[j])[:k+1]
            neighbors = [ids[t] for t in top_idx if t != idx][:k]
            knn_graph[ids[idx]] = neighbors

    return knn_graph


def expand_with_neighbors(ranked_lists, knn_graph, top_k_to_expand=30, max_neighbors=10):
    """Expand ranked lists by adding neighbors of top-k documents."""
    expanded = {}

    for qid, doc_ids in ranked_lists.items():
        # Start with original ranking
        scores = defaultdict(float)
        for rank, doc_id in enumerate(doc_ids):
            base_score = 1.0 / (30 + rank + 1)  # RRF-like score
            scores[doc_id] = max(scores[doc_id], base_score)

            # Add neighbors of top-k documents
            if rank < top_k_to_expand and doc_id in knn_graph:
                neighbors = knn_graph[doc_id][:max_neighbors]
                for n_rank, neighbor in enumerate(neighbors):
                    if neighbor == qid:  # Skip self-citations
                        continue
                    # Neighbor gets a discounted score based on both original rank and neighbor distance
                    n_score = 0.5 * base_score / (1 + n_rank)
                    scores[neighbor] = max(scores[neighbor], n_score)

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        expanded[qid] = [d for d, _ in sorted_docs[:300]]

    return expanded


def main():
    start_time = time.time()

    cache_path = RESULTS_DIR / "stage1_knn_expanded_public.json"
    if cache_path.exists():
        print(f"Loading cached results from {cache_path}")
        results = load_ranked_lists(cache_path)
        qrels = load_qrels(DATA_DIR / "qrels.json")
        r = evaluate(results, qrels, verbose=False)
        print(f"k-NN Expanded: NDCG@10={r['overall']['NDCG@10']:.4f}  "
              f"Recall@100={r['overall']['Recall@100']:.4f}  MAP={r['overall']['MAP']:.4f}")
        return

    print("=" * 68)
    print("K-NN GRAPH NEIGHBOR EXPANSION")
    print("=" * 68)

    # Load SPECTER2 corpus embeddings (best for scientific similarity)
    print("\nLoading SPECTER2 corpus embeddings...")
    emb_dir = EMB_DIR / "specter2_asymmetric"
    corpus_embs, corpus_ids = load_embeddings(
        emb_dir / "corpus_embeddings.npy",
        emb_dir / "corpus_ids.json"
    )
    print(f"  Embeddings: {corpus_embs.shape}")

    # Build k-NN graph
    knn_graph = build_knn_graph(corpus_embs, corpus_ids, k=20)
    print(f"  Graph: {len(knn_graph)} nodes, avg {np.mean([len(v) for v in knn_graph.values()]):.0f} neighbors")

    # Load fused results
    print("\nLoading fused results...")
    fused = load_ranked_lists(RESULTS_DIR / "stage1_fused_public.json")
    qrels = load_qrels(DATA_DIR / "qrels.json")

    # Expand with neighbors
    print("\nExpanding with k-NN neighbors...")
    expanded = expand_with_neighbors(fused, knn_graph, top_k_to_expand=50, max_neighbors=10)

    # Evaluate
    r_orig = evaluate(fused, qrels, verbose=False)
    r_exp = evaluate(expanded, qrels, verbose=False)

    print(f"\n  Original fused:    NDCG@10={r_orig['overall']['NDCG@10']:.4f}  "
          f"Recall@100={r_orig['overall']['Recall@100']:.4f}")
    print(f"  After expansion:   NDCG@10={r_exp['overall']['NDCG@10']:.4f}  "
          f"Recall@100={r_exp['overall']['Recall@100']:.4f}")

    improvement = r_exp['overall']['Recall@100'] - r_orig['overall']['Recall@100']
    print(f"  Recall improvement: {'+' if improvement > 0 else ''}{improvement:.4f}")

    save_ranked_lists(expanded, cache_path)

    elapsed = time.time() - start_time
    print(f"\n  Time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()

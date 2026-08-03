#!/usr/bin/env python3
"""
Use cross-encoder scores as an additional retriever signal in fusion.

The insight: cross-encoder reranking HURTS when used to re-order (0.5918 → 0.4762).
But what if we use cross-encoder scores as just ANOTHER signal alongside the existing
7 retrievers? The fusion can learn to weight it down if it's not helpful.

This script:
1. Score top-200 candidates with cross-encoder
2. Rank by cross-encoder score → save as a "retriever" ranked list
3. Include in the refusion with the other 7 retrievers
"""
import sys
import time
import gc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from irchallenge.metrics import evaluate
from irchallenge.paths import DATA_DIR, RESULTS_DIR
from irchallenge.storage import load_corpus, load_qrels, load_queries, load_ranked_lists, save_ranked_lists
from irchallenge.text import get_ta
from tqdm.auto import tqdm


def main():
    start_time = time.time()

    cache_path = RESULTS_DIR / "stage1_crossenc_public.json"
    if cache_path.exists():
        print(f"Loading cached results from {cache_path}")
        results = load_ranked_lists(cache_path)
        qrels = load_qrels(DATA_DIR / "qrels.json")
        r = evaluate(results, qrels, verbose=False)
        print(f"Cross-encoder signal: NDCG@10={r['overall']['NDCG@10']:.4f}  "
              f"Recall@100={r['overall']['Recall@100']:.4f}")
        return

    print("=" * 68)
    print("CROSS-ENCODER AS RETRIEVER SIGNAL")
    print("=" * 68)

    # Load data
    corpus_df = load_corpus(DATA_DIR / "corpus.parquet")
    queries_df = load_queries(DATA_DIR / "queries.parquet")
    qrels = load_qrels(DATA_DIR / "qrels.json")

    # Load fused results to get candidate pool
    fused = load_ranked_lists(RESULTS_DIR / "stage1_fused_v2_public.json")

    # Build lookups
    query_lookup = {row["doc_id"]: get_ta(row)[:512] for _, row in queries_df.iterrows()}
    corpus_lookup = {row["doc_id"]: get_ta(row)[:512] for _, row in corpus_df.iterrows()}

    # Load cross-encoder
    from sentence_transformers import CrossEncoder
    print("\nLoading cross-encoder (ms-marco-MiniLM-L-12-v2)...")
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2", device="mps")

    # Score and rank candidates
    print("\nScoring candidates with cross-encoder...")
    results = {}

    for qid in tqdm(fused, desc="Cross-encoder scoring"):
        q_text = query_lookup.get(qid, "")
        candidates = fused[qid][:200]

        pairs = [[q_text, corpus_lookup.get(cid, "")] for cid in candidates]
        scores = model.predict(pairs, show_progress_bar=False)

        # Rank by cross-encoder score
        scored = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        results[qid] = [d for d, _ in scored]

    del model
    gc.collect()

    save_ranked_lists(results, cache_path)

    r = evaluate(results, qrels, verbose=False)
    print(f"\nCross-encoder signal: NDCG@10={r['overall']['NDCG@10']:.4f}  "
          f"Recall@100={r['overall']['Recall@100']:.4f}  MAP={r['overall']['MAP']:.4f}")

    elapsed = time.time() - start_time
    print(f"\n  Time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()

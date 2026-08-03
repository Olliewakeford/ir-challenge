#!/usr/bin/env python3
"""
Citation Context Retriever — extract sentences around citation markers
in query full_text and use them as additional retrieval queries.

The idea: text around [1], [2], [82,83] describes EXACTLY why a paper cites
another. These contexts capture citation intent that title+abstract queries miss.

For each query paper:
1. Extract citation context windows (±150 chars around each marker)
2. Deduplicate and clean
3. Use each context as a BM25 query against corpus TA
4. Aggregate scores across all contexts for that query
5. Return top-200 candidates
"""
import re
import sys
import gc
import json
import time
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from irchallenge.config import PipelineConfig
from irchallenge.metrics import evaluate
from irchallenge.paths import DATA_DIR, RESULTS_DIR
from irchallenge.storage import load_corpus, load_qrels, load_queries, save_ranked_lists
from irchallenge.text import clean_citation_markers, get_ta, tokenize_simple
from rank_bm25 import BM25Okapi
from tqdm.auto import tqdm


def extract_citation_contexts(full_text, window=150, max_contexts=30):
    """Extract text windows around citation markers like [1], [82,83]."""
    pattern = r'\[(\d+(?:,\s*\d+)*)\]'
    matches = list(re.finditer(pattern, full_text))

    contexts = []
    seen_starts = set()

    for m in matches:
        # Get window around the citation
        start = max(0, m.start() - window)
        end = min(len(full_text), m.end() + window)
        ctx = full_text[start:end].strip()

        # Remove the citation marker itself
        ctx = re.sub(pattern, '', ctx).strip()

        # Deduplicate by approximate start position (within 50 chars)
        approx_start = start // 50
        if approx_start in seen_starts:
            continue
        seen_starts.add(approx_start)

        if len(ctx) > 30:  # Skip very short contexts
            contexts.append(ctx)

    # Return up to max_contexts
    return contexts[:max_contexts]


def main():
    start_time = time.time()
    config = PipelineConfig(query_set="public")

    cache_path = RESULTS_DIR / f"stage1_citation_ctx_public.json"
    if cache_path.exists():
        print(f"Loading cached results from {cache_path}")
        results = json.load(open(cache_path))
        qrels = load_qrels(DATA_DIR / "qrels.json")
        r = evaluate(results, qrels, verbose=False)
        print(f"Citation Context: NDCG@10={r['overall']['NDCG@10']:.4f}  "
              f"Recall@100={r['overall']['Recall@100']:.4f}  MAP={r['overall']['MAP']:.4f}")
        return

    print("=" * 68)
    print("CITATION CONTEXT RETRIEVER")
    print("=" * 68)

    # Load data
    print("\nLoading data...")
    corpus_df = load_corpus(DATA_DIR / "corpus.parquet")
    queries_df = load_queries(DATA_DIR / "queries.parquet")
    qrels = load_qrels(DATA_DIR / "qrels.json")

    # Build BM25 index on corpus title+abstract
    print("\nBuilding BM25 index on corpus TA...")
    corpus_ids = corpus_df["doc_id"].tolist()
    corpus_ta_tokens = [
        tokenize_simple(get_ta(row))
        for _, row in tqdm(corpus_df.iterrows(), total=len(corpus_df), desc="Tokenizing")
    ]
    bm25 = BM25Okapi(corpus_ta_tokens)

    # Process each query
    print(f"\nExtracting citation contexts and retrieving...")
    results = {}
    ctx_counts = []

    for _, row in tqdm(queries_df.iterrows(), total=len(queries_df), desc="Citation context retrieval"):
        qid = row["doc_id"]
        full_text = str(row.get("full_text", "") or "")

        # Extract citation contexts
        contexts = extract_citation_contexts(full_text, window=150, max_contexts=30)
        ctx_counts.append(len(contexts))

        if not contexts:
            # Fallback to TA query
            query_tokens = tokenize_simple(get_ta(row))
            scores = bm25.get_scores(query_tokens)
            top_idx = np.argsort(-scores)[:200]
            results[qid] = [corpus_ids[j] for j in top_idx]
            continue

        # Score each candidate by aggregating across all contexts
        agg_scores = np.zeros(len(corpus_ids), dtype=np.float64)

        for ctx in contexts:
            ctx_tokens = tokenize_simple(clean_citation_markers(ctx))
            if len(ctx_tokens) < 3:
                continue
            scores = bm25.get_scores(ctx_tokens)
            # Normalize scores for this context
            mx = scores.max()
            if mx > 0:
                scores = scores / mx
            agg_scores += scores

        top_idx = np.argsort(-agg_scores)[:200]
        results[qid] = [corpus_ids[j] for j in top_idx]

    # Save
    save_ranked_lists(results, cache_path)

    # Evaluate
    r = evaluate(results, qrels, verbose=False)
    print(f"\nCitation Context Retriever:")
    print(f"  NDCG@10={r['overall']['NDCG@10']:.4f}  "
          f"Recall@100={r['overall']['Recall@100']:.4f}  MAP={r['overall']['MAP']:.4f}")
    print(f"  Avg contexts per query: {np.mean(ctx_counts):.1f}")
    print(f"  Min/Max contexts: {min(ctx_counts)}/{max(ctx_counts)}")

    elapsed = time.time() - start_time
    print(f"\n  Time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()

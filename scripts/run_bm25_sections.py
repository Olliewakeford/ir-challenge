#!/usr/bin/env python3
"""Quick script to run BM25-Sections retrieval and cache it."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import (
    PipelineConfig, DATA_DIR, RESULTS_DIR,
    load_corpus, load_queries, load_qrels,
    retrieve_bm25_sections, evaluate,
)

config = PipelineConfig(query_set="public")
corpus_df = load_corpus(DATA_DIR / "corpus.parquet")
queries_df = load_queries(DATA_DIR / "queries.parquet")
qrels = load_qrels(DATA_DIR / "qrels.json")

print("Running BM25-Sections...")
results = retrieve_bm25_sections(queries_df, corpus_df, config)
r = evaluate(results, qrels, verbose=False)
print(f"BM25-Sections: NDCG@10={r['overall']['NDCG@10']:.4f}  Recall@100={r['overall']['Recall@100']:.4f}  MAP={r['overall']['MAP']:.4f}")

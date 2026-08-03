#!/usr/bin/env python3
"""
Generate held-out submission with v4 pipeline:
- 8 retrievers (7 existing + TF-IDF full-text)
- Tuned fusion weights (v4 coordinate descent)
- Domain + venue boost (our biggest gain)
- Optionally: LLM reranking

Usage:
  python scripts/generate_submission_v4.py                     # With domain boost, no LLM
  python scripts/generate_submission_v4.py --llm               # With LLM reranking
  python scripts/generate_submission_v4.py --boost 10          # Custom boost
"""
import argparse
import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from irchallenge.boost import apply_domain_venue_boost
from irchallenge.config import PipelineConfig
from irchallenge.fusion import weighted_rrf_fuse as fuse_rrf_weighted
from irchallenge.metrics import evaluate
from irchallenge.paths import CHALLENGE_DIR, DATA_DIR, EMB_DIR, RESULTS_DIR, SUBMISSIONS_DIR
from irchallenge.retrievers import (
    retrieve_bm25_ft,
    retrieve_bm25_sections,
    retrieve_bm25_ta,
    retrieve_citation_ctx,
    retrieve_minilm,
    retrieve_scincl,
    retrieve_specter2,
    retrieve_tfidf_ft,
)
from irchallenge.storage import load_corpus, load_qrels, load_queries, save_ranked_lists
from irchallenge.submission import assemble_submission


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", action="store_true", help="Include LLM reranking")
    parser.add_argument("--boost", type=float, default=10.0, help="Domain boost factor")
    parser.add_argument("--venue-boost", type=float, default=2.0, help="Venue boost factor")
    parser.add_argument("--label", default="submission_v4", help="Submission label")
    return parser.parse_args()


def main():
    args = parse_args()
    start = time.time()

    # V4 weights from coordinate descent
    rrf_weights = {
        "bm25_ft": 3.5,
        "minilm": 1.8,
        "tfidf_ft": 1.2,
        "scincl": 0.7,
        "specter2": 0.3,
        "bm25_sections": 0.2,
        "citation_ctx": 0.15,
        "bm25_ta": 0.0,
    }
    rrf_k = 6

    config = PipelineConfig(
        query_set="held_out",
        apply_year_filter=False,
        rrf_k=rrf_k,
        rrf_weights=rrf_weights,
        fusion_top_k=300,
        llm_top_k=30,
    )

    print("=" * 68)
    print(f"GENERATING V4 HELD-OUT SUBMISSION")
    print(f"  Domain boost: {args.boost}, Venue boost: {args.venue_boost}")
    print(f"  LLM reranking: {args.llm}")
    print("=" * 68)

    # Load data
    print("\nLoading data...")
    corpus_df = load_corpus(DATA_DIR / "corpus.parquet")
    queries_df = load_queries(DATA_DIR / "held_out_queries.parquet")
    print(f"  Corpus: {len(corpus_df)} | Held-out queries: {len(queries_df)}")

    # ── Stage 1: Run all retrievers ──
    print("\n" + "=" * 68)
    print("STAGE 1: RETRIEVAL (8 signals)")
    print("=" * 68)

    all_retrievers = {}

    # Dense retrievers
    print("\n--- SPECTER2 ---")
    all_retrievers["specter2"] = retrieve_specter2(queries_df, corpus_df, config)
    gc.collect()

    print("\n--- SciNCL ---")
    all_retrievers["scincl"] = retrieve_scincl(queries_df, corpus_df, config)
    gc.collect()

    print("\n--- MiniLM ---")
    all_retrievers["minilm"] = retrieve_minilm(queries_df, corpus_df, config)
    gc.collect()

    # BM25 retrievers
    if rrf_weights.get("bm25_ta", 0) > 0:
        print("\n--- BM25-TA ---")
        all_retrievers["bm25_ta"] = retrieve_bm25_ta(queries_df, corpus_df, config)
        gc.collect()

    print("\n--- BM25-FT ---")
    all_retrievers["bm25_ft"] = retrieve_bm25_ft(queries_df, corpus_df, config)
    gc.collect()

    if rrf_weights.get("bm25_sections", 0) > 0:
        print("\n--- BM25-Sections ---")
        all_retrievers["bm25_sections"] = retrieve_bm25_sections(queries_df, corpus_df, config)
        gc.collect()

    # Citation context
    if rrf_weights.get("citation_ctx", 0) > 0:
        print("\n--- Citation Context ---")
        all_retrievers["citation_ctx"] = retrieve_citation_ctx(queries_df, corpus_df, config)
        gc.collect()

    # TF-IDF full text (new in v4)
    print("\n--- TF-IDF Full Text ---")
    all_retrievers["tfidf_ft"] = retrieve_tfidf_ft(queries_df, corpus_df, config)
    gc.collect()

    print(f"\n  Active retrievers: {list(all_retrievers.keys())}")

    # ── Stage 2: RRF Fusion ──
    print("\n" + "=" * 68)
    print("STAGE 2: WEIGHTED RRF FUSION (v4)")
    print("=" * 68)
    fused = fuse_rrf_weighted(all_retrievers, rrf_weights, rrf_k)
    save_ranked_lists(fused, RESULTS_DIR / "stage1_fused_v4_held_out.json")
    print(f"  Fused: {len(fused)} queries, {len(next(iter(fused.values())))} candidates each")

    # ── Stage 3: Domain + Venue Boost ──
    print("\n" + "=" * 68)
    print(f"STAGE 3: DOMAIN BOOST ({args.boost}x) + VENUE BOOST ({args.venue_boost}x)")
    print("=" * 68)
    boosted = apply_domain_venue_boost(fused, queries_df, corpus_df,
                                        args.boost, args.venue_boost)
    save_ranked_lists(boosted, RESULTS_DIR / "stage2_boosted_v4_held_out.json")

    # ── Stage 4: LLM Reranking (optional) ──
    if args.llm:
        print("\n" + "=" * 68)
        print("STAGE 4: LLM RERANKING")
        print("=" * 68)
        from run_llm_rerank import rerank_llm_sliding_window
        final = rerank_llm_sliding_window(
            boosted, queries_df, corpus_df, config,
            window_size=30, step_size=20, n_windows=2
        )
    else:
        print("\n  Skipping LLM reranking")
        final = boosted

    # ── Stage 5: Assemble submission ──
    print("\n" + "=" * 68)
    print("ASSEMBLING SUBMISSION")
    print("=" * 68)

    submission = assemble_submission(final, config, label=args.label)

    # Verify
    print("\n  Verification:")
    print(f"    Queries: {len(submission)}")
    print(f"    Docs per query: {len(next(iter(submission.values())))}")

    corpus_ids = set(corpus_df["doc_id"].tolist())
    query_ids = set(queries_df["doc_id"].tolist())
    missing = 0
    for qid, docs in submission.items():
        for d in docs:
            if d not in corpus_ids:
                missing += 1
    if missing:
        print(f"    WARNING: {missing} doc_ids not found in corpus!")
    else:
        print(f"    All doc_ids verified in corpus")

    elapsed = time.time() - start
    print(f"\n{'='*68}")
    print(f"V4 SUBMISSION GENERATED — {elapsed/60:.1f} minutes")
    print(f"{'='*68}")
    print(f"  ZIP: submissions/{args.label}.zip")
    print(f"  Upload to: https://www.codabench.org/competitions/15308/")


if __name__ == "__main__":
    main()

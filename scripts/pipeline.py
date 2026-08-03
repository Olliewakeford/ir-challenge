#!/usr/bin/env python3
"""Backwards-compatible entry point for the old single-file pipeline.

The implementation now lives in irchallenge/ (paths, config, storage, text,
metrics, device, retrievers, fusion, filters, boost, rerank, llm_rerank,
submission, cli). This module re-exports the names other scripts used to
import from here, and keeps `python scripts/pipeline.py` working as a CLI.

Usage:
  python scripts/pipeline.py                     # Full pipeline on public queries
  python scripts/pipeline.py --query-set held_out # Generate submission
  python scripts/pipeline.py --stages 1,2,3       # Run specific stages
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from irchallenge.cli import eval_if_public, main
from irchallenge.config import PipelineConfig
from irchallenge.device import get_device
from irchallenge.filters import apply_year_filter
from irchallenge.fusion import fuse_rrf, fuse_rrf_with_scores, tune_rrf_weights
from irchallenge.llm_rerank import rerank_llm
from irchallenge.metrics import (
    average_precision,
    evaluate,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from irchallenge.paths import CHALLENGE_DIR, DATA_DIR, EMB_DIR, RESULTS_DIR, SUBMISSIONS_DIR
from irchallenge.rerank import compare_cross_encoders, rerank_bge, rerank_cross_encoder
from irchallenge.retrievers import (
    dense_retrieve,
    retrieve_bm25_ft,
    retrieve_bm25_sections,
    retrieve_bm25_ta,
    retrieve_citation_ctx,
    retrieve_minilm,
    retrieve_scincl,
    retrieve_specter2,
)
from irchallenge.storage import (
    load_corpus,
    load_embeddings,
    load_qrels,
    load_queries,
    load_ranked_lists,
    save_ranked_lists,
)
from irchallenge.submission import assemble_submission
from irchallenge.text import (
    clean_citation_markers,
    format_text,
    get_body_chunks,
    get_chunks,
    get_ta,
    tokenize_simple,
)

__all__ = [
    "PipelineConfig", "CHALLENGE_DIR", "DATA_DIR", "EMB_DIR", "SUBMISSIONS_DIR", "RESULTS_DIR",
    "load_queries", "load_corpus", "load_qrels", "load_embeddings",
    "save_ranked_lists", "load_ranked_lists",
    "format_text", "get_chunks", "get_ta", "get_body_chunks",
    "clean_citation_markers", "tokenize_simple", "get_device",
    "recall_at_k", "precision_at_k", "mrr_at_k", "ndcg_at_k", "average_precision", "evaluate",
    "dense_retrieve", "retrieve_specter2", "retrieve_scincl", "retrieve_minilm",
    "retrieve_bm25_ta", "retrieve_bm25_ft", "retrieve_bm25_sections", "retrieve_citation_ctx",
    "fuse_rrf", "fuse_rrf_with_scores", "tune_rrf_weights",
    "apply_year_filter",
    "rerank_cross_encoder", "rerank_bge", "compare_cross_encoders",
    "rerank_llm",
    "assemble_submission",
    "eval_if_public", "main",
]

if __name__ == "__main__":
    main()

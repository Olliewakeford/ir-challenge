#!/usr/bin/env python3
"""
pipeline.py — Full IR Challenge Pipeline for Scientific Citation Prediction.

Multi-stage retrieval pipeline:
  Stage 1: Multi-signal retrieval (5+ retrievers → Weighted RRF)
  Stage 2: Year filter + metadata boosting
  Stage 3: Cross-encoder reranking
  Stage 4: LLM listwise reranking (Claude)
  Stage 5: Assembly → submission

Usage:
  python scripts/pipeline.py                     # Full pipeline on public queries
  python scripts/pipeline.py --query-set held_out # Generate submission
  python scripts/pipeline.py --stages 1,2,3      # Run specific stages
"""

import argparse
import dataclasses
import gc
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# ─── Paths ───────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
CHALLENGE_DIR = SCRIPT_DIR.parent
DATA_DIR = CHALLENGE_DIR / "data"
EMB_DIR = DATA_DIR / "embeddings"
SUBMISSIONS_DIR = CHALLENGE_DIR / "submissions"
RESULTS_DIR = DATA_DIR / "intermediate"

for d in [EMB_DIR, SUBMISSIONS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

@dataclass
class PipelineConfig:
    # Query set
    query_set: str = "public"  # "public" or "held_out"

    # Stage 1: Retrieval
    retriever_top_k: int = 200
    rrf_k: int = 60
    rrf_weights: dict = field(default_factory=lambda: {
        "specter2": 1.5,
        "scincl": 1.0,
        "minilm": 0.7,
        "bm25_ta": 1.0,
        "bm25_ft": 0.8,
        "bm25_sections": 0.9,
    })
    fusion_top_k: int = 300

    # Stage 2: Filters
    apply_year_filter: bool = True

    # Stage 3: Cross-encoder
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    cross_encoder_batch_size: int = 64
    cross_encoder_top_k: int = 100
    cross_encoder_device: str = "mps"

    # Stage 4: LLM
    llm_model: str = "claude-sonnet-4-5-20250929"
    llm_top_k: int = 20

    # Bi-encoder batch sizes
    bi_encoder_batch_size: int = 128


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def load_queries(path) -> pd.DataFrame:
    return pd.read_parquet(path)

def load_corpus(path) -> pd.DataFrame:
    return pd.read_parquet(path)

def load_qrels(path) -> dict:
    with open(path) as f:
        return json.load(f)

def load_embeddings(emb_path, ids_path):
    embeddings = np.load(emb_path).astype(np.float32)
    with open(ids_path) as f:
        ids = json.load(f)
    assert len(embeddings) == len(ids), "Embedding count mismatch"
    return embeddings, ids

def format_text(row) -> str:
    title = str(row.get("title", "") or "").strip()
    abstract = str(row.get("abstract", "") or "").strip()
    if title and abstract:
        return title + " " + abstract
    return title or abstract

def get_chunks(full_text: str, chunk_meta_json) -> list:
    meta = json.loads(chunk_meta_json) if isinstance(chunk_meta_json, str) else chunk_meta_json
    chunks = []
    for i, entry in enumerate(meta):
        char_start = entry["char_start"]
        if entry["type"] == "ta":
            char_end = entry["char_end"]
        else:
            char_end = meta[i + 1]["char_start"] if i + 1 < len(meta) else len(full_text)
        text = full_text[char_start:char_end].strip()
        chunks.append({"type": entry["type"], "text": text,
                       "char_start": char_start, "char_end": char_end})
    return chunks

def get_ta(row) -> str:
    return str(row.get("ta", "") or "").strip()

def get_body_chunks(row, min_chars: int = 100) -> list:
    chunks = get_chunks(row["full_text"], row["chunk_meta"])
    return [c["text"] for c in chunks if c["type"] == "body" and len(c["text"]) >= min_chars]

def recall_at_k(ranked: list, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    return sum(1 for doc in ranked[:k] if doc in relevant) / len(relevant)

def precision_at_k(ranked: list, relevant: set, k: int) -> float:
    if k == 0:
        return 0.0
    return sum(1 for doc in ranked[:k] if doc in relevant) / k

def mrr_at_k(ranked: list, relevant: set, k: int) -> float:
    for rank, doc in enumerate(ranked[:k], start=1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0

def ndcg_at_k(ranked: list, relevant: set, k: int) -> float:
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc in enumerate(ranked[:k], start=1)
        if doc in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0

def average_precision(ranked: list, relevant: set) -> float:
    if not relevant:
        return 0.0
    hits, score = 0, 0.0
    for rank, doc in enumerate(ranked, start=1):
        if doc in relevant:
            hits += 1
            score += hits / rank
    return score / len(relevant)

def evaluate(submission, qrels, ks=None, query_domains=None, verbose=True):
    if ks is None:
        ks = [10, 100]
    per_query = {}
    for qid, rel_list in qrels.items():
        relevant = set(rel_list)
        ranked = submission.get(qid, [])
        q = {}
        for k in ks:
            q[f"Recall@{k}"] = recall_at_k(ranked, relevant, k)
            q[f"Precision@{k}"] = precision_at_k(ranked, relevant, k)
            q[f"MRR@{k}"] = mrr_at_k(ranked, relevant, k)
            q[f"NDCG@{k}"] = ndcg_at_k(ranked, relevant, k)
        q["AP"] = average_precision(ranked, relevant)
        per_query[qid] = q

    metric_keys = list(next(iter(per_query.values())).keys()) if per_query else []
    overall = {}
    for key in metric_keys:
        vals = [per_query[qid][key] for qid in per_query]
        overall[key] = float(np.mean(vals))
    overall["MAP"] = overall.pop("AP", 0.0)
    overall["num_queries"] = len(per_query)
    result = {"overall": overall, "per_query": per_query}

    if query_domains:
        per_domain = {}
        for domain in sorted(set(query_domains.values())):
            dqids = [q for q in per_query if query_domains.get(q) == domain]
            if not dqids:
                continue
            dm = {}
            for key in metric_keys:
                dm[key] = float(np.mean([per_query[q][key] for q in dqids]))
            dm["MAP"] = dm.pop("AP", 0.0)
            dm["num_queries"] = len(dqids)
            per_domain[domain] = dm
        result["per_domain"] = per_domain

    if verbose:
        _print_results(result, ks)
    return result

def _print_results(results, ks):
    o = results["overall"]
    print("\n" + "=" * 68)
    print("OVERALL RESULTS")
    print("=" * 68)
    for label, keys in [
        ("Recall", [f"Recall@{k}" for k in ks]),
        ("Precision", [f"Precision@{k}" for k in ks]),
        ("MRR", [f"MRR@{k}" for k in ks]),
        ("NDCG", [f"NDCG@{k}" for k in ks]),
    ]:
        row = f"{label:<14}"
        for k, key in zip(ks, keys):
            row += f"  @{k:>3}: {o.get(key, 0):.4f}"
        print(row)
    print(f"{'MAP':<14}  {o.get('MAP', 0):.4f}")
    print(f"{'Queries':<14}  {int(o.get('num_queries', 0))}")

    if "per_domain" in results:
        print("\n" + "-" * 68)
        print("PER-DOMAIN  (first k only)")
        print("-" * 68)
        k = ks[0]
        print(f"  {'Domain':<28} R@{k:<3} P@{k:<3} MRR@{k:<3} NDCG@{k:<3}  MAP    n")
        for domain, dm in sorted(results["per_domain"].items()):
            print(
                f"  {domain:<28}"
                f" {dm.get(f'Recall@{k}', 0):.3f}"
                f" {dm.get(f'Precision@{k}', 0):.3f}"
                f" {dm.get(f'MRR@{k}', 0):.3f}  "
                f" {dm.get(f'NDCG@{k}', 0):.3f}"
                f"  {dm.get('MAP', 0):.3f}"
                f"  {int(dm.get('num_queries', 0))}"
            )
    print()


# ═══════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def save_ranked_lists(ranked_lists: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(ranked_lists, f)
    print(f"  Saved {len(ranked_lists)} queries to {path}")

def load_ranked_lists(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)

def clean_citation_markers(text: str) -> str:
    return re.sub(r'\[\d+(?:,\s*\d+)*\]', '', text)

def tokenize_simple(text: str) -> list:
    return re.findall(r'\w+', text.lower())

def get_device(preferred="mps"):
    import torch
    if preferred == "mps" and torch.backends.mps.is_available():
        try:
            torch.zeros(1, device="mps")
            return "mps"
        except Exception:
            pass
    return "cpu"

def dense_retrieve(query_embs, corpus_embs, query_ids, corpus_ids, top_k=200):
    """Compute cosine similarity (assumes L2-normalized) and return top-k per query."""
    scores = query_embs @ corpus_embs.T  # (n_queries, n_corpus)
    results = {}
    for i, qid in enumerate(query_ids):
        top_idx = np.argsort(-scores[i])[:top_k]
        results[qid] = [corpus_ids[j] for j in top_idx]
    return results

def eval_if_public(results, qrels, config, label=""):
    """Evaluate only if we have qrels (public query set)."""
    if config.query_set == "public" and qrels:
        print(f"\n>>> Evaluating: {label}")
        return evaluate(results, qrels)
    return None


# ═══════════════════════════════════════════════════════════════
# STAGE 1: MULTI-SIGNAL RETRIEVAL
# ═══════════════════════════════════════════════════════════════

def retrieve_specter2(queries_df, corpus_df, config):
    """SPECTER2 asymmetric retrieval using adhoc_query + proximity adapters."""
    import torch
    import torch.nn.functional as F

    tag = config.query_set
    cache_path = RESULTS_DIR / f"stage1_specter2_{tag}.json"
    if cache_path.exists():
        print(f"  [SPECTER2] Loading cached results from {cache_path}")
        return load_ranked_lists(cache_path)

    emb_dir = EMB_DIR / "specter2_asymmetric"
    corpus_emb_path = emb_dir / "corpus_embeddings.npy"
    corpus_ids_path = emb_dir / "corpus_ids.json"
    query_emb_path = emb_dir / f"query_embeddings_{tag}.npy"
    query_ids_path = emb_dir / f"query_ids_{tag}.json"

    # Check if corpus embeddings already exist
    have_corpus = corpus_emb_path.exists() and corpus_ids_path.exists()
    have_queries = query_emb_path.exists() and query_ids_path.exists()

    if have_corpus and have_queries:
        print("  [SPECTER2] Loading cached embeddings...")
        corpus_embs, corpus_ids = load_embeddings(corpus_emb_path, corpus_ids_path)
        query_embs, query_ids = load_embeddings(query_emb_path, query_ids_path)
    else:
        from adapters import AutoAdapterModel
        from transformers import AutoTokenizer

        print("  [SPECTER2] Loading model and adapters...")
        tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
        model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
        model.load_adapter("allenai/specter2", source="hf", load_as="proximity")
        model.load_adapter("allenai/specter2_aug2023refresh_adhoc_query",
                           source="hf", load_as="adhoc_query")

        device = get_device("mps")
        model = model.to(device)
        model.eval()

        def encode_batch(texts, adapter_name, batch_size=128, desc="Encoding"):
            model.set_active_adapters(adapter_name)
            all_embs = []
            for i in tqdm(range(0, len(texts), batch_size), desc=desc):
                batch_texts = texts[i:i+batch_size]
                inputs = tokenizer(batch_texts, padding=True, truncation=True,
                                   max_length=512, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                cls_embs = outputs.last_hidden_state[:, 0, :]
                cls_embs = F.normalize(cls_embs, p=2, dim=1)
                all_embs.append(cls_embs.cpu().numpy())
            return np.concatenate(all_embs, axis=0).astype(np.float32)

        sep = tokenizer.sep_token or "[SEP]"

        # Encode corpus with proximity adapter
        if not have_corpus:
            print("  [SPECTER2] Encoding corpus with proximity adapter...")
            corpus_texts = [
                f"{str(row.get('title', '') or '').strip()}{sep}{str(row.get('abstract', '') or '').strip()}"
                for _, row in corpus_df.iterrows()
            ]
            corpus_ids = corpus_df["doc_id"].tolist()
            corpus_embs = encode_batch(corpus_texts, "proximity",
                                       batch_size=config.bi_encoder_batch_size,
                                       desc="SPECTER2 corpus")
            emb_dir.mkdir(parents=True, exist_ok=True)
            np.save(corpus_emb_path, corpus_embs)
            with open(corpus_ids_path, "w") as f:
                json.dump(corpus_ids, f)
            print(f"  Saved corpus embeddings: {corpus_embs.shape}")
        else:
            corpus_embs, corpus_ids = load_embeddings(corpus_emb_path, corpus_ids_path)

        # Encode queries with adhoc_query adapter
        print(f"  [SPECTER2] Encoding {tag} queries with adhoc_query adapter...")
        query_texts = [
            f"{str(row.get('title', '') or '').strip()}{sep}{str(row.get('abstract', '') or '').strip()}"
            for _, row in queries_df.iterrows()
        ]
        query_ids = queries_df["doc_id"].tolist()
        query_embs = encode_batch(query_texts, "adhoc_query",
                                  batch_size=config.bi_encoder_batch_size,
                                  desc="SPECTER2 queries")
        np.save(query_emb_path, query_embs)
        with open(query_ids_path, "w") as f:
            json.dump(query_ids, f)
        print(f"  Saved query embeddings: {query_embs.shape}")

        # Free model
        del model, tokenizer
        gc.collect()
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass

    results = dense_retrieve(query_embs, corpus_embs, query_ids, corpus_ids,
                             top_k=config.retriever_top_k)
    save_ranked_lists(results, cache_path)
    return results


def retrieve_scincl(queries_df, corpus_df, config):
    """SciNCL dense retrieval."""
    tag = config.query_set
    cache_path = RESULTS_DIR / f"stage1_scincl_{tag}.json"
    if cache_path.exists():
        print(f"  [SciNCL] Loading cached results from {cache_path}")
        return load_ranked_lists(cache_path)

    emb_dir = EMB_DIR / "scincl"
    corpus_emb_path = emb_dir / "corpus_embeddings.npy"
    corpus_ids_path = emb_dir / "corpus_ids.json"
    query_emb_path = emb_dir / f"query_embeddings_{tag}.npy"
    query_ids_path = emb_dir / f"query_ids_{tag}.json"

    have_corpus = corpus_emb_path.exists() and corpus_ids_path.exists()
    have_queries = query_emb_path.exists() and query_ids_path.exists()

    if have_corpus and have_queries:
        print("  [SciNCL] Loading cached embeddings...")
        corpus_embs, corpus_ids = load_embeddings(corpus_emb_path, corpus_ids_path)
        query_embs, query_ids = load_embeddings(query_emb_path, query_ids_path)
    else:
        from sentence_transformers import SentenceTransformer

        device = get_device("mps")
        print(f"  [SciNCL] Loading model on {device}...")
        model = SentenceTransformer("malteos/scincl", device=device)

        if not have_corpus:
            print("  [SciNCL] Encoding corpus...")
            corpus_texts = [format_text(row) for _, row in corpus_df.iterrows()]
            corpus_ids = corpus_df["doc_id"].tolist()
            corpus_embs = model.encode(corpus_texts, batch_size=config.bi_encoder_batch_size,
                                       show_progress_bar=True, normalize_embeddings=True,
                                       convert_to_numpy=True).astype(np.float32)
            emb_dir.mkdir(parents=True, exist_ok=True)
            np.save(corpus_emb_path, corpus_embs)
            with open(corpus_ids_path, "w") as f:
                json.dump(corpus_ids, f)
            print(f"  Saved corpus embeddings: {corpus_embs.shape}")
        else:
            corpus_embs, corpus_ids = load_embeddings(corpus_emb_path, corpus_ids_path)

        print(f"  [SciNCL] Encoding {tag} queries...")
        query_texts = [format_text(row) for _, row in queries_df.iterrows()]
        query_ids = queries_df["doc_id"].tolist()
        query_embs = model.encode(query_texts, batch_size=config.bi_encoder_batch_size,
                                  show_progress_bar=True, normalize_embeddings=True,
                                  convert_to_numpy=True).astype(np.float32)
        emb_dir.mkdir(parents=True, exist_ok=True)
        np.save(query_emb_path, query_embs)
        with open(query_ids_path, "w") as f:
            json.dump(query_ids, f)
        print(f"  Saved query embeddings: {query_embs.shape}")

        del model
        gc.collect()

    results = dense_retrieve(query_embs, corpus_embs, query_ids, corpus_ids,
                             top_k=config.retriever_top_k)
    save_ranked_lists(results, cache_path)
    return results


def retrieve_minilm(queries_df, corpus_df, config):
    """MiniLM-L6-v2 dense retrieval (pre-computed embeddings for public queries)."""
    tag = config.query_set
    cache_path = RESULTS_DIR / f"stage1_minilm_{tag}.json"
    if cache_path.exists():
        print(f"  [MiniLM] Loading cached results from {cache_path}")
        return load_ranked_lists(cache_path)

    pre_dir = EMB_DIR / "sentence-transformers_all-MiniLM-L6-v2"
    corpus_embs, corpus_ids = load_embeddings(
        pre_dir / "corpus_embeddings.npy", pre_dir / "corpus_ids.json")

    # Check if pre-computed query embeddings match our query set
    pre_query_embs, pre_query_ids = load_embeddings(
        pre_dir / "query_embeddings.npy", pre_dir / "query_ids.json")

    current_query_ids = queries_df["doc_id"].tolist()
    if set(pre_query_ids) == set(current_query_ids):
        # Reorder to match current query order
        id_to_idx = {qid: i for i, qid in enumerate(pre_query_ids)}
        order = [id_to_idx[qid] for qid in current_query_ids]
        query_embs = pre_query_embs[order]
        query_ids = current_query_ids
    else:
        # Need to encode new queries
        print(f"  [MiniLM] Encoding {tag} queries (not pre-computed)...")
        from sentence_transformers import SentenceTransformer
        device = get_device("mps")
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
        query_texts = [format_text(row) for _, row in queries_df.iterrows()]
        query_ids = current_query_ids
        query_embs = model.encode(query_texts, batch_size=256,
                                  show_progress_bar=True, normalize_embeddings=True,
                                  convert_to_numpy=True).astype(np.float32)
        del model
        gc.collect()

    results = dense_retrieve(query_embs, corpus_embs, query_ids, corpus_ids,
                             top_k=config.retriever_top_k)
    save_ranked_lists(results, cache_path)
    return results


def retrieve_bm25_ta(queries_df, corpus_df, config):
    """BM25 on title + abstract."""
    from rank_bm25 import BM25Okapi

    tag = config.query_set
    cache_path = RESULTS_DIR / f"stage1_bm25_ta_{tag}.json"
    if cache_path.exists():
        print(f"  [BM25-TA] Loading cached results from {cache_path}")
        return load_ranked_lists(cache_path)

    print("  [BM25-TA] Building index on title+abstract...")
    corpus_ids = corpus_df["doc_id"].tolist()
    corpus_tokens = [tokenize_simple(get_ta(row)) for _, row in tqdm(corpus_df.iterrows(),
                     total=len(corpus_df), desc="Tokenizing corpus TA")]
    bm25 = BM25Okapi(corpus_tokens)

    print(f"  [BM25-TA] Querying {len(queries_df)} queries...")
    results = {}
    for _, row in tqdm(queries_df.iterrows(), total=len(queries_df), desc="BM25-TA queries"):
        qid = row["doc_id"]
        query_tokens = tokenize_simple(get_ta(row))
        scores = bm25.get_scores(query_tokens)
        top_idx = np.argsort(-scores)[:config.retriever_top_k]
        results[qid] = [corpus_ids[j] for j in top_idx]

    del bm25, corpus_tokens
    gc.collect()
    save_ranked_lists(results, cache_path)
    return results


def retrieve_bm25_ft(queries_df, corpus_df, config):
    """BM25 on full text (with citation marker cleanup)."""
    from rank_bm25 import BM25Okapi

    tag = config.query_set
    cache_path = RESULTS_DIR / f"stage1_bm25_ft_{tag}.json"
    if cache_path.exists():
        print(f"  [BM25-FT] Loading cached results from {cache_path}")
        return load_ranked_lists(cache_path)

    print("  [BM25-FT] Building index on full text (cleaned)...")
    corpus_ids = corpus_df["doc_id"].tolist()
    corpus_tokens = [
        tokenize_simple(clean_citation_markers(str(row.get("full_text", "") or "")))
        for _, row in tqdm(corpus_df.iterrows(), total=len(corpus_df),
                          desc="Tokenizing corpus FT")
    ]
    bm25 = BM25Okapi(corpus_tokens)

    print(f"  [BM25-FT] Querying {len(queries_df)} queries (using TA as query)...")
    results = {}
    for _, row in tqdm(queries_df.iterrows(), total=len(queries_df), desc="BM25-FT queries"):
        qid = row["doc_id"]
        # Use title+abstract as query (short, focused) against full-text index
        query_tokens = tokenize_simple(get_ta(row))
        scores = bm25.get_scores(query_tokens)
        top_idx = np.argsort(-scores)[:config.retriever_top_k]
        results[qid] = [corpus_ids[j] for j in top_idx]

    del bm25, corpus_tokens
    gc.collect()
    save_ranked_lists(results, cache_path)
    return results


def retrieve_bm25_sections(queries_df, corpus_df, config):
    """Section-aware BM25: separate scores for title, abstract, and body sections."""
    from rank_bm25 import BM25Okapi

    tag = config.query_set
    cache_path = RESULTS_DIR / f"stage1_bm25_sections_{tag}.json"
    if cache_path.exists():
        print(f"  [BM25-Sections] Loading cached results from {cache_path}")
        return load_ranked_lists(cache_path)

    print("  [BM25-Sections] Building section-aware index...")
    corpus_ids = corpus_df["doc_id"].tolist()

    # Build separate BM25 indexes for title, abstract, body
    title_tokens = [tokenize_simple(str(row.get("title", "") or ""))
                    for _, row in tqdm(corpus_df.iterrows(), total=len(corpus_df), desc="Title tokens")]
    abstract_tokens = [tokenize_simple(str(row.get("abstract", "") or ""))
                       for _, row in tqdm(corpus_df.iterrows(), total=len(corpus_df), desc="Abstract tokens")]
    body_tokens = []
    for _, row in tqdm(corpus_df.iterrows(), total=len(corpus_df), desc="Body tokens"):
        body_chunks = get_body_chunks(row, min_chars=50)
        body_text = " ".join(body_chunks)
        body_tokens.append(tokenize_simple(clean_citation_markers(body_text)))

    bm25_title = BM25Okapi(title_tokens)
    bm25_abstract = BM25Okapi(abstract_tokens)
    bm25_body = BM25Okapi(body_tokens)

    # Section weights (title matches are strongest signal)
    w_title, w_abstract, w_body = 2.0, 1.5, 0.8

    print(f"  [BM25-Sections] Querying with weights: title={w_title}, abstract={w_abstract}, body={w_body}")
    results = {}
    for _, row in tqdm(queries_df.iterrows(), total=len(queries_df), desc="BM25-Sections queries"):
        qid = row["doc_id"]
        q_ta_tokens = tokenize_simple(get_ta(row))

        scores_t = bm25_title.get_scores(q_ta_tokens)
        scores_a = bm25_abstract.get_scores(q_ta_tokens)
        # For body, use TA query (full text is too slow as query)
        scores_b = bm25_body.get_scores(q_ta_tokens)

        # Normalize each to [0,1] range before combining
        for scores in [scores_t, scores_a, scores_b]:
            mx = scores.max()
            if mx > 0:
                scores /= mx

        combined = w_title * scores_t + w_abstract * scores_a + w_body * scores_b
        top_idx = np.argsort(-combined)[:config.retriever_top_k]
        results[qid] = [corpus_ids[j] for j in top_idx]

    del bm25_title, bm25_abstract, bm25_body, title_tokens, abstract_tokens, body_tokens
    gc.collect()
    save_ranked_lists(results, cache_path)
    return results


def retrieve_citation_ctx(queries_df, corpus_df, config):
    """Citation context retriever: use text around [digit] markers as BM25 queries."""
    import re
    from rank_bm25 import BM25Okapi

    tag = config.query_set
    cache_path = RESULTS_DIR / f"stage1_citation_ctx_{tag}.json"
    if cache_path.exists():
        print(f"  [CitCtx] Loading cached results from {cache_path}")
        return load_ranked_lists(cache_path)

    print("  [CitCtx] Building BM25 index on corpus TA...")
    corpus_ids = corpus_df["doc_id"].tolist()
    corpus_ta_tokens = [
        tokenize_simple(get_ta(row))
        for _, row in tqdm(corpus_df.iterrows(), total=len(corpus_df), desc="Tokenizing")
    ]
    bm25 = BM25Okapi(corpus_ta_tokens)

    pattern = r'\[(\d+(?:,\s*\d+)*)\]'

    print("  [CitCtx] Extracting citation contexts and retrieving...")
    results = {}
    for _, row in tqdm(queries_df.iterrows(), total=len(queries_df), desc="CitCtx queries"):
        qid = row["doc_id"]
        full_text = str(row.get("full_text", "") or "")

        # Extract citation context windows
        matches = list(re.finditer(pattern, full_text))
        contexts = []
        seen_starts = set()
        for m in matches:
            start_pos = max(0, m.start() - 150)
            end_pos = min(len(full_text), m.end() + 150)
            ctx = full_text[start_pos:end_pos].strip()
            ctx = re.sub(pattern, '', ctx).strip()
            approx_start = start_pos // 50
            if approx_start in seen_starts:
                continue
            seen_starts.add(approx_start)
            if len(ctx) > 30:
                contexts.append(ctx)
        contexts = contexts[:30]

        if not contexts:
            query_tokens = tokenize_simple(get_ta(row))
            scores = bm25.get_scores(query_tokens)
            top_idx = np.argsort(-scores)[:200]
            results[qid] = [corpus_ids[j] for j in top_idx]
            continue

        agg_scores = np.zeros(len(corpus_ids), dtype=np.float64)
        for ctx in contexts:
            ctx_tokens = tokenize_simple(clean_citation_markers(ctx))
            if len(ctx_tokens) < 3:
                continue
            scores = bm25.get_scores(ctx_tokens)
            mx = scores.max()
            if mx > 0:
                scores = scores / mx
            agg_scores += scores

        top_idx = np.argsort(-agg_scores)[:200]
        results[qid] = [corpus_ids[j] for j in top_idx]

    del bm25, corpus_ta_tokens
    gc.collect()
    save_ranked_lists(results, cache_path)
    return results


def fuse_rrf(all_retrievers: dict, config) -> dict:
    """Weighted Reciprocal Rank Fusion across all retrievers."""
    print(f"\n  [RRF] Fusing {len(all_retrievers)} retrievers with k={config.rrf_k}")
    print(f"  Weights: {config.rrf_weights}")

    all_qids = set()
    for rl in all_retrievers.values():
        all_qids.update(rl.keys())

    fused = {}
    for qid in all_qids:
        scores = defaultdict(float)
        for name, ranked_lists in all_retrievers.items():
            w = config.rrf_weights.get(name, 1.0)
            for rank, doc_id in enumerate(ranked_lists.get(qid, [])):
                scores[doc_id] += w / (config.rrf_k + rank + 1)
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        fused[qid] = [doc_id for doc_id, _ in sorted_docs[:config.fusion_top_k]]

    print(f"  [RRF] Fused → {len(fused)} queries, top-{config.fusion_top_k} each")
    return fused


def fuse_rrf_with_scores(all_retrievers: dict, config) -> tuple:
    """RRF fusion that also returns per-doc scores for downstream use."""
    all_qids = set()
    for rl in all_retrievers.values():
        all_qids.update(rl.keys())

    fused = {}
    fused_scores = {}  # {qid: {doc_id: score}}
    for qid in all_qids:
        scores = defaultdict(float)
        for name, ranked_lists in all_retrievers.items():
            w = config.rrf_weights.get(name, 1.0)
            for rank, doc_id in enumerate(ranked_lists.get(qid, [])):
                scores[doc_id] += w / (config.rrf_k + rank + 1)
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        fused[qid] = [doc_id for doc_id, _ in sorted_docs[:config.fusion_top_k]]
        fused_scores[qid] = dict(sorted_docs[:config.fusion_top_k])

    return fused, fused_scores


# ═══════════════════════════════════════════════════════════════
# STAGE 2: YEAR FILTER
# ═══════════════════════════════════════════════════════════════

def apply_year_filter(ranked_lists, queries_df, corpus_df, config):
    """Remove candidates published after the query paper."""
    if not config.apply_year_filter:
        return ranked_lists

    print("\n  [YEAR] Applying year filter (candidate_year <= query_year)...")
    query_years = dict(zip(queries_df["doc_id"], queries_df["year"]))
    corpus_years = dict(zip(corpus_df["doc_id"], corpus_df["year"]))

    filtered = {}
    total_removed = 0
    for qid, doc_ids in ranked_lists.items():
        q_year = query_years.get(qid, 9999)
        kept = [d for d in doc_ids if corpus_years.get(d, 0) <= q_year]
        total_removed += len(doc_ids) - len(kept)
        filtered[qid] = kept

    avg_removed = total_removed / len(ranked_lists) if ranked_lists else 0
    print(f"  [YEAR] Removed {total_removed} total ({avg_removed:.1f} avg per query)")
    return filtered


# ═══════════════════════════════════════════════════════════════
# STAGE 3: CROSS-ENCODER RERANKING
# ═══════════════════════════════════════════════════════════════

def rerank_cross_encoder(ranked_lists, queries_df, corpus_df, config):
    """Cross-encoder reranking of candidates."""
    tag = config.query_set
    model_slug = config.cross_encoder_model.replace("/", "_")
    cache_path = RESULTS_DIR / f"stage3_{model_slug}_{tag}.json"
    if cache_path.exists():
        print(f"  [CrossEncoder] Loading cached results from {cache_path}")
        return load_ranked_lists(cache_path)

    from sentence_transformers import CrossEncoder

    device = get_device(config.cross_encoder_device)
    print(f"\n  [CrossEncoder] Loading {config.cross_encoder_model} on {device}...")
    reranker = CrossEncoder(config.cross_encoder_model, device=device)

    # Build lookup dicts
    corpus_ta = dict(zip(corpus_df["doc_id"], corpus_df["ta"]))
    query_ta = dict(zip(queries_df["doc_id"], queries_df["ta"]))

    reranked = {}
    for qid in tqdm(ranked_lists, desc="Cross-encoder reranking"):
        candidates = ranked_lists[qid]
        q_text = query_ta.get(qid, "")
        pairs = [(q_text, corpus_ta.get(cid, "")) for cid in candidates]
        scores = reranker.predict(pairs, batch_size=config.cross_encoder_batch_size)
        sorted_idx = np.argsort(-scores)
        reranked[qid] = [candidates[i] for i in sorted_idx[:config.cross_encoder_top_k]]

    del reranker
    gc.collect()

    save_ranked_lists(reranked, cache_path)
    return reranked


def rerank_bge(ranked_lists, queries_df, corpus_df, config,
               model_name="BAAI/bge-reranker-large"):
    """BGE reranker using FlagEmbedding library."""
    tag = config.query_set
    model_slug = model_name.replace("/", "_")
    cache_path = RESULTS_DIR / f"stage3_{model_slug}_{tag}.json"
    if cache_path.exists():
        print(f"  [BGE] Loading cached results from {cache_path}")
        return load_ranked_lists(cache_path)

    from FlagEmbedding import FlagReranker

    print(f"\n  [BGE] Loading {model_name}...")
    reranker = FlagReranker(model_name, use_fp16=False)

    corpus_ta = dict(zip(corpus_df["doc_id"], corpus_df["ta"]))
    query_ta = dict(zip(queries_df["doc_id"], queries_df["ta"]))

    reranked = {}
    for qid in tqdm(ranked_lists, desc=f"BGE reranking ({model_name.split('/')[-1]})"):
        candidates = ranked_lists[qid]
        q_text = query_ta.get(qid, "")
        pairs = [[q_text, corpus_ta.get(cid, "")] for cid in candidates]
        scores = reranker.compute_score(pairs, normalize=True)
        if isinstance(scores, (int, float)):
            scores = [scores]
        sorted_idx = np.argsort([-s for s in scores])
        reranked[qid] = [candidates[i] for i in sorted_idx[:config.cross_encoder_top_k]]

    del reranker
    gc.collect()

    save_ranked_lists(reranked, cache_path)
    return reranked


def compare_cross_encoders(ranked_lists, queries_df, corpus_df, qrels, config):
    """Compare multiple cross-encoder models on public queries."""
    models = [
        "cross-encoder/ms-marco-MiniLM-L-12-v2",
        "Alibaba-NLP/gte-reranker-modernbert-base",
    ]

    print("\n" + "=" * 68)
    print("CROSS-ENCODER COMPARISON")
    print("=" * 68)

    best_score = 0
    best_model = ""
    results_by_model = {}

    for model_name in models:
        print(f"\n--- {model_name} ---")
        cfg_copy = PipelineConfig(
            query_set=config.query_set,
            cross_encoder_model=model_name,
            cross_encoder_batch_size=config.cross_encoder_batch_size,
            cross_encoder_top_k=config.cross_encoder_top_k,
            cross_encoder_device=config.cross_encoder_device,
        )
        reranked = rerank_cross_encoder(ranked_lists, queries_df, corpus_df, cfg_copy)
        result = evaluate(reranked, qrels)
        ndcg10 = result["overall"]["NDCG@10"]
        results_by_model[model_name] = {"ndcg10": ndcg10, "results": reranked}
        if ndcg10 > best_score:
            best_score = ndcg10
            best_model = model_name

    # Also try BGE reranker
    print(f"\n--- BAAI/bge-reranker-large ---")
    bge_results = rerank_bge(ranked_lists, queries_df, corpus_df, config)
    bge_eval = evaluate(bge_results, qrels)
    bge_ndcg = bge_eval["overall"]["NDCG@10"]
    results_by_model["BAAI/bge-reranker-large"] = {"ndcg10": bge_ndcg, "results": bge_results}
    if bge_ndcg > best_score:
        best_score = bge_ndcg
        best_model = "BAAI/bge-reranker-large"

    print(f"\n{'='*68}")
    print(f"BEST CROSS-ENCODER: {best_model} (NDCG@10 = {best_score:.4f})")
    print(f"{'='*68}")

    return best_model, results_by_model[best_model]["results"]


# ═══════════════════════════════════════════════════════════════
# STAGE 4: LLM LISTWISE RERANKING
# ═══════════════════════════════════════════════════════════════

def rerank_llm(ranked_lists, queries_df, corpus_df, config):
    """LLM listwise reranking using Claude (RankGPT-style)."""
    import anthropic
    from dotenv import load_dotenv
    load_dotenv(CHALLENGE_DIR / ".env")

    tag = config.query_set
    cache_path = RESULTS_DIR / f"stage4_llm_{tag}.json"
    if cache_path.exists():
        print(f"  [LLM] Loading cached results from {cache_path}")
        return load_ranked_lists(cache_path)

    client = anthropic.Anthropic()
    corpus_lookup = {}
    for _, row in corpus_df.iterrows():
        corpus_lookup[row["doc_id"]] = {
            "title": str(row.get("title", "") or "").strip(),
            "abstract": str(row.get("abstract", "") or "").strip()[:300],
        }

    query_lookup = {}
    for _, row in queries_df.iterrows():
        query_lookup[row["doc_id"]] = {
            "title": str(row.get("title", "") or "").strip(),
            "abstract": str(row.get("abstract", "") or "").strip()[:500],
        }

    reranked = {}
    n = config.llm_top_k
    errors = 0

    for qid in tqdm(ranked_lists, desc="LLM reranking"):
        candidates = ranked_lists[qid][:n]
        remaining = ranked_lists[qid][n:]

        q = query_lookup.get(qid, {})
        cand_str = ""
        for i, cid in enumerate(candidates):
            c = corpus_lookup.get(cid, {})
            cand_str += f"[{i+1}] {c.get('title', 'N/A')}\n{c.get('abstract', 'N/A')}\n\n"

        prompt = f"""You are an expert scientific researcher. Given a query paper, rank the candidate papers by how likely the query paper would CITE each one.

Consider: methodological dependencies, foundational concepts the query builds on, direct comparisons, and domain relevance.

QUERY PAPER:
Title: {q.get('title', 'N/A')}
Abstract: {q.get('abstract', 'N/A')}

CANDIDATE PAPERS (rank ALL {n} from most to least likely to be cited):
{cand_str}
Return ONLY the numbers in descending relevance order, separated by " > ". Example: 3 > 1 > 5 > 2 > 4 > ..."""

        try:
            response = client.messages.create(
                model=config.llm_model,
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}]
            )
            text = response.content[0].text.strip()  # type: ignore[union-attr]
            # Parse: "3 > 1 > 5 > ..."
            parts = [p.strip() for p in text.split(">")]
            order = []
            seen = set()
            for p in parts:
                try:
                    idx = int(p) - 1  # 1-indexed to 0-indexed
                    if 0 <= idx < n and idx not in seen:
                        order.append(idx)
                        seen.add(idx)
                except ValueError:
                    continue
            # Add any missing indices at the end
            for i in range(n):
                if i not in seen:
                    order.append(i)

            reranked_top = [candidates[i] for i in order]
            reranked[qid] = reranked_top + remaining

        except Exception as e:
            print(f"  [LLM] Error for {qid}: {e}")
            errors += 1
            reranked[qid] = ranked_lists[qid]  # Keep original order

    if errors:
        print(f"  [LLM] {errors} errors (fell back to cross-encoder order)")

    save_ranked_lists(reranked, cache_path)
    return reranked


# ═══════════════════════════════════════════════════════════════
# STAGE 5: ASSEMBLY & SUBMISSION
# ═══════════════════════════════════════════════════════════════

def assemble_submission(ranked_lists, config, label="submission"):
    """Trim to top 100 and save as submission."""
    submission = {}
    for qid, docs in ranked_lists.items():
        submission[qid] = docs[:100]

    out_json = SUBMISSIONS_DIR / f"{label}_data.json"
    out_zip = SUBMISSIONS_DIR / f"{label}.zip"

    # Save JSON (must be named submission_data.json inside zip)
    tmp_json = SUBMISSIONS_DIR / "submission_data.json"
    with open(tmp_json, "w") as f:
        json.dump(submission, f)

    # Create zip
    subprocess.run(["zip", "-j", str(out_zip), str(tmp_json)],
                   capture_output=True, check=True)

    # Also keep a copy with the label
    if str(out_json) != str(tmp_json):
        with open(out_json, "w") as f:
            json.dump(submission, f)

    print(f"\n  Submission saved:")
    print(f"    JSON: {out_json}")
    print(f"    ZIP:  {out_zip}")
    print(f"    Queries: {len(submission)}")
    print(f"    Docs per query: {len(next(iter(submission.values())))}")
    return submission


# ═══════════════════════════════════════════════════════════════
# WEIGHT TUNING
# ═══════════════════════════════════════════════════════════════

def tune_rrf_weights(all_retrievers, qrels, queries_df, corpus_df):
    """Grid search over RRF weights using public qrels."""
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
    from itertools import product
    keys = sorted(active_grid.keys())
    all_combos = list(product(*[active_grid[k] for k in keys]))
    total = len(all_combos) * len(k_values)
    print(f"  Testing {total} combinations ({len(all_combos)} weight combos × {len(k_values)} k values)...")

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


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="IR Challenge Pipeline")
    parser.add_argument("--query-set", default="public",
                        choices=["public", "held_out"],
                        help="Which query set to use")
    parser.add_argument("--stages", default="all",
                        help="Comma-separated stage numbers, 'all', or 'tune'")
    parser.add_argument("--cross-encoder", default=None,
                        help="Override cross-encoder model")
    args = parser.parse_args()

    config = PipelineConfig(query_set=args.query_set)
    if args.cross_encoder:
        config.cross_encoder_model = args.cross_encoder

    stages = args.stages
    if stages == "all":
        run_stages = {1, 2, 3, 4, 5}
    elif stages == "tune":
        run_stages = {"tune"}
    else:
        run_stages = set(int(s) for s in stages.split(","))

    # Load data
    print("=" * 68)
    print(f"IR CHALLENGE PIPELINE — Query set: {config.query_set}")
    print("=" * 68)

    print("\nLoading data...")
    corpus_df = load_corpus(DATA_DIR / "corpus.parquet")
    if config.query_set == "public":
        queries_df = load_queries(DATA_DIR / "queries.parquet")
        qrels = load_qrels(DATA_DIR / "qrels.json")
    else:
        queries_df = load_queries(DATA_DIR / "held_out_queries.parquet")
        qrels = {}
    print(f"  Corpus: {len(corpus_df)} docs | Queries: {len(queries_df)} | Qrels: {len(qrels)}")

    query_domains = dict(zip(queries_df["doc_id"], queries_df["domain"])) if "domain" in queries_df.columns else None

    # ── Stage 1: Retrieval ──
    all_retrievers = {}
    if 1 in run_stages or "tune" in run_stages:
        print("\n" + "=" * 68)
        print("STAGE 1: MULTI-SIGNAL RETRIEVAL")
        print("=" * 68)

        print("\n[1/6] SPECTER2 Asymmetric...")
        all_retrievers["specter2"] = retrieve_specter2(queries_df, corpus_df, config)
        eval_if_public(all_retrievers["specter2"], qrels, config, "SPECTER2 only")

        print("\n[2/6] SciNCL...")
        all_retrievers["scincl"] = retrieve_scincl(queries_df, corpus_df, config)
        eval_if_public(all_retrievers["scincl"], qrels, config, "SciNCL only")

        print("\n[3/6] MiniLM-L6-v2...")
        all_retrievers["minilm"] = retrieve_minilm(queries_df, corpus_df, config)
        eval_if_public(all_retrievers["minilm"], qrels, config, "MiniLM only")

        print("\n[4/6] BM25 on title+abstract...")
        all_retrievers["bm25_ta"] = retrieve_bm25_ta(queries_df, corpus_df, config)
        eval_if_public(all_retrievers["bm25_ta"], qrels, config, "BM25-TA only")

        print("\n[5/6] BM25 on full text...")
        all_retrievers["bm25_ft"] = retrieve_bm25_ft(queries_df, corpus_df, config)
        eval_if_public(all_retrievers["bm25_ft"], qrels, config, "BM25-FT only")

        print("\n[6/6] BM25 section-aware...")
        all_retrievers["bm25_sections"] = retrieve_bm25_sections(queries_df, corpus_df, config)
        eval_if_public(all_retrievers["bm25_sections"], qrels, config, "BM25-Sections only")

    # ── Weight tuning ──
    if "tune" in run_stages:
        best = tune_rrf_weights(all_retrievers, qrels, queries_df, corpus_df)
        print("\nApply these weights to PipelineConfig and re-run.")
        return

    # ── RRF Fusion ──
    if 1 in run_stages:
        fused = fuse_rrf(all_retrievers, config)
        save_ranked_lists(fused, RESULTS_DIR / f"stage1_fused_{config.query_set}.json")
        eval_if_public(fused, qrels, config, "RRF Fusion (pre-filter)")

    # ── Stage 2: Year filter ──
    if 2 in run_stages:
        print("\n" + "=" * 68)
        print("STAGE 2: YEAR FILTER")
        print("=" * 68)
        if 1 not in run_stages:
            fused = load_ranked_lists(RESULTS_DIR / f"stage1_fused_{config.query_set}.json")
        fused = apply_year_filter(fused, queries_df, corpus_df, config)
        save_ranked_lists(fused, RESULTS_DIR / f"stage2_filtered_{config.query_set}.json")
        eval_if_public(fused, qrels, config, "After year filter")

    # ── Stage 3: Cross-encoder ──
    if 3 in run_stages:
        print("\n" + "=" * 68)
        print("STAGE 3: CROSS-ENCODER RERANKING")
        print("=" * 68)
        if 2 not in run_stages:
            fused = load_ranked_lists(RESULTS_DIR / f"stage2_filtered_{config.query_set}.json")
        reranked = rerank_cross_encoder(fused, queries_df, corpus_df, config)
        eval_if_public(reranked, qrels, config, f"Cross-encoder: {config.cross_encoder_model}")

    # ── Stage 4: LLM reranking ──
    if 4 in run_stages:
        print("\n" + "=" * 68)
        print("STAGE 4: LLM LISTWISE RERANKING")
        print("=" * 68)
        if 3 not in run_stages:
            model_slug = config.cross_encoder_model.replace("/", "_")
            reranked = load_ranked_lists(
                RESULTS_DIR / f"stage3_{model_slug}_{config.query_set}.json")
        final = rerank_llm(reranked, queries_df, corpus_df, config)
        eval_if_public(final, qrels, config, "After LLM reranking")

    # ── Stage 5: Submission ──
    if 5 in run_stages:
        print("\n" + "=" * 68)
        print("STAGE 5: ASSEMBLY & SUBMISSION")
        print("=" * 68)
        if 4 not in run_stages:
            # Load best available results
            tag = config.query_set
            llm_path = RESULTS_DIR / f"stage4_llm_{tag}.json"
            model_slug = config.cross_encoder_model.replace("/", "_")
            ce_path = RESULTS_DIR / f"stage3_{model_slug}_{tag}.json"
            if llm_path.exists():
                final = load_ranked_lists(llm_path)
            elif ce_path.exists():
                final = load_ranked_lists(ce_path)
            else:
                fused_path = RESULTS_DIR / f"stage2_filtered_{tag}.json"
                final = load_ranked_lists(fused_path)

        label = f"submission_{config.query_set}"
        submission = assemble_submission(final, config, label=label)
        if config.query_set == "public" and qrels:
            print("\nFinal evaluation:")
            evaluate(submission, qrels, query_domains=query_domains)

    print("\n" + "=" * 68)
    print("PIPELINE COMPLETE")
    print("=" * 68)


if __name__ == "__main__":
    main()

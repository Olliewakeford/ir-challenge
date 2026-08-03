"""Stage 3: cross-encoder and BGE reranking.

Kept for reproducibility of the "what didn't work" finding: cross-encoder
reranking (ms-marco-MiniLM-L-12-v2, and a fine-tuned variant) dropped NDCG@10
from 0.5918 to 0.4762. Not part of the submitted pipeline.
"""
import gc

import numpy as np
from tqdm.auto import tqdm

from .config import PipelineConfig
from .device import get_device
from .paths import RESULTS_DIR
from .storage import load_ranked_lists, save_ranked_lists


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
    from .metrics import evaluate

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
    print("\n--- BAAI/bge-reranker-large ---")
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

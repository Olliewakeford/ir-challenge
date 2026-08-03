#!/usr/bin/env python3
"""
Try E5-large-v2 (1024d) as an additional dense retrieval signal.
E5 models require "query: " and "passage: " prefixes for asymmetric retrieval.
"""
import sys
import gc
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from irchallenge.metrics import evaluate
from irchallenge.paths import DATA_DIR, EMB_DIR, RESULTS_DIR
from irchallenge.storage import load_corpus, load_qrels, load_queries, load_ranked_lists, save_ranked_lists
from irchallenge.text import get_ta
from tqdm.auto import tqdm


def embed_e5(texts, model, tokenizer, prefix="passage: ", batch_size=64, device="mps"):
    """Encode texts with E5 model."""
    import torch

    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc=f"Embedding ({prefix.strip()})"):
        batch = [prefix + t[:512] for t in texts[i:i+batch_size]]
        encoded = tokenizer(batch, padding=True, truncation=True, max_length=512,
                           return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**encoded)
            # Average pooling
            attention_mask = encoded["attention_mask"]
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                input_mask_expanded.sum(1), min=1e-9)
            # Normalize
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu().numpy())

    return np.vstack(all_embeddings)


def main():
    start_time = time.time()

    cache_path = RESULTS_DIR / "stage1_e5large_public.json"
    if cache_path.exists():
        print(f"Loading cached E5-large results from {cache_path}")
        results = load_ranked_lists(cache_path)
        qrels = load_qrels(DATA_DIR / "qrels.json")
        r = evaluate(results, qrels, verbose=False)
        print(f"E5-large: NDCG@10={r['overall']['NDCG@10']:.4f}  "
              f"Recall@100={r['overall']['Recall@100']:.4f}")
        return

    print("=" * 68)
    print("E5-LARGE-V2 DENSE RETRIEVAL")
    print("=" * 68)

    # Load data
    corpus_df = load_corpus(DATA_DIR / "corpus.parquet")
    queries_df = load_queries(DATA_DIR / "queries.parquet")
    qrels = load_qrels(DATA_DIR / "qrels.json")

    # Prepare texts (title + abstract)
    corpus_ids = corpus_df["doc_id"].tolist()
    corpus_texts = [get_ta(row) for _, row in corpus_df.iterrows()]
    query_ids = queries_df["doc_id"].tolist()
    query_texts = [get_ta(row) for _, row in queries_df.iterrows()]

    # Check for cached embeddings
    emb_dir = EMB_DIR / "e5-large-v2"
    corpus_emb_path = emb_dir / "corpus.npy"
    query_emb_path = emb_dir / "queries_public.npy"

    if corpus_emb_path.exists() and query_emb_path.exists():
        print("  Loading cached E5-large embeddings...")
        corpus_emb = np.load(corpus_emb_path)
        query_emb = np.load(query_emb_path)
    else:
        emb_dir.mkdir(parents=True, exist_ok=True)

        from transformers import AutoTokenizer, AutoModel
        import torch

        print("\n  Loading E5-large-v2 model...")
        tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-large-v2")
        model = AutoModel.from_pretrained("intfloat/e5-large-v2")

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model = model.to(device)
        model.eval()
        print(f"  Device: {device}")

        # Encode corpus
        print("\n  Encoding corpus (20K papers)...")
        corpus_emb = embed_e5(corpus_texts, model, tokenizer, prefix="passage: ",
                             batch_size=32, device=device)
        np.save(corpus_emb_path, corpus_emb)
        print(f"  Corpus embeddings: {corpus_emb.shape}")

        # Encode queries
        print("\n  Encoding queries...")
        query_emb = embed_e5(query_texts, model, tokenizer, prefix="query: ",
                            batch_size=32, device=device)
        np.save(query_emb_path, query_emb)
        print(f"  Query embeddings: {query_emb.shape}")

        del model, tokenizer
        gc.collect()

    # Compute similarities and rank
    print("\n  Computing similarities...")
    from sklearn.metrics.pairwise import cosine_similarity

    sims = cosine_similarity(query_emb, corpus_emb)

    results = {}
    for i, qid in enumerate(query_ids):
        top_idx = sims[i].argsort()[::-1][:300]
        results[qid] = [corpus_ids[k] for k in top_idx if corpus_ids[k] != qid][:300]

    save_ranked_lists(results, cache_path)

    r = evaluate(results, qrels, verbose=False)
    print(f"\n  E5-large: NDCG@10={r['overall']['NDCG@10']:.4f}  "
          f"Recall@100={r['overall']['Recall@100']:.4f}  MAP={r['overall']['MAP']:.4f}")

    elapsed = time.time() - start_time
    print(f"\n  Time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()

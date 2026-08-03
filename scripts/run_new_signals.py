#!/usr/bin/env python3
"""
Try new signals to improve fusion:
1. TF-IDF cosine similarity on full text (different from BM25)
2. Domain boost (97.6% of citations are same-domain)
3. Title-only BM25 (titles carry strong signal for citation)
4. Abstract bigram overlap

Run each, save as a retriever signal, then re-fuse with all signals.
"""
import sys
import re
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from irchallenge.fusion import weighted_rrf_fuse as fuse_rrf
from irchallenge.metrics import evaluate
from irchallenge.paths import DATA_DIR, RESULTS_DIR
from irchallenge.storage import load_corpus, load_qrels, load_queries, load_ranked_lists, save_ranked_lists
from irchallenge.text import get_ta
from tqdm.auto import tqdm


def clean_text(text):
    return re.sub(r'\[\d+(?:,\s*\d+)*\]', '', str(text))


def run_tfidf_fulltext(queries_df, corpus_df, config_tag="public"):
    """TF-IDF cosine similarity on full text — different signal from BM25."""
    cache_path = RESULTS_DIR / f"stage1_tfidf_ft_{config_tag}.json"
    if cache_path.exists():
        print(f"  Loading cached TF-IDF results from {cache_path}")
        return load_ranked_lists(cache_path)

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    print("\n  Building TF-IDF vectors on full text...")

    # Combine corpus and query texts
    corpus_ids = corpus_df["doc_id"].tolist()
    corpus_texts = [clean_text(row["full_text"])[:10000] for _, row in corpus_df.iterrows()]

    query_ids = queries_df["doc_id"].tolist()
    query_texts = [clean_text(row["full_text"])[:10000] for _, row in queries_df.iterrows()]

    # Fit TF-IDF on corpus
    print("  Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=50000,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        ngram_range=(1, 2),
    )
    corpus_tfidf = vectorizer.fit_transform(corpus_texts)
    query_tfidf = vectorizer.transform(query_texts)

    print(f"  Vocab size: {len(vectorizer.vocabulary_)} | Matrix: {corpus_tfidf.shape}")

    # Compute similarities
    results = {}
    batch_size = 10
    for i in tqdm(range(0, len(query_ids), batch_size), desc="TF-IDF similarity"):
        batch_queries = query_tfidf[i:i+batch_size]
        sims = cosine_similarity(batch_queries, corpus_tfidf)

        for j in range(sims.shape[0]):
            qid = query_ids[i + j]
            top_idx = sims[j].argsort()[::-1][:300]
            results[qid] = [corpus_ids[k] for k in top_idx if corpus_ids[k] != qid][:300]

    save_ranked_lists(results, cache_path)
    return results


def run_title_bm25(queries_df, corpus_df, config_tag="public"):
    """BM25 on title only — titles carry strong citation signal."""
    cache_path = RESULTS_DIR / f"stage1_bm25_title_{config_tag}.json"
    if cache_path.exists():
        print(f"  Loading cached title BM25 from {cache_path}")
        return load_ranked_lists(cache_path)

    from rank_bm25 import BM25Okapi

    print("\n  Building BM25 on titles only...")
    corpus_ids = corpus_df["doc_id"].tolist()
    corpus_titles = [re.findall(r'\w+', str(row.get("title", "")).lower())
                     for _, row in corpus_df.iterrows()]

    bm25 = BM25Okapi(corpus_titles)

    results = {}
    for _, row in tqdm(queries_df.iterrows(), total=len(queries_df), desc="Title BM25"):
        qid = row["doc_id"]
        q_title = re.findall(r'\w+', str(row.get("title", "")).lower())
        scores = bm25.get_scores(q_title)
        top_idx = scores.argsort()[::-1][:300]
        results[qid] = [corpus_ids[k] for k in top_idx if corpus_ids[k] != qid][:300]

    save_ranked_lists(results, cache_path)
    return results


def run_abstract_overlap(queries_df, corpus_df, config_tag="public"):
    """Jaccard overlap of abstract keywords — simple but effective for citations."""
    cache_path = RESULTS_DIR / f"stage1_abstract_overlap_{config_tag}.json"
    if cache_path.exists():
        print(f"  Loading cached abstract overlap from {cache_path}")
        return load_ranked_lists(cache_path)

    print("\n  Computing abstract keyword overlap...")

    # Build keyword sets for corpus
    corpus_ids = corpus_df["doc_id"].tolist()
    corpus_keywords = []
    for _, row in corpus_df.iterrows():
        abstract = clean_text(str(row.get("abstract", "")))
        words = set(re.findall(r'\b[a-z]{3,}\b', abstract.lower()))
        corpus_keywords.append(words)

    results = {}
    for _, row in tqdm(queries_df.iterrows(), total=len(queries_df), desc="Abstract overlap"):
        qid = row["doc_id"]
        q_abstract = clean_text(str(row.get("abstract", "")))
        q_words = set(re.findall(r'\b[a-z]{3,}\b', q_abstract.lower()))

        if not q_words:
            results[qid] = []
            continue

        scores = []
        for i, c_words in enumerate(corpus_keywords):
            if not c_words:
                scores.append(0.0)
                continue
            intersection = len(q_words & c_words)
            union = len(q_words | c_words)
            scores.append(intersection / union if union > 0 else 0.0)

        scores = np.array(scores)
        top_idx = scores.argsort()[::-1][:300]
        results[qid] = [corpus_ids[k] for k in top_idx if corpus_ids[k] != qid][:300]

    save_ranked_lists(results, cache_path)
    return results


def main():
    start = time.time()

    corpus_df = load_corpus(DATA_DIR / "corpus.parquet")
    queries_df = load_queries(DATA_DIR / "queries.parquet")
    qrels = load_qrels(DATA_DIR / "qrels.json")

    # Load existing retrievers
    retrievers = {}
    for name in ["specter2", "scincl", "minilm", "bm25_ta", "bm25_ft",
                  "bm25_sections", "citation_ctx"]:
        path = RESULTS_DIR / f"stage1_{name}_public.json"
        if path.exists():
            retrievers[name] = load_ranked_lists(path)

    # V2 baseline
    v2_weights = {
        "bm25_ft": 2.5, "bm25_sections": 0.2, "bm25_ta": 0.15,
        "citation_ctx": 0.15, "minilm": 1.65, "scincl": 0.7, "specter2": 0.3,
    }
    fused_v2 = fuse_rrf(retrievers, v2_weights, 10)
    r = evaluate(fused_v2, qrels, verbose=False)
    baseline = r['overall']['NDCG@10']
    print(f"\nBaseline v2 (7 retrievers): NDCG@10={baseline:.4f}")

    # Run new signals
    print("\n" + "=" * 68)
    print("RUNNING NEW RETRIEVAL SIGNALS")
    print("=" * 68)

    new_retrievers = {}

    # 1. TF-IDF full text
    print("\n--- TF-IDF Full Text ---")
    tfidf_results = run_tfidf_fulltext(queries_df, corpus_df)
    r = evaluate(tfidf_results, qrels, verbose=False)
    print(f"  TF-IDF standalone: NDCG@10={r['overall']['NDCG@10']:.4f}  "
          f"Recall@100={r['overall']['Recall@100']:.4f}")
    new_retrievers["tfidf_ft"] = tfidf_results

    # 2. Title BM25
    print("\n--- Title BM25 ---")
    title_results = run_title_bm25(queries_df, corpus_df)
    r = evaluate(title_results, qrels, verbose=False)
    print(f"  Title BM25 standalone: NDCG@10={r['overall']['NDCG@10']:.4f}  "
          f"Recall@100={r['overall']['Recall@100']:.4f}")
    new_retrievers["bm25_title"] = title_results

    # 3. Abstract overlap
    print("\n--- Abstract Keyword Overlap ---")
    overlap_results = run_abstract_overlap(queries_df, corpus_df)
    r = evaluate(overlap_results, qrels, verbose=False)
    print(f"  Abstract overlap standalone: NDCG@10={r['overall']['NDCG@10']:.4f}  "
          f"Recall@100={r['overall']['Recall@100']:.4f}")
    new_retrievers["abstract_overlap"] = overlap_results

    # Test each new signal added to v2 fusion
    print("\n" + "=" * 68)
    print("TESTING NEW SIGNALS IN FUSION")
    print("=" * 68)

    best_overall = baseline
    best_config = None

    for new_name, new_data in new_retrievers.items():
        print(f"\n  Testing {new_name}...")
        test_retrievers = dict(retrievers)
        test_retrievers[new_name] = new_data

        for w in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5]:
            weights = dict(v2_weights)
            weights[new_name] = w
            for k in [8, 10, 12]:
                fused = fuse_rrf(test_retrievers, weights, k)
                r = evaluate(fused, qrels, verbose=False)
                ndcg = r['overall']['NDCG@10']
                recall = r['overall']['Recall@100']
                if ndcg > best_overall:
                    best_overall = ndcg
                    best_config = (new_name, w, k)
                    print(f"    {new_name}={w}, k={k}: NDCG@10={ndcg:.4f} Recall@100={recall:.4f} ↑")

    if best_config:
        print(f"\n  Best new signal: {best_config[0]} w={best_config[1]} k={best_config[2]}")
        print(f"  NDCG@10: {baseline:.4f} → {best_overall:.4f} (+{best_overall-baseline:.4f})")

        # Save improved fusion
        test_retrievers = dict(retrievers)
        test_retrievers[best_config[0]] = new_retrievers[best_config[0]]
        weights = dict(v2_weights)
        weights[best_config[0]] = best_config[1]
        fused_improved = fuse_rrf(test_retrievers, weights, best_config[2])
        save_ranked_lists(fused_improved, RESULTS_DIR / "stage1_fused_v4_public.json")
    else:
        print(f"\n  No new signal improved over baseline {baseline:.4f}")

    # Also test: all new signals together
    print(f"\n{'='*68}")
    print("TESTING ALL NEW SIGNALS TOGETHER")
    print("=" * 68)
    all_retrievers = dict(retrievers)
    all_retrievers.update(new_retrievers)

    # Quick grid over new signal weights
    best_combo = baseline
    for tfidf_w in [0.0, 0.1, 0.3, 0.5]:
        for title_w in [0.0, 0.1, 0.3, 0.5]:
            for overlap_w in [0.0, 0.1, 0.3]:
                if tfidf_w == 0 and title_w == 0 and overlap_w == 0:
                    continue
                weights = dict(v2_weights)
                weights["tfidf_ft"] = tfidf_w
                weights["bm25_title"] = title_w
                weights["abstract_overlap"] = overlap_w
                fused = fuse_rrf(all_retrievers, weights, 10)
                r = evaluate(fused, qrels, verbose=False)
                ndcg = r['overall']['NDCG@10']
                if ndcg > best_combo:
                    best_combo = ndcg
                    print(f"  tfidf={tfidf_w} title={title_w} overlap={overlap_w}: "
                          f"NDCG@10={ndcg:.4f} ↑")

    print(f"\n  Best combo NDCG@10: {best_combo:.4f} (baseline: {baseline:.4f})")
    print(f"\n  Time: {time.time()-start:.1f}s")


if __name__ == "__main__":
    main()

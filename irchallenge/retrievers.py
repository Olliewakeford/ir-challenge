"""The retrieval signals: two dense encoders, one asymmetric adapter model,
three BM25 variants, a citation-context retriever, and TF-IDF.

Each `retrieve_*` function returns {qid: [doc_ids ranked]} and caches its
result to RESULTS_DIR so a rerun reuses it instead of recomputing.
"""
import gc
import re

import numpy as np
from tqdm.auto import tqdm

from .device import get_device
from .paths import EMB_DIR, RESULTS_DIR
from .storage import load_embeddings, load_ranked_lists, save_ranked_lists
from .text import clean_citation_markers, format_text, get_body_chunks, get_ta, tokenize_simple


def dense_retrieve(query_embs, corpus_embs, query_ids, corpus_ids, top_k=200):
    """Compute cosine similarity (assumes L2-normalized) and return top-k per query."""
    scores = query_embs @ corpus_embs.T  # (n_queries, n_corpus)
    results = {}
    for i, qid in enumerate(query_ids):
        top_idx = np.argsort(-scores[i])[:top_k]
        results[qid] = [corpus_ids[j] for j in top_idx]
    return results


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
                import json
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
            import json
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
                import json
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
            import json
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


def retrieve_tfidf_ft(queries_df, corpus_df, config):
    """TF-IDF cosine similarity on full text. The eighth signal, added in v4."""
    tag = config.query_set
    cache_path = RESULTS_DIR / f"stage1_tfidf_ft_{tag}.json"
    if cache_path.exists():
        print(f"  [TF-IDF] Loading cached results from {cache_path}")
        return load_ranked_lists(cache_path)

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    corpus_ids = corpus_df["doc_id"].tolist()
    corpus_texts = [clean_citation_markers(str(row["full_text"]))[:10000]
                    for _, row in corpus_df.iterrows()]
    query_ids = queries_df["doc_id"].tolist()
    query_texts = [clean_citation_markers(str(row["full_text"]))[:10000]
                   for _, row in queries_df.iterrows()]

    print("  [TF-IDF] Fitting vectorizer on full text...")
    vectorizer = TfidfVectorizer(
        max_features=50000, min_df=2, max_df=0.95,
        sublinear_tf=True, ngram_range=(1, 2),
    )
    corpus_tfidf = vectorizer.fit_transform(corpus_texts)
    query_tfidf = vectorizer.transform(query_texts)

    results = {}
    batch_size = 10
    for i in tqdm(range(0, len(query_ids), batch_size), desc="TF-IDF similarity"):
        batch_queries = query_tfidf[i:i + batch_size]
        sims = cosine_similarity(batch_queries, corpus_tfidf)
        for j in range(sims.shape[0]):
            qid = query_ids[i + j]
            top_idx = sims[j].argsort()[::-1][:300]
            results[qid] = [corpus_ids[k] for k in top_idx if corpus_ids[k] != qid][:300]

    save_ranked_lists(results, cache_path)
    return results

#!/usr/bin/env python3
"""
Fine-tune a cross-encoder on our ~740 labeled citation pairs.

Web-search cross-encoders DESTROY citation prediction (0.5760 → 0.2342).
But fine-tuning on actual citation data could create a model that HELPS.

Approach:
1. Generate training pairs from public qrels (positive = cited, negative = top-ranked non-cited)
2. Fine-tune ms-marco-MiniLM-L-12-v2 for 2-3 epochs with citation-specific data
3. Evaluate on public queries (leave-one-out or full set since we're evaluating held-out)
4. Use the fine-tuned model to rerank fused results

Uses sentence-transformers CrossEncoder training API.
"""
import sys
import gc
import json
import time
import random
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from irchallenge.metrics import evaluate
from irchallenge.paths import DATA_DIR, RESULTS_DIR
from irchallenge.storage import load_corpus, load_qrels, load_queries, load_ranked_lists, save_ranked_lists
from irchallenge.text import get_ta
from tqdm.auto import tqdm


def generate_training_data(queries_df, corpus_df, qrels, fused_results):
    """Generate training pairs for cross-encoder fine-tuning.

    For each query:
    - Positives: all cited docs from qrels
    - Hard negatives: top-ranked non-cited docs from fused results (harder to distinguish)
    """
    query_lookup = {row["doc_id"]: get_ta(row) for _, row in queries_df.iterrows()}
    corpus_lookup = {row["doc_id"]: get_ta(row) for _, row in corpus_df.iterrows()}

    pairs = []  # (query_text, candidate_text, label)

    for qid, relevant_docs in qrels.items():
        q_text = query_lookup.get(qid, "")
        if not q_text:
            continue

        relevant_set = set(relevant_docs)

        # Positive pairs
        for doc_id in relevant_docs:
            c_text = corpus_lookup.get(doc_id, "")
            if c_text:
                pairs.append((q_text, c_text, 1))

        # Hard negative pairs: top-ranked non-relevant docs from fused results
        if qid in fused_results:
            neg_count = 0
            for doc_id in fused_results[qid]:
                if doc_id not in relevant_set:
                    c_text = corpus_lookup.get(doc_id, "")
                    if c_text:
                        pairs.append((q_text, c_text, 0))
                        neg_count += 1
                        if neg_count >= len(relevant_docs) * 2:  # 2:1 negative ratio
                            break

    random.seed(42)
    random.shuffle(pairs)
    return pairs


def finetune_cross_encoder(pairs, output_dir, base_model="cross-encoder/ms-marco-MiniLM-L-12-v2",
                           epochs=3, batch_size=32):
    """Fine-tune a cross-encoder on citation pairs."""
    from sentence_transformers import CrossEncoder
    from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
    from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments
    from datasets import Dataset

    print(f"\n  Fine-tuning {base_model} on {len(pairs)} pairs...")
    print(f"  Positives: {sum(1 for _, _, l in pairs if l == 1)}")
    print(f"  Negatives: {sum(1 for _, _, l in pairs if l == 0)}")

    # Create dataset
    dataset_dict = {
        "sentence1": [p[0][:512] for p in pairs],
        "sentence2": [p[1][:512] for p in pairs],
        "label": [float(p[2]) for p in pairs],
    }
    dataset = Dataset.from_dict(dataset_dict)

    # Split train/eval (90/10)
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    print(f"  Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")

    # Load model
    model = CrossEncoder(base_model, device="mps", num_labels=1)

    # Training arguments
    output_dir = str(output_dir)
    args = CrossEncoderTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=2,
        logging_steps=10,
        seed=42,
    )

    trainer = CrossEncoderTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()
    model.save_pretrained(output_dir + "/best")

    return model


def rerank_with_finetuned(ranked_lists, queries_df, corpus_df, model, top_k=100):
    """Rerank using the fine-tuned cross-encoder."""
    query_lookup = {row["doc_id"]: get_ta(row) for _, row in queries_df.iterrows()}
    corpus_lookup = {row["doc_id"]: get_ta(row) for _, row in corpus_df.iterrows()}

    reranked = {}
    for qid in tqdm(ranked_lists, desc="Fine-tuned reranking"):
        q_text = query_lookup.get(qid, "")
        candidates = ranked_lists[qid][:top_k]
        c_texts = [corpus_lookup.get(cid, "") for cid in candidates]

        # Score all pairs
        pairs = [[q_text[:512], ct[:512]] for ct in c_texts]
        scores = model.predict(pairs, show_progress_bar=False)

        # Sort by score
        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        # Reranked top + remaining
        reranked_ids = [d for d, _ in scored]
        covered = set(reranked_ids)
        for doc_id in ranked_lists[qid]:
            if doc_id not in covered:
                reranked_ids.append(doc_id)

        reranked[qid] = reranked_ids[:100]

    return reranked


def main():
    start_time = time.time()

    cache_path = RESULTS_DIR / "stage3_finetuned_reranker_public.json"
    if cache_path.exists():
        print(f"Loading cached results from {cache_path}")
        results = load_ranked_lists(cache_path)
        qrels = load_qrels(DATA_DIR / "qrels.json")
        r = evaluate(results, qrels, verbose=False)
        print(f"Fine-tuned reranker: NDCG@10={r['overall']['NDCG@10']:.4f}  "
              f"Recall@100={r['overall']['Recall@100']:.4f}  MAP={r['overall']['MAP']:.4f}")
        return

    print("=" * 68)
    print("FINE-TUNING CITATION CROSS-ENCODER")
    print("=" * 68)

    # Load data
    print("\nLoading data...")
    corpus_df = load_corpus(DATA_DIR / "corpus.parquet")
    queries_df = load_queries(DATA_DIR / "queries.parquet")
    qrels = load_qrels(DATA_DIR / "qrels.json")

    # Load fused v2 results for hard negatives
    fused_path = RESULTS_DIR / "stage1_fused_v2_public.json"
    if not fused_path.exists():
        fused_path = RESULTS_DIR / "stage1_fused_public.json"
    fused = load_ranked_lists(fused_path)

    # Evaluate baseline
    r_base = evaluate(fused, qrels, verbose=False)
    print(f"\n  Baseline fused: NDCG@10={r_base['overall']['NDCG@10']:.4f}  "
          f"Recall@100={r_base['overall']['Recall@100']:.4f}")

    # Generate training data
    print("\nGenerating training data...")
    pairs = generate_training_data(queries_df, corpus_df, qrels, fused)
    print(f"  Total pairs: {len(pairs)}")

    # Fine-tune
    model_dir = Path(RESULTS_DIR).parent / "models" / "citation_reranker"
    model_dir.mkdir(parents=True, exist_ok=True)

    model = finetune_cross_encoder(
        pairs, model_dir,
        base_model="cross-encoder/ms-marco-MiniLM-L-12-v2",
        epochs=3, batch_size=32
    )

    # Rerank
    print("\n" + "=" * 68)
    print("RERANKING WITH FINE-TUNED MODEL")
    print("=" * 68)
    reranked = rerank_with_finetuned(fused, queries_df, corpus_df, model, top_k=100)

    # Evaluate
    r_reranked = evaluate(reranked, qrels, verbose=False)
    print(f"\n  After fine-tuned reranking:")
    print(f"    NDCG@10={r_reranked['overall']['NDCG@10']:.4f}  "
          f"Recall@100={r_reranked['overall']['Recall@100']:.4f}  MAP={r_reranked['overall']['MAP']:.4f}")

    improvement = r_reranked['overall']['NDCG@10'] - r_base['overall']['NDCG@10']
    print(f"    Improvement: {'+' if improvement > 0 else ''}{improvement:.4f}")

    if improvement > 0:
        print("\n  Fine-tuned reranker HELPED! Saving results...")
        save_ranked_lists(reranked, cache_path)
    else:
        print("\n  Fine-tuned reranker didn't help. Not saving.")

    del model
    gc.collect()

    elapsed = time.time() - start_time
    print(f"\n  Time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()

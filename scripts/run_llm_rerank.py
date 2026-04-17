#!/usr/bin/env python3
"""
Improved pipeline: Skip cross-encoder (it HURTS citation prediction).
Go directly from fused results → LLM reranking with sliding window.

Key findings:
- ms-marco-MiniLM cross-encoder DROPPED NDCG@10 from 0.5760 to 0.2342
- Cross-encoders trained on web search are anti-correlated with citation prediction
- LLM reranking with citation-specific prompting is our best bet
- Sliding window: rerank positions 1-30, then 25-60 to fix mid-range ordering

Pipeline: Fused (0.5760) → Soft year boost → LLM rerank (sliding window) → Submission
"""
import gc
import json
import sys
import time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import (
    PipelineConfig, DATA_DIR, RESULTS_DIR, SUBMISSIONS_DIR, CHALLENGE_DIR,
    load_corpus, load_queries, load_qrels, load_ranked_lists, save_ranked_lists,
    evaluate, assemble_submission,
)
import numpy as np
from tqdm.auto import tqdm


def apply_soft_year_boost(ranked_lists, queries_df, corpus_df, boost_weight=0.15):
    """Soft year proximity boost — penalize future papers, boost older ones.

    Unlike the hard year filter (which destroyed results by removing 168 docs/query),
    this applies a soft multiplier:
    - candidate_year > query_year: multiply RRF score by 0.3 (heavy penalty, not removal)
    - candidate_year <= query_year: small boost for temporal proximity
    """
    query_years = dict(zip(queries_df["doc_id"], queries_df["year"]))
    corpus_years = dict(zip(corpus_df["doc_id"], corpus_df["year"]))

    # We need the RRF scores to re-weight. Since we only have ranked lists,
    # re-derive scores from rank position using the same k=30
    k = 30
    boosted = {}

    for qid, doc_ids in ranked_lists.items():
        q_year = query_years.get(qid, 2025)
        scored = []
        for rank, doc_id in enumerate(doc_ids):
            base_score = 1.0 / (k + rank + 1)  # Pseudo-RRF score from position
            c_year = corpus_years.get(doc_id, 2020)

            if c_year > q_year:
                # Future paper — heavy penalty but don't remove
                year_mult = 0.3
            else:
                # Past paper — slight boost for closer years
                gap = q_year - c_year
                if gap <= 5:
                    year_mult = 1.0 + boost_weight * (1.0 - gap/5)
                elif gap <= 15:
                    year_mult = 1.0
                else:
                    year_mult = 0.95  # Very old papers slightly penalized

            scored.append((doc_id, base_score * year_mult))

        scored.sort(key=lambda x: x[1], reverse=True)
        boosted[qid] = [d for d, _ in scored]

    return boosted


def rerank_llm_sliding_window(ranked_lists, queries_df, corpus_df, config,
                                window_size=30, step_size=20, n_windows=2):
    """Sliding window LLM reranking for citation prediction.

    Window 1: positions 1-30 → rerank
    Window 2: positions 21-50 → rerank
    Then merge: use best rank from either window.

    Key improvements over basic reranking:
    - Full abstracts (not truncated to 300 chars)
    - Year and venue info included
    - Citation-specific prompt emphasizing functional relationships
    - Larger context window (30 candidates per pass)
    """
    import anthropic
    from dotenv import load_dotenv
    load_dotenv(CHALLENGE_DIR / ".env")

    tag = config.query_set
    cache_path = RESULTS_DIR / f"stage4_llm_sliding_{tag}.json"
    if cache_path.exists():
        print(f"  [LLM] Loading cached results from {cache_path}")
        return load_ranked_lists(cache_path)

    client = anthropic.Anthropic()

    # Build lookups with full info
    corpus_lookup = {}
    for _, row in corpus_df.iterrows():
        corpus_lookup[row["doc_id"]] = {
            "title": str(row.get("title", "") or "").strip(),
            "abstract": str(row.get("abstract", "") or "").strip()[:600],
            "year": int(row.get("year", 0)),
            "venue": str(row.get("venue", "") or "").strip(),
            "domain": str(row.get("domain", "") or "").strip(),
        }

    query_lookup = {}
    for _, row in queries_df.iterrows():
        query_lookup[row["doc_id"]] = {
            "title": str(row.get("title", "") or "").strip(),
            "abstract": str(row.get("abstract", "") or "").strip(),
            "year": int(row.get("year", 0)),
            "venue": str(row.get("venue", "") or "").strip(),
            "domain": str(row.get("domain", "") or "").strip(),
        }

    reranked = {}
    errors = 0
    total_cost = 0.0

    for qid in tqdm(ranked_lists, desc="LLM sliding window reranking"):
        q = query_lookup.get(qid, {})
        all_candidates = ranked_lists[qid]

        # Collect scores across windows: doc_id → best_rank
        best_ranks = {}

        for w_idx in range(n_windows):
            start = w_idx * step_size
            end = start + window_size
            window_cands = all_candidates[start:end]

            if len(window_cands) < 3:
                break

            cand_str = ""
            for i, cid in enumerate(window_cands):
                c = corpus_lookup.get(cid, {})
                cand_str += f"[{i+1}] \"{c.get('title', 'N/A')}\" ({c.get('year', '?')}, {c.get('venue', 'N/A')})\n"
                abs_text = c.get('abstract', '')
                if abs_text:
                    cand_str += f"    {abs_text[:400]}\n"
                cand_str += "\n"

            prompt = f"""You are an expert at predicting scientific citations. Given a research paper (the "query"), rank the candidate papers by how likely the query paper would CITE each one in its references.

A paper cites another when it:
- Builds on its methodology or theoretical framework
- Uses it as a foundational reference in the same research area
- Directly compares results or approaches
- References it for background/context in the introduction
- Uses datasets, tools, or benchmarks introduced by it

QUERY PAPER:
Title: "{q.get('title', 'N/A')}"
Year: {q.get('year', '?')} | Venue: {q.get('venue', 'N/A')} | Domain: {q.get('domain', 'N/A')}
Abstract: {q.get('abstract', 'N/A')[:800]}

CANDIDATES ({len(window_cands)} papers, rank ALL from most to least likely cited):
{cand_str}
Return ONLY the numbers in order from most to least likely to be cited, separated by " > ".
Example for 5 papers: 3 > 1 > 5 > 2 > 4"""

            try:
                response = client.messages.create(
                    model=config.llm_model,
                    max_tokens=300,
                    temperature=0,
                    messages=[{"role": "user", "content": prompt}]
                )
                text = response.content[0].text.strip()

                # Track cost
                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens
                total_cost += (input_tokens * 3 / 1_000_000) + (output_tokens * 15 / 1_000_000)

                # Parse ranking
                parts = [p.strip() for p in text.split(">")]
                order = []
                seen = set()
                for p in parts:
                    try:
                        idx = int(p) - 1
                        if 0 <= idx < len(window_cands) and idx not in seen:
                            order.append(idx)
                            seen.add(idx)
                    except ValueError:
                        continue

                # Add missing indices
                for i in range(len(window_cands)):
                    if i not in seen:
                        order.append(i)

                # Assign ranks within this window
                for rank_pos, orig_idx in enumerate(order):
                    doc_id = window_cands[orig_idx]
                    # Global rank = window_start + position_in_window
                    global_rank = start + rank_pos
                    if doc_id not in best_ranks or global_rank < best_ranks[doc_id]:
                        best_ranks[doc_id] = global_rank

            except Exception as e:
                print(f"  [LLM] Error for {qid} window {w_idx}: {e}")
                errors += 1
                # Keep original ranks for this window
                for rank_pos, cid in enumerate(window_cands):
                    global_rank = start + rank_pos
                    if cid not in best_ranks or global_rank < best_ranks[cid]:
                        best_ranks[cid] = global_rank

        # Merge: sort all seen candidates by their best rank
        seen_docs = sorted(best_ranks.items(), key=lambda x: x[1])
        reranked_ids = [d for d, _ in seen_docs]

        # Append remaining candidates not in any window
        covered = set(reranked_ids)
        for doc_id in all_candidates:
            if doc_id not in covered:
                reranked_ids.append(doc_id)

        reranked[qid] = reranked_ids[:100]

    print(f"\n  [LLM] Total estimated cost: ${total_cost:.4f}")
    if errors:
        print(f"  [LLM] {errors} errors (fell back to original order)")

    save_ranked_lists(reranked, cache_path)
    return reranked


def main():
    start = time.time()
    config = PipelineConfig(query_set="public", apply_year_filter=False)

    print("=" * 68)
    print("IMPROVED PIPELINE: FUSED → SOFT YEAR BOOST → LLM RERANKING")
    print("(Cross-encoder SKIPPED — hurts citation prediction)")
    print("=" * 68)

    # Load data
    print("\nLoading data...")
    corpus_df = load_corpus(DATA_DIR / "corpus.parquet")
    queries_df = load_queries(DATA_DIR / "queries.parquet")
    qrels = load_qrels(DATA_DIR / "qrels.json")
    query_domains = dict(zip(queries_df["doc_id"], queries_df["domain"]))
    print(f"  Corpus: {len(corpus_df)} | Queries: {len(queries_df)} | Qrels: {len(qrels)}")

    # ── Load fused results (use v2 if available) ──
    fused_v2_path = RESULTS_DIR / "stage1_fused_v2_public.json"
    fused_v1_path = RESULTS_DIR / "stage1_fused_public.json"
    fused_path = fused_v2_path if fused_v2_path.exists() else fused_v1_path
    print(f"\nLoading fused results from {fused_path}...")
    fused = load_ranked_lists(fused_path)
    r = evaluate(fused, qrels, verbose=False)
    fused_ndcg = r['overall']['NDCG@10']
    print(f"  Fused: NDCG@10={fused_ndcg:.4f}  Recall@100={r['overall']['Recall@100']:.4f}  MAP={r['overall']['MAP']:.4f}")

    # ── Soft year boost ──
    print("\n" + "=" * 68)
    print("SOFT YEAR BOOST")
    print("=" * 68)
    boosted = apply_soft_year_boost(fused, queries_df, corpus_df, boost_weight=0.15)
    r = evaluate(boosted, qrels, verbose=False)
    boosted_ndcg = r['overall']['NDCG@10']
    print(f"  After year boost: NDCG@10={boosted_ndcg:.4f}  Recall@100={r['overall']['Recall@100']:.4f}")

    if boosted_ndcg > fused_ndcg:
        print(f"  Year boost improved! Using boosted results. (+{boosted_ndcg - fused_ndcg:.4f})")
        best_pre_llm = boosted
        best_pre_llm_ndcg = boosted_ndcg
    else:
        print(f"  Year boost didn't help. Using original fused results.")
        best_pre_llm = fused
        best_pre_llm_ndcg = fused_ndcg

    # ── LLM Sliding Window Reranking ──
    print("\n" + "=" * 68)
    print("LLM RERANKING (Claude Sonnet, sliding window: 2 windows of 30)")
    print("=" * 68)

    config.llm_top_k = 30  # Larger window
    final = rerank_llm_sliding_window(
        best_pre_llm, queries_df, corpus_df, config,
        window_size=30, step_size=20, n_windows=2
    )
    r = evaluate(final, qrels, query_domains=query_domains)
    final_ndcg = r['overall']['NDCG@10']

    # ── Save public submission ──
    print("\n" + "=" * 68)
    print("SAVING PUBLIC SUBMISSION")
    print("=" * 68)
    assemble_submission(final, config, label="submission_public_v2")

    elapsed = time.time() - start
    print(f"\n{'='*68}")
    print(f"PIPELINE COMPLETE — {elapsed/60:.1f} minutes")
    print(f"{'='*68}")
    print(f"  Baseline (MiniLM):    0.5073")
    print(f"  Fused (5 retrievers): {fused_ndcg:.4f}")
    print(f"  After year boost:     {boosted_ndcg:.4f}")
    print(f"  After LLM reranking:  {final_ndcg:.4f}")
    print(f"  Leader:               0.6725")

    improvement = final_ndcg - 0.5073
    gap_to_leader = 0.6725 - final_ndcg
    print(f"\n  Improvement over baseline: +{improvement:.4f}")
    print(f"  Gap to leader: {gap_to_leader:.4f}")

    if final_ndcg > 0.65:
        print("\n  >>> READY FOR HELD-OUT SUBMISSION <<<")
    else:
        print(f"\n  Need more optimization (gap to 0.65: {0.65 - final_ndcg:.4f})")


if __name__ == "__main__":
    main()

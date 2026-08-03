"""Stage 4: Claude listwise reranking (RankGPT-style).

Kept for reproducibility of the "what didn't work" finding: this dropped
NDCG@10 to 0.5573 at roughly $0.18 per 100 queries, because the model ranks
by semantic/topical plausibility rather than the structural signals (domain,
venue, direct lexical overlap) that actually predict citation here. Not part
of the submitted pipeline; `generate_submission_v4.py --llm` routes through
`scripts/run_llm_rerank.py`'s sliding-window variant, not this function.
"""
from tqdm.auto import tqdm

from .paths import CHALLENGE_DIR, RESULTS_DIR
from .storage import load_ranked_lists, save_ranked_lists


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

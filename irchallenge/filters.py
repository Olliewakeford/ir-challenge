"""Stage 2: the year filter.

Kept for reproducibility of the "what didn't work" findings in the README:
removing candidates published after the query paper looked like a reasonable
prior, but dropped NDCG@10 from 0.6151 to 0.3782 on this dataset, because most
of the "citation" relation here reflects related/updated work rather than a
strict citation-graph edge. Disabled (`apply_year_filter=False`) in the
submitted pipeline.
"""


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

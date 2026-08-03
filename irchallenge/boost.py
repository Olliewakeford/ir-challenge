"""The domain/venue boost: the pipeline's single largest contributor to the
final score, applied after fusion, before assembling the submission.

97.6% of citations in this corpus are same-domain (18 of 736 query-relevant
pairs cross a domain boundary), so multiplying same-domain candidates'
post-fusion score keeps them from being pushed out by broadly on-topic but
wrong-domain results. Same-venue gets a smaller boost on the same logic.
"""


def apply_domain_venue_boost(fused_results, queries_df, corpus_df, domain_boost, venue_boost):
    """Rescore each candidate list, boosting same-domain and same-venue matches."""
    query_domains = dict(zip(queries_df["doc_id"], queries_df["domain"]))
    corpus_domains = dict(zip(corpus_df["doc_id"], corpus_df["domain"]))
    query_venues = dict(zip(queries_df["doc_id"], queries_df["venue"]))
    corpus_venues = dict(zip(corpus_df["doc_id"], corpus_df["venue"]))

    boosted = {}
    for qid, ranked in fused_results.items():
        q_domain = query_domains.get(qid, "")
        q_venue = query_venues.get(qid, "")
        scored = []
        for rank, doc_id in enumerate(ranked):
            base_score = 1.0 / (rank + 1)
            c_domain = corpus_domains.get(doc_id, "")
            c_venue = corpus_venues.get(doc_id, "")
            if c_domain == q_domain and q_domain:
                base_score *= domain_boost
            if c_venue == q_venue and q_venue:
                base_score *= venue_boost
            scored.append((doc_id, base_score))
        scored.sort(key=lambda x: x[1], reverse=True)
        boosted[qid] = [d for d, _ in scored[:300]]
    return boosted

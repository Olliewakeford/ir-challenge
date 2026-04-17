# IR Challenge — Scientific Citation Retrieval

Codabench submission for the M1 AI Information Retrieval course (Paris-Saclay, 2026).

Task: given a query paper, retrieve the 100 papers it most likely cites from a 20,000-paper corpus. Primary metric: NDCG@10.

## Pipeline (v4 + boost)

Eight retrievers fused with weighted Reciprocal Rank Fusion, then domain/venue boosting:

1. **Dense retrievers**: SPECTER2 (asymmetric adapters), SciNCL, MiniLM-L6-v2
2. **Sparse retrievers**: BM25 title+abstract, BM25 full text, BM25 per-section, TF-IDF full text
3. **Citation context**: sentences around `[n]` markers in the query full text used as extra queries
4. **Weighted RRF fusion** (k=6, weights tuned by coordinate descent on public qrels)
5. **Domain + venue boost**: same-domain candidates boosted ×10, same-venue ×2

## Layout

```
scripts/
  pipeline.py                 # core retrieval + fusion + boosting + LLM rerank
  run_refusion_v4.py          # coordinate-descent weight tuning on public qrels
  generate_submission_v4.py   # build held-out submission with v4 weights
submissions/
  submission_v4_boost10.zip   # final Codabench submission (v4 + 10x domain boost)
  submission_v4_boost10_data.json
requirements.txt
```

## Running

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Tune fusion weights on public qrels
python scripts/run_refusion_v4.py

# Build held-out submission
python scripts/generate_submission_v4.py --boost 10
```

Data files (corpus, queries, qrels, cached embeddings) aren't in the repo — they're too large and provided by the course.

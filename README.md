# IR Challenge: Scientific Citation Retrieval

Team project for the M1 AI & Data Science Information Retrieval course
(Université Paris-Saclay, 2026): **Ahzam Afaq, Frederic Busch, Said Abolhassan
Razavi, Oliver Wakeford**. Submitted to a course leaderboard hosted on
Codabench.

## Task

Given a query paper, retrieve the 100 papers from a 20,000-paper corpus it is
most likely to cite. Scored with NDCG@10. Each query has roughly 7.4 relevant
documents on average, drawn from 19 subject domains. The corpus is split into
100 public queries (with released relevance judgments, for local tuning) and
100 held-out queries (ungraded locally, scored on the leaderboard).

## Result

The team's final submission scored **0.7494** NDCG@10 on the held-out
queries, about 0.02 behind the top score on the leaderboard (0.7707). The
leaderboard is public:
[codabench.org/competitions/15308](https://www.codabench.org/competitions/15308/).
It is a leaderboard for the course cohort, not a public IR benchmark.

This repository holds the pipeline the team presented mid-competition (v4
fusion plus the domain/venue boost, submitted 14 April 2026), which scored
**0.6965** held out. The team kept submitting until the deadline, and the
last-day work behind the 0.7494 submission is not in this repository. The
approach, ablations and caveats below all describe the presented pipeline.

## Approach

Eight retrieval signals fused with weighted Reciprocal Rank Fusion, then
reranked with two structural boosts:

| Signal | RRF weight |
|---|---|
| BM25, full text | 3.50 |
| MiniLM-L6-v2 (dense) | 1.80 |
| TF-IDF, full text | 1.20 |
| SciNCL (dense) | 0.70 |
| SPECTER2 (dense, asymmetric adapters) | 0.30 |
| BM25, per-section | 0.20 |
| Citation-context BM25 (text around `[n]` markers) | 0.15 |
| BM25, title+abstract | 0.00 (dropped by the optimizer) |

Fusion uses RRF with `k=6`; weights were tuned by coordinate descent against
the public query set's relevance judgments (`scripts/run_refusion_v4.py`).
After fusion, candidates are rescored with two multiplicative boosts:
same-domain candidates ×10, same-venue candidates ×2.

The domain boost is the single largest contributor to the presented
pipeline's score. It follows directly from a corpus property found during
error analysis: of 736 query-relevant document pairs, only 18 (2.4%) cross a
domain boundary, so 97.6% of citations are same-domain. Applying the boost adds about +0.10 NDCG@10
(+0.1019 at the submitted 10x; +0.1026 at 26x, the best factor on the public
queries), roughly half of the total gain over the baseline.

### Score progression (public queries, local evaluation)

| Stage | NDCG@10 |
|---|---|
| MiniLM baseline (single dense retriever) | 0.5073 |
| BM25, full text (best single signal) | 0.5429 |
| 5-signal RRF fusion | 0.5760 |
| 7-signal RRF fusion | 0.5918 |
| 8 signals + retuned weights (v4) | 0.6151 |
| + domain/venue boost (10x/2x, submitted) | ~0.71–0.72 (local) |

### Leave-one-out ablation (from the 8-signal v4 fusion, before boosting)

| Signal removed | NDCG@10 drop |
|---|---|
| BM25, full text | −0.0540 |
| TF-IDF, full text | −0.0295 |
| MiniLM-L6-v2 | −0.0229 |
| SPECTER2 | −0.0042 |

BM25 full-text is the backbone signal by a wide margin. In this pipeline it
also beat every dense retriever, including SPECTER2 (NDCG@10 0.4164
standalone) and SciNCL (0.4722 standalone). Lexical overlap on specific terms (method names,
dataset names, model names) turned out to be a stronger citation signal than
learned semantic similarity for this task.

### Recall ceiling

Across the 8 signals, 663 of 736 relevant documents (90.1%) are retrievable
in the top 100 candidates. The remaining 73 (9.9%) are never surfaced by any
of the 8 signals at any rank, a hard ceiling on what reranking alone can fix.

## What didn't work

These are the most useful part of the project. Several intuitive additions
made results worse, and understanding why constrained the rest of the design.

- **Cross-encoder reranking** (`cross-encoder/ms-marco-MiniLM-L-12-v2`, and a
  fine-tuned variant of the same model): NDCG@10 dropped from 0.5918 to
  0.4762 in both cases. The cross-encoder optimizes for topical relevance,
  which is not the same as "would this paper cite that paper", so it pushed
  down narrow, highly-relevant methodology papers in favor of broadly
  on-topic ones.
- **LLM listwise reranking** (Claude, RankGPT-style prompting over the top
  candidates): NDCG@10 dropped to 0.5573, at a cost of roughly $0.18 per 100
  queries. The model ranks by semantic similarity and topical/temporal
  plausibility, not by the structural signals (domain, venue, direct lexical
  overlap) that actually predict citation.
- **Publication-year filter** (removing candidates published after the query
  paper, on the assumption a paper can't cite something written after it):
  NDCG@10 dropped from 0.5918 to 0.3782. The assumption was wrong for this
  dataset: 68.5% of documents marked as cited have a *later* publication year
  than the query paper (mean gap +4.3 years), most likely because the
  "citation" relation in this corpus reflects related/updated work rather
  than a strict citation-graph edge. This filter would have been catastrophic
  in the actual submission.
- **E5-large-v2** as an additional dense retriever: hurt NDCG@10 at every
  weight tried in the fusion; dropped entirely from the final pipeline.
- **k-NN graph expansion** (expanding candidates via nearest neighbors of the
  query in embedding space): no measurable effect (0.5760 → 0.5760).

## Honest caveats

- The public qrels (the only relevance judgments available locally) were
  used both to build the pipeline and to tune the RRF weights and boost
  factors. That inflates the local NDCG@10 (0.71–0.72) relative to the
  held-out score of this pipeline (0.6965). Some of the gap is normal
  train/test variance, some is overfitting to the 100 public queries.
- The domain boost factor (10x rather than a higher value that scored better
  locally) was deliberately chosen to be conservative, specifically to limit
  that local-vs-held-out gap rather than to maximize the local score.
- One of the eight fusion weights (title+abstract BM25) was tuned to zero by
  the optimizer, i.e. dropped. This is expected behaviour of coordinate
  descent given redundancy between title+abstract BM25 and the other BM25
  variants, not a bug.

## Layout

The retrieval, fusion and boosting logic lives in `irchallenge/`, one module
per stage. `scripts/pipeline.py` was originally a single 1,200-line file;
it's now a thin entry point over the package, kept so the CLI (`python
scripts/pipeline.py --stages ...`) and the other scripts' imports still work
unchanged.

```
irchallenge/
  paths.py       data/results/submission directories
  config.py      PipelineConfig, the run configuration
  storage.py     loading and caching corpus/query/qrels/embeddings/ranked lists
  text.py        text normalisation shared by the retrievers
  metrics.py     Recall/Precision/MRR/NDCG/MAP and the evaluate() report
  device.py      torch device selection (mps/cpu)
  retrievers.py  the eight retrieval signals
  fusion.py      weighted Reciprocal Rank Fusion
  filters.py     the year filter (kept for reproducibility; see "what didn't work")
  boost.py       the domain/venue boost, the pipeline's biggest single gain
  rerank.py      cross-encoder and BGE reranking stages
  llm_rerank.py  Claude listwise reranking
  submission.py  assembling and zipping a Codabench submission
  cli.py         the scripts/pipeline.py entry point

scripts/
  pipeline.py                 thin CLI wrapper over irchallenge/, kept for compatibility
  embed.py                    corpus/query encoding and embedding cache
  fusion_lib.py               RRF fusion helpers (see tests/test_fusion_lib.py)
  run_bm25_sections.py             per-section BM25 signal
  run_citation_context_retriever.py  citation-context BM25 signal
  run_new_signals.py               the signals added in the 7 and 8-way fusions
  run_refusion_v4.py          coordinate-descent weight tuning on public qrels (v4)
  run_domain_boost_tune.py    sweeps the domain/venue boost factors
  generate_submission_v4.py   builds the presented submission (v4 weights + boost)

  # the experiments in "What didn't work", kept so the claims can be rerun
  run_cross_encoder_as_signal.py   cross-encoder reranking (0.5918 -> 0.4762)
  run_finetune_reranker.py         the fine-tuned variant of the same model
  run_llm_rerank.py                RankGPT-style listwise reranking with Claude
  run_e5_large.py                  E5-large-v2 as an extra dense retriever
  run_knn_expansion.py             k-NN graph expansion (no measurable effect)
tests/
  test_fusion_lib.py, test_fusion.py, test_metrics.py, test_boost.py,
  test_filters.py, test_text.py   unit tests for the pure logic (fusion, metrics,
                                  boosting, filtering, text normalisation); none
                                  of it needs the course data, so it runs in CI
submissions/
  submission_v4_boost10_data.json  presented Codabench submission, 14 April (v4 weights + 10x boost)
docs/
  IR_challenge_presentation_WAKEFORD_BUSCH_AFAQ_RAZAVI.pdf   the team's presentation
requirements.txt
```

## Running

Data files (corpus, queries, qrels, cached embeddings) are provided by the
course and are not included in this repo.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Tune fusion weights on the public query set
python scripts/run_refusion_v4.py

# Build the held-out submission (8 signals, v4 weights, 10x domain boost)
python scripts/generate_submission_v4.py --boost 10
```

The unit tests cover the pure logic only (fusion, metrics, boosting, filtering,
text normalisation) and don't need the course data, so they run without any of
the above:

```bash
pytest
```

A full run (encoding all corpus/query embeddings from scratch, all 8
retrievers, fusion, boosting) takes roughly 40–60 minutes on an M5 Pro
MacBook. Cached embeddings and per-retriever results are reused on
subsequent runs.

`scripts/generate_submission_v4.py --llm` routes through `run_llm_rerank.py`.
That path is included so the result can be reproduced, but it is not what was
submitted, and it scored worse. See "What didn't work" above. The default
(non-`--llm`) path does not touch it.

`run_llm_rerank.py` reads `ANTHROPIC_API_KEY` from a `.env` file in the repo
root. That file is gitignored and is not included, so supply your own key if
you want to rerun that experiment.

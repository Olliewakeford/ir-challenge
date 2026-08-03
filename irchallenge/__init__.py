"""irchallenge — scientific citation retrieval pipeline.

Eight retrieval signals fused with weighted Reciprocal Rank Fusion, plus a
domain/venue boosting stage. See the repo README for the full writeup and
results; this package holds the code, split out of the original single-file
scripts/pipeline.py into one module per concern:

  paths        data/results/submission directories
  config       PipelineConfig, the run configuration
  storage      loading and caching corpus/query/qrels/embeddings/ranked lists
  text         text normalisation shared by the retrievers
  metrics      Recall/Precision/MRR/NDCG/MAP and the evaluate() report
  device       torch device selection (mps/cpu)
  retrievers   the eight retrieval signals
  fusion       weighted Reciprocal Rank Fusion
  filters      the year filter (kept for reproducibility; see "what didn't work")
  boost        the domain/venue boost, the pipeline's biggest single gain
  rerank       cross-encoder and BGE reranking stages
  llm_rerank   Claude listwise reranking
  submission   assembling and zipping a Codabench submission
  cli          the `python -m irchallenge` / scripts/pipeline.py entry point
"""

"""Command-line entry point: the same five-stage pipeline `scripts/pipeline.py`
used to run directly, now assembled from the split-out modules.

  python scripts/pipeline.py                       # full pipeline on public queries
  python scripts/pipeline.py --query-set held_out   # generate submission
  python scripts/pipeline.py --stages 1,2,3         # run specific stages
"""
import argparse

from .config import PipelineConfig
from .filters import apply_year_filter
from .fusion import fuse_rrf, tune_rrf_weights
from .llm_rerank import rerank_llm
from .metrics import evaluate
from .paths import DATA_DIR, RESULTS_DIR
from .rerank import rerank_cross_encoder
from .retrievers import (
    retrieve_bm25_sections,
    retrieve_bm25_ta,
    retrieve_bm25_ft,
    retrieve_citation_ctx,
    retrieve_minilm,
    retrieve_scincl,
    retrieve_specter2,
)
from .storage import load_corpus, load_qrels, load_queries, load_ranked_lists, save_ranked_lists
from .submission import assemble_submission


def eval_if_public(results, qrels, config, label=""):
    """Evaluate only if we have qrels (public query set)."""
    if config.query_set == "public" and qrels:
        print(f"\n>>> Evaluating: {label}")
        return evaluate(results, qrels)
    return None


def main():
    parser = argparse.ArgumentParser(description="IR Challenge Pipeline")
    parser.add_argument("--query-set", default="public",
                        choices=["public", "held_out"],
                        help="Which query set to use")
    parser.add_argument("--stages", default="all",
                        help="Comma-separated stage numbers, 'all', or 'tune'")
    parser.add_argument("--cross-encoder", default=None,
                        help="Override cross-encoder model")
    args = parser.parse_args()

    config = PipelineConfig(query_set=args.query_set)
    if args.cross_encoder:
        config.cross_encoder_model = args.cross_encoder

    stages = args.stages
    if stages == "all":
        run_stages = {1, 2, 3, 4, 5}
    elif stages == "tune":
        run_stages = {"tune"}
    else:
        run_stages = set(int(s) for s in stages.split(","))

    # Load data
    print("=" * 68)
    print(f"IR CHALLENGE PIPELINE — Query set: {config.query_set}")
    print("=" * 68)

    print("\nLoading data...")
    corpus_df = load_corpus(DATA_DIR / "corpus.parquet")
    if config.query_set == "public":
        queries_df = load_queries(DATA_DIR / "queries.parquet")
        qrels = load_qrels(DATA_DIR / "qrels.json")
    else:
        queries_df = load_queries(DATA_DIR / "held_out_queries.parquet")
        qrels = {}
    print(f"  Corpus: {len(corpus_df)} docs | Queries: {len(queries_df)} | Qrels: {len(qrels)}")

    query_domains = dict(zip(queries_df["doc_id"], queries_df["domain"])) if "domain" in queries_df.columns else None

    # -- Stage 1: Retrieval --
    all_retrievers = {}
    if 1 in run_stages or "tune" in run_stages:
        print("\n" + "=" * 68)
        print("STAGE 1: MULTI-SIGNAL RETRIEVAL")
        print("=" * 68)

        print("\n[1/6] SPECTER2 Asymmetric...")
        all_retrievers["specter2"] = retrieve_specter2(queries_df, corpus_df, config)
        eval_if_public(all_retrievers["specter2"], qrels, config, "SPECTER2 only")

        print("\n[2/6] SciNCL...")
        all_retrievers["scincl"] = retrieve_scincl(queries_df, corpus_df, config)
        eval_if_public(all_retrievers["scincl"], qrels, config, "SciNCL only")

        print("\n[3/6] MiniLM-L6-v2...")
        all_retrievers["minilm"] = retrieve_minilm(queries_df, corpus_df, config)
        eval_if_public(all_retrievers["minilm"], qrels, config, "MiniLM only")

        print("\n[4/6] BM25 on title+abstract...")
        all_retrievers["bm25_ta"] = retrieve_bm25_ta(queries_df, corpus_df, config)
        eval_if_public(all_retrievers["bm25_ta"], qrels, config, "BM25-TA only")

        print("\n[5/6] BM25 on full text...")
        all_retrievers["bm25_ft"] = retrieve_bm25_ft(queries_df, corpus_df, config)
        eval_if_public(all_retrievers["bm25_ft"], qrels, config, "BM25-FT only")

        print("\n[6/6] BM25 section-aware...")
        all_retrievers["bm25_sections"] = retrieve_bm25_sections(queries_df, corpus_df, config)
        eval_if_public(all_retrievers["bm25_sections"], qrels, config, "BM25-Sections only")

    # -- Weight tuning --
    if "tune" in run_stages:
        tune_rrf_weights(all_retrievers, qrels, queries_df, corpus_df)
        print("\nApply these weights to PipelineConfig and re-run.")
        return

    # -- RRF Fusion --
    fused = None
    if 1 in run_stages:
        fused = fuse_rrf(all_retrievers, config)
        save_ranked_lists(fused, RESULTS_DIR / f"stage1_fused_{config.query_set}.json")
        eval_if_public(fused, qrels, config, "RRF Fusion (pre-filter)")

    # -- Stage 2: Year filter --
    if 2 in run_stages:
        print("\n" + "=" * 68)
        print("STAGE 2: YEAR FILTER")
        print("=" * 68)
        if 1 not in run_stages:
            fused = load_ranked_lists(RESULTS_DIR / f"stage1_fused_{config.query_set}.json")
        fused = apply_year_filter(fused, queries_df, corpus_df, config)
        save_ranked_lists(fused, RESULTS_DIR / f"stage2_filtered_{config.query_set}.json")
        eval_if_public(fused, qrels, config, "After year filter")

    # -- Stage 3: Cross-encoder --
    reranked = None
    if 3 in run_stages:
        print("\n" + "=" * 68)
        print("STAGE 3: CROSS-ENCODER RERANKING")
        print("=" * 68)
        if 2 not in run_stages:
            fused = load_ranked_lists(RESULTS_DIR / f"stage2_filtered_{config.query_set}.json")
        reranked = rerank_cross_encoder(fused, queries_df, corpus_df, config)
        eval_if_public(reranked, qrels, config, f"Cross-encoder: {config.cross_encoder_model}")

    # -- Stage 4: LLM reranking --
    final = None
    if 4 in run_stages:
        print("\n" + "=" * 68)
        print("STAGE 4: LLM LISTWISE RERANKING")
        print("=" * 68)
        if 3 not in run_stages:
            model_slug = config.cross_encoder_model.replace("/", "_")
            reranked = load_ranked_lists(
                RESULTS_DIR / f"stage3_{model_slug}_{config.query_set}.json")
        final = rerank_llm(reranked, queries_df, corpus_df, config)
        eval_if_public(final, qrels, config, "After LLM reranking")

    # -- Stage 5: Submission --
    if 5 in run_stages:
        print("\n" + "=" * 68)
        print("STAGE 5: ASSEMBLY & SUBMISSION")
        print("=" * 68)
        if 4 not in run_stages:
            # Load best available results
            tag = config.query_set
            llm_path = RESULTS_DIR / f"stage4_llm_{tag}.json"
            model_slug = config.cross_encoder_model.replace("/", "_")
            ce_path = RESULTS_DIR / f"stage3_{model_slug}_{tag}.json"
            if llm_path.exists():
                final = load_ranked_lists(llm_path)
            elif ce_path.exists():
                final = load_ranked_lists(ce_path)
            else:
                fused_path = RESULTS_DIR / f"stage2_filtered_{tag}.json"
                final = load_ranked_lists(fused_path)

        label = f"submission_{config.query_set}"
        submission = assemble_submission(final, config, label=label)
        if config.query_set == "public" and qrels:
            print("\nFinal evaluation:")
            evaluate(submission, qrels, query_domains=query_domains)

    print("\n" + "=" * 68)
    print("PIPELINE COMPLETE")
    print("=" * 68)


if __name__ == "__main__":
    main()

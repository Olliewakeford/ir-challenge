"""Run configuration for the pipeline."""
from dataclasses import dataclass, field


@dataclass
class PipelineConfig:
    # Query set
    query_set: str = "public"  # "public" or "held_out"

    # Stage 1: Retrieval
    retriever_top_k: int = 200
    rrf_k: int = 60
    rrf_weights: dict = field(default_factory=lambda: {
        "specter2": 1.5,
        "scincl": 1.0,
        "minilm": 0.7,
        "bm25_ta": 1.0,
        "bm25_ft": 0.8,
        "bm25_sections": 0.9,
    })
    fusion_top_k: int = 300

    # Stage 2: Filters
    apply_year_filter: bool = True

    # Stage 3: Cross-encoder
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    cross_encoder_batch_size: int = 64
    cross_encoder_top_k: int = 100
    cross_encoder_device: str = "mps"

    # Stage 4: LLM
    llm_model: str = "claude-sonnet-4-5-20250929"
    llm_top_k: int = 20

    # Bi-encoder batch sizes
    bi_encoder_batch_size: int = 128

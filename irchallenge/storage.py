"""Loading and saving corpus/query/qrels/embeddings/ranked-list data."""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_queries(path) -> pd.DataFrame:
    return pd.read_parquet(path)


def load_corpus(path) -> pd.DataFrame:
    return pd.read_parquet(path)


def load_qrels(path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_embeddings(emb_path, ids_path):
    embeddings = np.load(emb_path).astype(np.float32)
    with open(ids_path) as f:
        ids = json.load(f)
    assert len(embeddings) == len(ids), "Embedding count mismatch"
    return embeddings, ids


def save_ranked_lists(ranked_lists: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(ranked_lists, f)
    print(f"  Saved {len(ranked_lists)} queries to {path}")


def load_ranked_lists(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)

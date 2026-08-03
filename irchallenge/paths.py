"""Repo-relative paths shared by every stage of the pipeline."""
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
CHALLENGE_DIR = PACKAGE_DIR.parent
DATA_DIR = CHALLENGE_DIR / "data"
EMB_DIR = DATA_DIR / "embeddings"
SUBMISSIONS_DIR = CHALLENGE_DIR / "submissions"
RESULTS_DIR = DATA_DIR / "intermediate"

for _d in (EMB_DIR, SUBMISSIONS_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

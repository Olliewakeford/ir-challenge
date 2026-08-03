"""Text normalisation shared by the retrievers."""
import json
import re


def format_text(row) -> str:
    title = str(row.get("title", "") or "").strip()
    abstract = str(row.get("abstract", "") or "").strip()
    if title and abstract:
        return title + " " + abstract
    return title or abstract


def get_chunks(full_text: str, chunk_meta_json) -> list:
    meta = json.loads(chunk_meta_json) if isinstance(chunk_meta_json, str) else chunk_meta_json
    chunks = []
    for i, entry in enumerate(meta):
        char_start = entry["char_start"]
        if entry["type"] == "ta":
            char_end = entry["char_end"]
        else:
            char_end = meta[i + 1]["char_start"] if i + 1 < len(meta) else len(full_text)
        text = full_text[char_start:char_end].strip()
        chunks.append({"type": entry["type"], "text": text,
                       "char_start": char_start, "char_end": char_end})
    return chunks


def get_ta(row) -> str:
    return str(row.get("ta", "") or "").strip()


def get_body_chunks(row, min_chars: int = 100) -> list:
    chunks = get_chunks(row["full_text"], row["chunk_meta"])
    return [c["text"] for c in chunks if c["type"] == "body" and len(c["text"]) >= min_chars]


def clean_citation_markers(text: str) -> str:
    return re.sub(r'\[\d+(?:,\s*\d+)*\]', '', text)


def tokenize_simple(text: str) -> list:
    return re.findall(r'\w+', text.lower())

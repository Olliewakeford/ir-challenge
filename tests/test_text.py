"""Tests for irchallenge/text.py: text normalisation shared by the retrievers."""
import json

from irchallenge.text import (
    clean_citation_markers,
    format_text,
    get_body_chunks,
    get_chunks,
    get_ta,
    tokenize_simple,
)


def test_clean_citation_markers_removes_single_and_grouped_refs():
    text = "Attention is useful [3] and so is scale [12, 13]."
    assert clean_citation_markers(text) == "Attention is useful  and so is scale ."


def test_clean_citation_markers_leaves_non_bracket_numbers_alone():
    assert clean_citation_markers("trained for 12 epochs") == "trained for 12 epochs"


def test_tokenize_simple_lowercases_and_splits_on_word_boundaries():
    assert tokenize_simple("BERT: Pre-training, v2!") == ["bert", "pre", "training", "v2"]


def test_format_text_joins_title_and_abstract():
    row = {"title": "A Study", "abstract": "We show results."}
    assert format_text(row) == "A Study We show results."


def test_format_text_falls_back_to_whichever_field_is_present():
    assert format_text({"title": "Only Title", "abstract": ""}) == "Only Title"
    assert format_text({"title": "", "abstract": "Only abstract"}) == "Only abstract"
    assert format_text({"title": None, "abstract": None}) == ""


def test_get_ta_strips_and_defaults_to_empty_string():
    assert get_ta({"ta": "  padded  "}) == "padded"
    assert get_ta({}) == ""


def test_get_chunks_splits_ta_and_body_by_char_offsets():
    full_text = "TITLE ABSTRACT bodytext continues"
    meta = [
        {"type": "ta", "char_start": 0, "char_end": 14},
        {"type": "body", "char_start": 15},
    ]
    chunks = get_chunks(full_text, json.dumps(meta))
    assert chunks[0] == {"type": "ta", "text": "TITLE ABSTRACT", "char_start": 0, "char_end": 14}
    assert chunks[1]["type"] == "body"
    assert chunks[1]["text"] == "bodytext continues"


def test_get_body_chunks_filters_short_chunks():
    row = {
        "full_text": "TA short body",
        "chunk_meta": json.dumps([
            {"type": "ta", "char_start": 0, "char_end": 2},
            {"type": "body", "char_start": 3},
        ]),
    }
    # body text is "short body" (10 chars) -> excluded at min_chars=100, included at min_chars=5
    assert get_body_chunks(row, min_chars=100) == []
    assert get_body_chunks(row, min_chars=5) == ["short body"]

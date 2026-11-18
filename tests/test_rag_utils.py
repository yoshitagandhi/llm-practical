"""Tests for the Task 2 text utilities (chunking, page lookup, hit rule).

Run from the repo root:  python -m pytest tests
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "task2_rag"))
from chunking import chunk_text, coverage, find_passage, is_hit, normalise_text, page_for_offset  # noqa: E402

TEXT = " ".join(f"w{i}" for i in range(10))  # "w0 w1 ... w9"


def test_normalise_text_fixes_ligatures_and_whitespace():
    assert normalise_text("eﬃcient  inci-\n dent\n") == "efficient inci- dent"


def test_normalise_text_drops_pdf_bullets_and_plain_punctuation():
    assert normalise_text(" Keep the team’s “log” — up to date") == \
        "Keep the team's \"log\" - up to date"


def test_chunks_have_requested_size_and_overlap():
    chunks = chunk_text(TEXT, chunk_size=4, overlap=1)
    assert [c["text"] for c in chunks] == ["w0 w1 w2 w3", "w3 w4 w5 w6", "w6 w7 w8 w9"]


def test_chunk_offsets_point_back_into_text():
    for chunk in chunk_text(TEXT, chunk_size=3, overlap=1):
        assert TEXT[chunk["start_char"]:chunk["end_char"]] == chunk["text"]


def test_last_chunk_reaches_end_of_text():
    chunks = chunk_text(TEXT, chunk_size=4, overlap=0)
    assert chunks[-1]["text"].endswith("w9")


def test_invalid_overlap_raises():
    try:
        chunk_text(TEXT, chunk_size=3, overlap=3)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_page_for_offset():
    page_starts = [[1, 0], [2, 100], [4, 250]]
    assert page_for_offset(page_starts, 0) == 1
    assert page_for_offset(page_starts, 150) == 2
    assert page_for_offset(page_starts, 300) == 4


def test_hit_rule_needs_half_the_passage():
    chunk = {"start_char": 0, "end_char": 100}
    assert coverage(0, 100, 50, 150) == 0.5
    assert is_hit(chunk, 50, 150)        # exactly 50% inside
    assert not is_hit(chunk, 60, 160)    # only 40% inside


def test_find_passage_ignores_case_and_spacing():
    doc = normalise_text("The incident  response\nteam (IRT) handles events.")
    span = find_passage(doc, "incident Response team")
    assert span is not None and doc[span[0]:span[1]].lower() == "incident response team"
    assert find_passage(doc, "not in the document") is None

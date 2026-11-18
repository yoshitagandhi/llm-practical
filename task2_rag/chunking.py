"""Text normalisation, word-based chunking and the retrieval hit rule (Task 2).

Pure Python so it can be unit-tested without the PDF or the embedding model.
"""

import re
import unicodedata

WORD_PATTERN = re.compile(r"\S+")

# Curly quotes and long dashes are often changed when text is copied out of a PDF viewer.
PUNCTUATION_MAP = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"',
                                 "–": "-", "—": "-"})


def normalise_text(text):
    """Make PDF text and copied passages comparable.

    - NFKC turns PDF ligatures like "ﬁ" into "fi".
    - Private-use characters are dropped: in the NIST PDF the bullet points
      extract as U+F03C (302 times), which a copied passage would not contain.
    - Curly quotes and dashes become plain ASCII.
    - Collapsing whitespace removes PDF line breaks and double spaces.
    """
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Co")
    text = text.translate(PUNCTUATION_MAP)
    return " ".join(text.split())


def chunk_text(text, chunk_size, overlap):
    """Split text into chunks of `chunk_size` words, each sharing `overlap` words with the previous one.

    Returns dicts with the chunk text and its [start_char, end_char) position in `text`,
    so we can later check whether a chunk covers a given source passage.
    """
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    words = list(WORD_PATTERN.finditer(text))
    step = chunk_size - overlap
    chunks = []
    start_word = 0
    while start_word < len(words):
        end_word = min(start_word + chunk_size, len(words))
        start_char = words[start_word].start()
        end_char = words[end_word - 1].end()
        chunks.append({
            "chunk_id": len(chunks),
            "start_char": start_char,
            "end_char": end_char,
            "text": text[start_char:end_char],
        })
        if end_word == len(words):
            break
        start_word += step
    return chunks


def page_for_offset(page_starts, offset):
    """Return the page number whose text contains character `offset`.

    `page_starts` is a list of (page_number, start_char) in document order.
    """
    current_page = page_starts[0][0]
    for page_number, start_char in page_starts:
        if start_char > offset:
            break
        current_page = page_number
    return current_page


def coverage(chunk_start, chunk_end, gold_start, gold_end):
    """Fraction of the gold passage's characters that fall inside the chunk."""
    overlap = max(0, min(chunk_end, gold_end) - max(chunk_start, gold_start))
    return overlap / (gold_end - gold_start)


def is_hit(chunk, gold_start, gold_end, min_coverage=0.5):
    """Hit rule (approved): the chunk contains at least half of the source passage."""
    return coverage(chunk["start_char"], chunk["end_char"], gold_start, gold_end) >= min_coverage


def find_passage(document_text, passage):
    """Locate a copied source passage in the normalised document.

    Returns (start, end) or None. Tries an exact match first, then a
    case-insensitive one (copying from a PDF viewer sometimes changes case).
    """
    passage = normalise_text(passage)
    start = document_text.find(passage)
    if start == -1:
        start = document_text.lower().find(passage.lower())
    if start == -1:
        return None
    return start, start + len(passage)

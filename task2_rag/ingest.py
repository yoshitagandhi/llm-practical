"""Task 2 step 1: extract text from the source PDF, page by page.

Input : task2_rag/data/raw/<the PDF>   (never modified)
Output: task2_rag/data/processed/document.json
        {"source": ..., "num_pages": ..., "text": <whole document, normalised>,
         "page_starts": [[page_number, start_char], ...]}

Usage (from the repo root):
    python task2_rag/ingest.py
"""

import json
import sys
from pathlib import Path

from pypdf import PdfReader

TASK_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TASK_DIR))
from chunking import normalise_text  # noqa: E402

RAW_DIR = TASK_DIR / "data" / "raw"
OUTPUT_FILE = TASK_DIR / "data" / "processed" / "document.json"
MIN_PAGES = 30  # required by the brief


def find_pdf(raw_dir):
    pdfs = sorted(raw_dir.glob("*.pdf"))
    if len(pdfs) != 1:
        raise SystemExit(f"Expected exactly one PDF in {raw_dir}, found {len(pdfs)}")
    return pdfs[0]


def extract_pages(pdf_path):
    """Return a list of (page_number, normalised_text), skipping empty pages."""
    reader = PdfReader(pdf_path)
    pages = []
    for index, page in enumerate(reader.pages, start=1):
        text = normalise_text(page.extract_text() or "")
        if text:
            pages.append((index, text))
    return len(reader.pages), pages


def build_document(pages):
    """Join pages into one string and remember where each page starts."""
    parts, page_starts, position = [], [], 0
    for page_number, text in pages:
        page_starts.append([page_number, position])
        parts.append(text)
        position += len(text) + 1  # +1 for the joining space
    return " ".join(parts), page_starts


def main():
    pdf_path = find_pdf(RAW_DIR)
    num_pages, pages = extract_pages(pdf_path)
    text, page_starts = build_document(pages)

    assert num_pages >= MIN_PAGES, f"PDF has {num_pages} pages; the brief needs at least {MIN_PAGES}"
    assert text, "no text extracted - is the PDF scanned images?"

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    document = {"source": pdf_path.name, "num_pages": num_pages, "text": text, "page_starts": page_starts}
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_file:
        json.dump(document, out_file, ensure_ascii=False)

    print(f"Source: {pdf_path.name}")
    print(f"Pages in PDF: {num_pages}, pages with text: {len(pages)}")
    print(f"Words: {len(text.split())}, characters: {len(text)}")
    print(f"First 300 characters: {text[:300]}")
    print(f"Wrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

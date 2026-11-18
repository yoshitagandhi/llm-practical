"""Task 2 step 2: chunk the document in two sizes and embed every chunk.

Input : task2_rag/data/processed/document.json
Output: task2_rag/index/<size>_chunks.jsonl and <size>_embeddings.npy for size in {small, large}

Usage (from the repo root):
    python task2_rag/build_index.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

TASK_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TASK_DIR))
from chunking import chunk_text, page_for_offset  # noqa: E402
from retrieve import CHUNK_CONFIGS, INDEX_DIR, MODEL_NAME, load_model  # noqa: E402

DOCUMENT_FILE = TASK_DIR / "data" / "processed" / "document.json"


def main():
    with open(DOCUMENT_FILE, encoding="utf-8") as doc_file:
        document = json.load(doc_file)

    model = load_model()
    print(f"Embedding model: {MODEL_NAME}, max tokens: {model.max_seq_length}")
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    for size, (chunk_size, overlap) in CHUNK_CONFIGS.items():
        chunks = chunk_text(document["text"], chunk_size, overlap)
        for chunk in chunks:
            chunk["page"] = page_for_offset(document["page_starts"], chunk["start_char"])

        # Check how many chunks are longer than the model can read (they would be truncated).
        token_counts = [len(model.tokenizer(c["text"])["input_ids"]) for c in chunks]
        too_long = sum(1 for n in token_counts if n > model.max_seq_length)

        start = time.time()
        embeddings = model.encode([c["text"] for c in chunks], normalize_embeddings=True,
                                  batch_size=32, show_progress_bar=True)
        seconds = time.time() - start

        np.save(INDEX_DIR / f"{size}_embeddings.npy", embeddings.astype(np.float32))
        with open(INDEX_DIR / f"{size}_chunks.jsonl", "w", encoding="utf-8") as chunk_file:
            for chunk in chunks:
                chunk_file.write(json.dumps(chunk, ensure_ascii=False) + "\n")

        print(f"{size}: {chunk_size} words / {overlap} overlap -> {len(chunks)} chunks, "
              f"tokens per chunk max {max(token_counts)}, over limit {too_long}, "
              f"embedding shape {embeddings.shape}, {seconds:.1f}s")


if __name__ == "__main__":
    main()

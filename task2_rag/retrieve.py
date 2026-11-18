"""Task 2: embedding model settings and top-k search with plain NumPy.

Usage (from the repo root, after build_index.py):
    python task2_rag/retrieve.py "What is an incident response policy?" --size small
"""

import argparse
import json
from pathlib import Path

import numpy as np  # sentence_transformers is imported inside load_model(), only when needed

TASK_DIR = Path(__file__).resolve().parent
INDEX_DIR = TASK_DIR / "index"

MODEL_NAME = "BAAI/bge-small-en-v1.5"
# bge models are trained with this instruction in front of short search queries
# (documents are embedded without it).
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# name -> (chunk_size_words, overlap_words); ~20% overlap for both
CHUNK_CONFIGS = {"small": (100, 20), "large": (350, 70)}


def load_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_NAME)


def load_index(size):
    embeddings = np.load(INDEX_DIR / f"{size}_embeddings.npy")
    with open(INDEX_DIR / f"{size}_chunks.jsonl", encoding="utf-8") as chunk_file:
        chunks = [json.loads(line) for line in chunk_file]
    assert len(chunks) == embeddings.shape[0], "index and chunk list are out of sync"
    return embeddings, chunks


def embed_queries(model, questions):
    return model.encode([QUERY_PREFIX + q for q in questions], normalize_embeddings=True)


def search(query_vector, embeddings, k=5):
    """Return (indices, scores) of the k most similar chunks.

    Vectors are L2-normalised, so the dot product equals cosine similarity.
    """
    scores = embeddings @ query_vector
    top = np.argsort(-scores)[:k]
    return top, scores[top]


def main():
    parser = argparse.ArgumentParser(description="Retrieve the top-k chunks for a question")
    parser.add_argument("question")
    parser.add_argument("--size", choices=CHUNK_CONFIGS, default="small")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    model = load_model()
    embeddings, chunks = load_index(args.size)
    query_vector = embed_queries(model, [args.question])[0]
    indices, scores = search(query_vector, embeddings, args.k)
    for rank, (index, score) in enumerate(zip(indices, scores), start=1):
        chunk = chunks[index]
        print(f"\n#{rank}  score={score:.3f}  page={chunk['page']}  chunk={chunk['chunk_id']}")
        print(chunk["text"][:400] + ("..." if len(chunk["text"]) > 400 else ""))


if __name__ == "__main__":
    main()

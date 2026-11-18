"""Task 2 step 4: answer a question from the retrieved chunks using a local Ollama model.

Usage (from the repo root, Ollama running):
    python task2_rag/answer.py "What should an incident response policy contain?" --size small
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TASK_DIR))
from retrieve import CHUNK_CONFIGS, embed_queries, load_index, load_model, search  # noqa: E402

OLLAMA_URL = "http://localhost:11434/api/generate"
ANSWER_MODEL = "qwen2.5:1.5b-instruct-q4_K_M"


def build_prompt(question, chunks):
    context = "\n\n".join(f"[{i}] (page {c['page']}) {c['text']}" for i, c in enumerate(chunks, start=1))
    return (
        "Answer the question using only the numbered context below. "
        "Cite the numbers of the passages you used, like [1]. "
        "If the context does not contain the answer, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )


def ask_ollama(prompt):
    payload = {"model": ANSWER_MODEL, "prompt": prompt, "stream": False,
               "options": {"temperature": 0, "seed": 42, "num_ctx": 4096}}
    request = urllib.request.Request(OLLAMA_URL, data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read().decode("utf-8"))["response"]


def main():
    parser = argparse.ArgumentParser(description="Retrieval-augmented answer from the document")
    parser.add_argument("question")
    parser.add_argument("--size", choices=CHUNK_CONFIGS, default="small")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    model = load_model()
    embeddings, chunks = load_index(args.size)
    query_vector = embed_queries(model, [args.question])[0]
    indices, _ = search(query_vector, embeddings, args.k)
    top_chunks = [chunks[i] for i in indices]

    answer = ask_ollama(build_prompt(args.question, top_chunks))
    print(f"Question: {args.question}\n")
    print(f"Answer:\n{answer.strip()}\n")
    print("Sources: " + ", ".join(f"[{i}] p.{c['page']}" for i, c in enumerate(top_chunks, start=1)))


if __name__ == "__main__":
    main()

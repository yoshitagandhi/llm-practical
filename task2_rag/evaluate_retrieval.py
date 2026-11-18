"""Task 2 step 3: recall@1 and recall@5 for both chunk sizes on the 15 hand-written questions.

Input : task2_rag/questions.jsonl  - one JSON object per line:
          {"id": 1, "question": "...", "source_passage": "<text copied from the PDF>", "page": 12}
        task2_rag/data/processed/document.json and task2_rag/index/*
Output: task2_rag/results/retrieval_results.json - per-question rank of the first hit, per size
        task2_rag/results/misses.md               - top-3 retrieved chunks for every miss at k=5

Hit rule (approved): a retrieved chunk is a hit if it contains >= 50% of the source passage.

Usage (from the repo root):
    python task2_rag/evaluate_retrieval.py --check-only   # only check every source passage is found
    python task2_rag/evaluate_retrieval.py
"""

import argparse
import json
import sys
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TASK_DIR))
from chunking import find_passage, is_hit  # noqa: E402
from retrieve import CHUNK_CONFIGS, embed_queries, load_index, load_model, search  # noqa: E402

QUESTIONS_FILE = TASK_DIR / "questions.jsonl"
DOCUMENT_FILE = TASK_DIR / "data" / "processed" / "document.json"
RESULTS_DIR = TASK_DIR / "results"
K_VALUES = (1, 5)
MAX_K = 10  # retrieve 10 so we can also see how far down a missed passage sits


def load_questions():
    with open(QUESTIONS_FILE, encoding="utf-8") as q_file:
        questions = [json.loads(line) for line in q_file if line.strip()]
    assert len(questions) >= 15, f"need at least 15 questions, found {len(questions)}"
    return questions


def locate_gold_spans(document_text, questions):
    """Find each source passage in the document; stop with a clear message if any is missing."""
    spans, missing = {}, []
    for q in questions:
        span = find_passage(document_text, q["source_passage"])
        if span is None:
            missing.append(q["id"])
        else:
            spans[q["id"]] = span
    if missing:
        raise SystemExit(
            f"Source passage not found in the document for question ids {missing}. "
            "Copy the passage again from the PDF (exact wording) and re-run."
        )
    return spans


def first_hit_rank(retrieved_indices, chunks, gold_span):
    for rank, index in enumerate(retrieved_indices, start=1):
        if is_hit(chunks[index], *gold_span):
            return rank
    return None


def main():
    parser = argparse.ArgumentParser(description="Recall@k for both chunk sizes")
    parser.add_argument("--check-only", action="store_true",
                        help="only check that every source passage is found in the document")
    args = parser.parse_args()

    questions = load_questions() if not args.check_only else [
        json.loads(line) for line in open(QUESTIONS_FILE, encoding="utf-8") if line.strip()
    ]
    with open(DOCUMENT_FILE, encoding="utf-8") as doc_file:
        document_text = json.load(doc_file)["text"]
    gold_spans = locate_gold_spans(document_text, questions)
    if args.check_only:
        print(f"All {len(questions)} source passages found in the document.")
        return

    model = load_model()
    query_vectors = embed_queries(model, [q["question"] for q in questions])

    results = {"hit_rule": ">= 50% of source passage inside chunk", "sizes": {}}
    miss_lines = ["# Retrieval misses at k=5\n"]

    for size, (chunk_size, overlap) in CHUNK_CONFIGS.items():
        embeddings, chunks = load_index(size)
        per_question = []
        for q, query_vector in zip(questions, query_vectors):
            indices, scores = search(query_vector, embeddings, MAX_K)
            rank = first_hit_rank(indices, chunks, gold_spans[q["id"]])
            per_question.append({"id": q["id"], "first_hit_rank": rank})

            if rank is None or rank > 5:
                miss_lines.append(f"\n## [{size}] Q{q['id']}: {q['question']}\n")
                miss_lines.append(f"First hit rank (top {MAX_K}): {rank}\n")
                miss_lines.append(f"**Source passage (p.{q.get('page', '?')}):** {q['source_passage']}\n")
                for r, (index, score) in enumerate(zip(indices[:3], scores[:3]), start=1):
                    text = chunks[index]["text"]
                    miss_lines.append(f"- Retrieved #{r} (score {score:.3f}, p.{chunks[index]['page']}): "
                                      f"{text[:300]}{'...' if len(text) > 300 else ''}\n")

        recall = {
            f"recall@{k}": sum(1 for p in per_question if p["first_hit_rank"] and p["first_hit_rank"] <= k) / len(questions)
            for k in K_VALUES
        }
        results["sizes"][size] = {
            "chunk_size_words": chunk_size,
            "overlap_words": overlap,
            "num_chunks": len(chunks),
            **recall,
            "per_question": per_question,
        }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "retrieval_results.json", "w", encoding="utf-8") as out_file:
        json.dump(results, out_file, indent=2)
    (RESULTS_DIR / "misses.md").write_text("\n".join(miss_lines), encoding="utf-8")

    print(f"{'size':<7}{'words':>6}{'chunks':>8}{'recall@1':>10}{'recall@5':>10}")
    for size, r in results["sizes"].items():
        print(f"{size:<7}{r['chunk_size_words']:>6}{r['num_chunks']:>8}{r['recall@1']:>10.3f}{r['recall@5']:>10.3f}")
    print("\nPer-question first-hit rank (None = not in top 10):")
    for i, q in enumerate(questions):
        ranks = {size: results["sizes"][size]["per_question"][i]["first_hit_rank"] for size in CHUNK_CONFIGS}
        print(f"  Q{q['id']:>2}: {ranks}")
    print(f"\nWrote {RESULTS_DIR / 'retrieval_results.json'} and misses.md")


if __name__ == "__main__":
    main()

# LLM practical – fine-tuning, RAG from scratch, local inference

Three tasks, one folder each. All numbers below come from runs saved in each task's `results/` folder.
The short PDF report is in `report/`.

| Folder | Task | Ran on |
|---|---|---|
| `task1_lora/` | LoRA fine-tuning of Qwen2.5-0.5B-Instruct: plain-English requests → fixed JSON | Colab, Tesla T4 |
| `task2_rag/` | Question answering over an 80-page PDF, no frameworks (embeddings + NumPy) | Laptop CPU |
| `task3_local/` | Qwen2.5-1.5B-Instruct in Ollama, fp16 vs 4-bit (q4_K_M) | Laptop CPU (no usable GPU) |
| `tests/` | Unit tests for the Task 2 text utilities | – |

Laptop: AMD integrated graphics (not usable by Ollama/PyTorch), 16 GB RAM, Windows, Python 3.14.

## Results at a glance

**Task 1** – 200 held-out test requests (MASSIVE, 4 scenarios, 11 intents, 30 slot types)

| metric | base | fine-tuned |
|---|---|---|
| strict JSON (output is only JSON) | 0.000 | 1.000 |
| intent accuracy | 0.385 | 0.935 |
| slot F1 | 0.142 | 0.843 |
| exact match (intent + all slots) | 0.115 | 0.705 |

**Task 2** – 15 hand-written questions on NIST SP 800-61r2

| chunks | recall@1 | recall@5 |
|---|---|---|
| 100 words (413 chunks) | 0.600 | 0.933 |
| 350 words (118 chunks) | 0.467 | 0.867 |

**Task 3** – same 10 prompts, final run (`task3_local/results/metrics.csv`)

| setting | generation tok/s (median) | memory (Ollama / process) | quality (0–20) |
|---|---|---|---|
| fp16 | 8.22 | 3.0 GB / 3.07 GB | 14 |
| q4_K_M | 20.67 | 1.04 GB / 1.11 GB | 13 |

## How to run

```
pip install -r requirements.txt
```

### Task 1 – LoRA fine-tuning (`task1_lora/`)
1. Download MASSIVE 1.1 from https://github.com/alexa/massive and extract it to
   `task1_lora/data/raw/amazon-massive-dataset-1.1/` (not committed).
2. `python task1_lora/prepare_data.py --inspect` – counts per scenario (writes nothing).
3. `python task1_lora/prepare_data.py` – writes `data/processed/` train / val / test (committed).
4. Open `task1_lora/train_lora.ipynb` in Colab with a T4 GPU, set `REPO_URL`, run all. It trains the adapter,
   then runs `evaluate.py` and saves `results/metrics.json`, `predictions.csv`, `train_log.json`.
- `python task1_lora/evaluate.py --self-test` checks the scoring code without a GPU.
- Analysis of remaining errors: `task1_lora/results/failures.md`.

### Task 2 – RAG from scratch (`task2_rag/`)
```
python task2_rag/ingest.py                         # PDF -> data/processed/document.json
python task2_rag/build_index.py                    # chunks + embeddings for both chunk sizes -> index/
python task2_rag/evaluate_retrieval.py --check-only
python task2_rag/evaluate_retrieval.py             # recall@1/@5 -> results/
python task2_rag/retrieve.py "your question" --size small
python task2_rag/answer.py "your question"         # needs Ollama running
python -m pytest tests
```
- Embedding model: `BAAI/bge-small-en-v1.5`; search is a dot product of normalised vectors in NumPy.
- A retrieved chunk counts as a hit if it contains at least 50% of the question's source passage.
- Questions and source passages: `questions.jsonl`. Misses and analysis: `results/misses.md`, `results/failures.md`.

### Task 3 – local model (`task3_local/`)
```
ollama pull qwen2.5:1.5b-instruct-fp16
ollama pull qwen2.5:1.5b-instruct-q4_K_M
python task3_local/benchmark.py
```
Writes per-prompt outputs and timings, `metrics.csv` and `side_by_side.md`. Scores and notes: `results/observations.md`.

## Sources
- MASSIVE: FitzGerald et al., 2022 – CC BY 4.0, Amazon.com Inc. or its affiliates.
- NIST SP 800-61 Rev. 2, *Computer Security Incident Handling Guide* (2012, withdrawn 2025) – US Government work.
- Models: Qwen2.5-0.5B-Instruct, Qwen2.5-1.5B-Instruct (Qwen team), bge-small-en-v1.5 (BAAI).

# LLM practical – fine-tuning, RAG and local inference

One folder per task:

| Folder | Task | Where it runs |
|---|---|---|
| `task1_lora/` | LoRA fine-tuning of Qwen2.5-0.5B-Instruct: plain-English requests → fixed JSON (MASSIVE dataset) | Colab T4 GPU |
| `task2_rag/` | Retrieval-augmented QA built from scratch (embeddings + FAISS/NumPy) | Laptop CPU |
| `task3_local/` | Running a model locally with Ollama in two quantisation settings | Laptop CPU |
| `report/` | Short PDF report with results and answers | – |

All numbers in the report come from the runs saved in each task's `results/` folder.

## Task 1 quick start

1. Download MASSIVE 1.1 (https://github.com/alexa/massive) and extract it to
   `task1_lora/data/raw/amazon-massive-dataset-1.1/` (not committed; see `.gitignore`).
2. `python task1_lora/prepare_data.py` – writes `task1_lora/data/processed/` (committed).
3. Open `task1_lora/train_lora.ipynb` in Colab with a T4 GPU and run all cells.

Dataset: MASSIVE (FitzGerald et al., 2022), CC BY 4.0, Amazon.com Inc. or its affiliates.

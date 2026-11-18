# LLM Practical: Fine-tuning, RAG and Local Inference

**Yoshita Gandhi**

Code and all raw results: [https://github.com/yoshitagandhi/llm-practical](https://github.com/yoshitagandhi/llm-practical)

All numbers come from my own runs and are saved in each task's `results/` folder. Task 1 ran on a free Colab Tesla T4.
Tasks 2 and 3 ran on my laptop (16 GB RAM, AMD integrated graphics, so CPU only).

## Task 1 – LoRA fine-tuning: plain-English requests → fixed JSON

**Task and data.** Turn a voice-assistant request into `{"intent": ..., "slots": [{"type": ..., "value": ...}]}`.
Data: MASSIVE en-US (CC BY 4.0), 4 scenarios (alarm, calendar, weather, transport), 11 intents, 30 slot types. I used
MASSIVE's own train/dev/test split so no example leaks into the test set: 3,222 train, 580 validation, and 200 test rows
sampled from the official test split (seed 42). Slots are a list, not a dict, because the same slot type can appear twice
(159 rows). Base and fine-tuned models get the identical prompt (allowed intents and slot types + the request) and greedy decoding.

**Model and LoRA settings** (Qwen2.5-0.5B-Instruct, 502.8M parameters; LoRA trains 8.8M = 1.75%)

| setting | value | why |
|---|---|---|
| rank r / alpha | 16 / 32 | This is a focused task, so a smaller rank is enough. Alpha is twice the rank, a common setting that scales the update. |
| target modules | all linear: q, k, v, o, gate, up, down | The model needs to learn the JSON format and the task labels, so I trained all the main linear layers. |
| dropout | 0.05 | A small amount of regularisation helps with about 3,000 examples. |
| learning rate | 2e-4, cosine, 3% warmup | Because LoRA updates only a small part of the model, it can use a higher learning rate than full fine-tuning. |
| epochs / batch | 2 / 16 (8 × 2 accumulation) | Two passes should teach the format; the validation loss helps show if training goes on too long. |
| loss | on the JSON answer only | This teaches the model to produce the JSON answer instead of copying the request. |

Training took 13.8 min with a peak of 2.58 GiB GPU memory. Validation loss fell from 0.0670 (step 50) to 0.0367 at step 350
and stayed flat to the end (step 404), with no rise, so 2 epochs did not overfit.

**Results on 200 held-out requests**

| metric | base | fine-tuned |
|---|---|---|
| strict JSON rate (output is only JSON) | 0.000 | **1.000** |
| intent accuracy | 0.385 | **0.935** |
| slot precision / recall / F1 | 0.188 / 0.113 / 0.142 | **0.829 / 0.858 / 0.843** |
| exact match (intent and every slot correct) | 0.115 | **0.705** |

Slot scores compare (type, value) pairs. The base model wrapped every answer in ```` ```json ```` fences (so strict JSON = 0,
though all 200 could be parsed) and often invented values that are not in the request (e.g. "2023-10-05", "18:00").

**Where the fine-tuned model still fails** (59 of 200 not exact; calendar has the most errors, 34/94)

1. *Inconsistent span labels.* "rain or snow" is one gold span (id 3407) but "mild or heavy" is two (id 790); the training examples
   use both labelling styles, so the model has no clear rule to follow.
2. *Rare slot type.* "accounting department" should be `business_name` but was tagged `relation` (id 6878). `business_name`
   has 13 training examples against 87 for `relation`, and departments are labelled three different ways in training.
3. *Short, vague requests.* "plan for tomorrow" → `calendar_set` instead of `calendar_query` (id 7569). With few cue words the
   model falls back to the largest class (`calendar_set`, 810 training rows vs 566).

**GPU memory to fine-tune a 7B model** (mixed precision, Adam; activations come on top and depend on batch and sequence length)

- *(a) Full fine-tuning:* per parameter: fp16 weights 2 B + fp16 gradients 2 B + fp32 master weights 4 B + Adam moments 8 B
  = 16 B. 7 × 10⁹ × 16 B ≈ **112 GB** + activations → several 80 GB GPUs.

- *(b) LoRA (r = 16, all linear layers):* frozen fp16 base = 7 × 10⁹ × 2 B = 14 GB. Adapters: each d_in × d_out matrix adds
  r × (d_in + d_out) parameters. With Llama-2-7B sizes (hidden 4096, MLP 11008, 32 layers):
  32 × [4 × 16 × (4096 + 4096) + 3 × 16 × (4096 + 11008)] = **40.0M** parameters (0.57%) × 16 B = 0.64 GB.
  Total ≈ 14 + 0.64 + activations ≈ **17–20 GB** → fits one 24 GB GPU.

- *(c) QLoRA:* base in 4-bit NF4 = 0.5 B/parameter = 3.5 GB (+ ~0.1 GB quantisation constants) + 0.64 GB adapters +
  activations ≈ **6–9 GB** → fits a free 16 GB T4.

- *Check against my run:* 0.5B base in fp32 = 502.8M × 4 B = 1.87 GiB, LoRA state = 8.8M × 16 B = 0.13 GiB, total 2.0 GiB.
  I measured 2.58 GiB, so about 0.6 GiB went to activations – the method matches reality.

## Task 2 – Retrieval-augmented QA from scratch

**Setup.** Document: NIST SP 800-61 Rev. 2, *Computer Security Incident Handling Guide* (80 pages, 33,031 words, public
domain). Pipeline (no frameworks): pypdf text extraction → normalisation (ligatures, PDF bullet symbols, curly quotes,
whitespace) → word-based chunks with ~20% overlap → `bge-small-en-v1.5` embeddings → cosine similarity as a NumPy dot
product of normalised vectors → top-5 chunks + question → local Qwen2.5-1.5B (q4_K_M) in Ollama writes the answer with
citations. I wrote 15 questions across chapters 2–4 and the appendices, each with its exact source passage (full list in task2_rag/questions.jsonl); a few are
deliberately hard (paraphrases, a list, a definition that also appears in the FAQ). A retrieved chunk counts as a **hit if it
contains at least 50% of the source passage**, which is fair across chunk sizes. End to end, the system answered correctly:
asked which criteria decide the containment strategy, it listed all six from p.45; asked how precursors differ from
indicators, it gave the p.36 definition with citations (`task2_rag/results/example_answers.md`). It did not always use the
requested [n] citation format.

| chunk size | chunks | recall@1 | recall@5 |
|---|---|---|---|
| 100 words, 20 overlap | 413 | **0.600** | **0.933** |
| 350 words, 70 overlap | 118 | 0.467 | 0.867 |

**Small chunks worked better.** Most questions ask about one or two sentences. A 100-word chunk usually focuses on one idea, so it can match a specific question more closely. A 350-word chunk combines more ideas, which can make the match less clear. Large chunks are also more likely to cross section boundaries; 13 of the 118 exceeded the model's 512-token limit and had their endings cut off. Small chunks give the language model less surrounding detail for broad questions.

**Two failures**

1. *Q10 (both sizes): "Should incidents be handled in the order they are reported?"* The guide says "first-come, first-served". The same rule appears on pages 42 and 54, and prioritisation is also discussed in the executive summary and FAQ. The search found other related passages, some of which could answer the question, but the page 42 passage did not appear in the top 10.
   *Fix:* mark all passages that answer each question as correct; combine exact-word search with meaning-based search; reword the question before searching; and re-rank a larger set of results.
2. *Q14 (large only): why is information sharing mutually beneficial?* The only large chunk with the answer starts in chapter 3. The answer begins at word 251 of 350, just after the chapter 4 heading, so about 70% of the chunk covers another topic. The small chunk with the same passage ranked first. *Fix:* split at section headings before making size-based chunks, or search small chunks and then give the language model nearby text for context ("small to big").

Data-quality note: pypdf splits some words ("pr ovide"), and the table-of-contents dot leaders produce chunks of up to
3,904 tokens. I kept the text as extracted and report it rather than adding cleaning rules.

**What BM25 adds.** BM25 ranks passages by the words they share with the question. It gives more importance to uncommon words and accounts for passage length. Embeddings match meaning, but may miss exact terms such as acronyms (MSSP), names, numbers, and error codes. Using both methods together can find passages either one might miss.

**What a re-ranker does.** First, quickly find a larger set of possible passages, such as the top 20–50 from word and meaning searches. Then a re-ranker (usually a cross-encoder model) reads the question and each passage together and sorts them by how well they answer it. This gives a more careful ranking, but takes too long to check every passage. It can only sort passages that the first search already found.

## Task 3 – Running a model locally and quantisation

**Setup.** Qwen2.5-1.5B-Instruct in Ollama on my laptop CPU (Ollama reported 0 GB on GPU), fp16 vs q4_K_M (4-bit). Same 10
prompts (factual, reasoning, code, format, summary), temperature 0, seed 42, max 256 new tokens, context 2,048, one untimed
warm-up. Speed = Ollama's `eval_count / eval_duration`. Memory = Ollama's reported model size and the peak resident memory
of its `llama-server` process.

| setting | gen. tok/s mean / median (final run) | other runs (mean) | prompt tok/s | memory: Ollama / process | quality |
|---|---|---|---|---|---|
| fp16 | 8.08 / 8.22 | 9.40, 8.58 | 92.7 | 3.0 GB / 3.07 GB | 14 / 20 |
| q4_K_M | 18.91 / 20.67 | 18.64, 26.34 | 151.6 | 1.04 GB / 1.11 GB | 13 / 20 |

**Observations.** q4_K_M used about a third of the memory and generated text about 2× faster in every run. Quality scores were close (13/20 versus 14/20): both models answered the facts, logic puzzle, code, and bullet-format prompts correctly. Both made the same mistake on the list question and both put JSON inside code fences when asked for JSON only. The only score difference was prompt 4: q4_K_M rounded too early and answered 80.04 instead of 80. One example is not enough to say quantisation caused this. Answers were the same across runs, but CPU speed varied by up to about 40%, likely because of background activity, heat, and laptop power settings. I report the medians and all runs.

**What the KV cache is.** When the model creates a new token, it compares it with earlier tokens using key (K) and value (V) data from each layer. It saves that data as it goes instead of calculating it again each time. This saved data is called the KV cache. It grows with the number of tokens:
bytes per token = 2 (K, V) × layers × KV heads × head size × bytes per number. For Qwen2.5-1.5B (28 layers, 2 KV heads, head size 128, fp16) that is 28 KB per token, so ~56 MB for my 2,048-token context – small next to the 3 GB of weights, but it becomes the main cost when many users have long contexts.

**Why quantisation saves memory.** Quantisation stores each weight using fewer bits: 4 rather than 16, with a little extra information per group of weights. For q4_K_M, this works out to about 4.5–5 bits per weight on average. The model file shrank from 3.1 GB to 986 MB. Since the computer reads the weights to generate each token, reading fewer bytes can make generation faster when memory speed is the limit, as it was on my CPU. The speed gain (~2×) is smaller than the file size reduction (~3×) because the CPU must unpack the 4-bit values. The trade-off is a small loss of precision.

**Plan: serving 20 concurrent users on one 24 GB GPU.**
*Assumptions:* model Qwen2.5-7B-Instruct (28 layers, 4 KV heads, head size 128), up to 4,096 tokens of context per user,
fp16 KV cache, served with vLLM.

- KV cache per token = 2 × 28 × 4 × 128 × 2 B = 56 KB → 20 users × 4,096 tokens ≈ **4.4 GB** (8.75 GB at 8,192 tokens).
- fp16 weights ≈ 14.2 GB → 14.2 + 4.4 ≈ 18.6 GB + ~1–2 GB runtime overhead → fits, but with little headroom.
- Better: 4-bit AWQ weights (≈ 4–5 GB) leave ~15 GB for KV cache → longer contexts or more users, small quality cost.
- vLLM's PagedAttention gives each chat KV memory in small blocks as needed, so a short chat does not hold space for a full context. Continuous batching groups active requests together at each step. This lets the 20 users share each read of the model weights and increases total speed, with a small slowdown for each user.
- To avoid running out of memory, limit how many tokens each request can use and how many requests can run at once. Put additional users in a queue.

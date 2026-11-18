# Task 2 – retrieval failures (DRAFT – rewrite in your own words)

Setup: bge-small-en-v1.5, cosine similarity in NumPy, 15 questions, hit = retrieved chunk holds >= 50% of the source passage.

| chunk size | chunks | recall@1 | recall@5 |
|---|---|---|---|
| small (100 words, 20 overlap) | 413 | 0.600 | 0.933 |
| large (350 words, 70 overlap) | 118 | 0.467 | 0.867 |

Misses at k=5: Q10 (both sizes, not even in the top 10) and Q14 (large only, not in the top 10). Small chunks also found
the right chunk at rank 1 more often (9/15 vs 7/15).

## Failure 1 – Q10: the question uses different words, and the answer appears in several places

Question: "Should incidents be handled in the order they are reported?"
Source passage (p.42): "Incidents should not be handled on a first -come, first -served basis as a result of resource limitations..."

What was retrieved instead: other passages about prioritising incidents – the executive summary (p.13, "Prioritizing the handling
of individual incidents is a critical decision point"), the chapter 3 recommendations (p.53-54) and the FAQ (p.76, "Generally,
the highest priority is ..."). These are on topic and some would answer the question, but none contains the p.42 passage.

Why:
- Vocabulary gap: the question says "in the order they are reported"; the text says "first-come, first-served". The embedding
  has to bridge that paraphrase, and many other chunks about prioritisation look just as similar.
- The same statement appears twice (p.42 and p.54), and the topic is repeated in the executive summary and FAQ. With one gold
  passage per question, a correct answer from another section still counts as a miss.
- pypdf extracted the phrase as "first -come, first -served", which slightly changes how it is tokenised.

What would fix it: label every passage that answers the question (several gold passages); add BM25 so a query that uses the
document's own wording ("first-come first-served") gets a strong exact-match boost; rewrite or expand the query before searching;
re-rank a larger candidate list (e.g. top 20) with a cross-encoder.

## Failure 2 – Q14 (large chunks only): the answer sits at the edge of a chunk that is mostly about something else

Question: "Why is sharing incident information between organizations often beneficial to both sides?"
Source passage (p.55): "Incident information sharing is frequently mutually beneficial because the same threats and attacks often affect multiple organizations simultaneously."

The only large chunk that contains the passage (chunk 82) starts on p.54 with the end of chapter 3 (evidence, legal staff,
recommendations). The passage only starts at word 251 of 350, right after the "4. Coordination and Information Sharing" heading.
So about 70% of that chunk is about a different topic, and its embedding is an average dominated by chapter 3.
The next chunk (83), ranked #1, starts just after the passage, so it talks about information sharing but does not contain the answer.
The small chunk containing the passage (chunk 290) is mostly about this topic and was ranked #1.

Why: a fixed-size chunk that crosses a section boundary mixes two topics, and one embedding per chunk cannot represent both.
Bigger chunks make this more likely. 13 of the 118 large chunks are also longer than the model's 512-token limit and get cut off.

What would fix it: split on section headings first and then by size (structure-aware chunking); smaller chunks or more overlap;
or retrieve small chunks and return their surrounding text to the LLM ("small to big").

## Why small chunks worked better here
Most questions target one or two specific sentences. A 100-word chunk is mostly about one idea, so its embedding is close to a
specific question. A 350-word chunk averages several ideas together, which blurs the match and makes chunks that cross topics more
likely. The trade-off: small chunks give the LLM less surrounding context, so for broad questions a larger chunk or "small to big"
retrieval can be better.

# Task 2 – retrieval failures

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

What was retrieved instead: The search returned other passages about incident priority: the executive summary (p.13), recommendations in chapter 3 (pp.53–54), and the FAQ (p.76). These passages are related, and some could answer the question, but they do not include the passage on p.42.

Why:
- The question says "in the order they are reported," while the guide says "first-come, first-served." The search must connect these different phrases, and many other passages about priority look similar.
- The statement appears on both p.42 and p.54, and the topic also appears in the executive summary and FAQ. Since only one passage was marked as correct, finding another passage that answers the question still counted as a miss.
- The PDF text extraction added spaces in "first -come, first -served," which may affect how the search reads the phrase.

What would fix it: mark every passage that answers the question as correct; add exact-word search so terms from the guide get extra weight; reword or expand the question before searching; and re-rank a larger set of results, such as the top 20.

## Failure 2 – Q14 (large chunks only): the answer sits at the edge of a chunk that is mostly about something else

Question: "Why is sharing incident information between organizations often beneficial to both sides?"
Source passage (p.55): "Incident information sharing is frequently mutually beneficial because the same threats and attacks often affect multiple organizations simultaneously."

The only large chunk with the passage (chunk 82) begins on p.54, near the end of chapter 3, where the text covers evidence, legal staff, and recommendations. The answer begins at word 251 of 350, just after the heading for section 4. About 70% of the chunk is on another topic, so its overall meaning is mostly about chapter 3. The next chunk (83), ranked first, starts after the answer. It discusses information sharing but does not contain the answer itself. The small chunk with the passage (chunk 290) is mostly about this topic and ranked first.

Why: A chunk that crosses into another section can mix two topics, while its search representation has to capture both at once. This happens more often with larger chunks. Also, 13 of the 118 large chunks exceeded the model's 512-token limit, so their endings were cut off.

What would fix it: split at section headings before dividing the text by size; use smaller chunks or more overlap; or search small chunks first and then give the language model the nearby text for extra context ("small to big").

## Why small chunks worked better here
Most questions ask about one or two specific sentences. A 100-word chunk usually focuses on one idea, so it can match a specific question more closely. A 350-word chunk combines more ideas, which can make the match less clear and increases the chance that it crosses into another topic. Small chunks give the language model less surrounding context, so broad questions may benefit from a larger chunk or the "small to big" approach.
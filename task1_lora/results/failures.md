# Task 1 – where the fine-tuned model still fails (DRAFT – rewrite in your own words)

Fine-tuned model: 141 / 200 exact matches (0.705). Of the 59 misses: 15 extra slot, 12 wrong slot type,
11 right slot type with different span boundaries, 8 missed slot, 9 wrong intent, 4 wrong intent + slots.
Errors by scenario: calendar 34/94, weather 15/43, transport 8/40, alarm 2/23.
(All numbers from results/predictions.csv and data/processed/train.jsonl.)

## Failure 1 – span boundaries that the labels themselves do not agree on

| id | request | correct (gold) | fine-tuned |
|---|---|---|---|
| 3407 | are there any forecasts of rain or snow this week | `rain or snow` (one span) | `rain`, `snow` (two spans) |
| 790 | mild or heavy weather | `mild`, `heavy` (two spans) | `mild or heavy` (one span) |

Why: the training data has no consistent rule for "X or Y". Most weather examples keep it as one span
(`hot or cold`, `light or heavy jacket`, `a dress or a sweat shirt`), but some split it (`sandals` / `sneakers`).
The model sees both patterns, so it cannot learn which one the annotator will use. In 790 the model actually
follows the more common training pattern and the gold label is the odd one out. Exact match punishes this
even though the model's answer is reasonable. Fix: clearer annotation guidelines, or a lenient slot metric.

## Failure 2 – a rare slot type loses to a common one

| id | request | correct | fine-tuned |
|---|---|---|---|
| 6878 | schedule a meeting with the accounting department on two thirty p. m. on friday | `business_name: accounting department` | `relation: accounting department` |

Why: `business_name` appears only 13 times in all 3,222 training rows (4 in calendar), while `relation`
appears 87 times. Departments are also labelled inconsistently in training: "h. r. department" is `person`,
"the v. p. of the department" is `relation`, "police department" is `business_name`. With so few and
conflicting examples, LoRA has little signal and falls back to a label it has seen more often. Everything
else in this example (intent, event, time, date) is right. Fix: more examples of rare slot types, or merge
labels that annotators do not separate consistently.

## Failure 3 – short, vague requests default to the most common intent

| id | request | correct | fine-tuned |
|---|---|---|---|
| 7569 | plan for tomorrow | `calendar_query` | `calendar_set` |
| 9116 | could you reschedule my meeting | `calendar_query` | `calendar_set` |

Why: these requests have only 3-5 words and no clear "what / show me" or "add / set" cue. `calendar_set`
is the largest class in training (810 rows vs 566 for `calendar_query`), so when the words do not decide
the intent the model leans towards the majority class. The slots are right in both cases. For 9116 the
gold label is also debatable: rescheduling changes the calendar, which sounds more like `set` than `query`.
Fix: class weighting or more short examples, and checking doubtful gold labels.

## Note on the base model (for comparison)
The base model wrapped every answer in ```json fences (strict JSON rate 0.000, but all 200 could be parsed).
It often invented values that are not in the request, e.g. date "2023-10-05" and time "18:00" for id 6878,
instead of copying the words from the request, which is why its slot F1 is only 0.142.

# Task 1 – where the fine-tuned model still fails

Fine-tuned model: 141 / 200 exact matches (0.705). Of the 59 misses: 15 extra slot, 12 wrong slot type,
11 right slot type with different span boundaries, 8 missed slot, 9 wrong intent, 4 wrong intent + slots.
Errors by scenario: calendar 34/94, weather 15/43, transport 8/40, alarm 2/23.
(All numbers from results/predictions.csv and data/processed/train.jsonl.)

## Failure 1 – span boundaries that the labels themselves do not agree on

| id | request | correct (gold) | fine-tuned |
|---|---|---|---|
| 3407 | are there any forecasts of rain or snow this week | `rain or snow` (one span) | `rain`, `snow` (two spans) |
| 790 | mild or heavy weather | `mild`, `heavy` (two spans) | `mild or heavy` (one span) |

Why: The training examples do not follow one clear rule for phrases like "X or Y." Most weather examples keep the phrase together, but some split it into two parts. The model sees both styles and cannot know which one the labeller expects. In example 790, it uses the more common style, even though the gold answer uses the less common one. Exact match marks this as wrong even though the answer makes sense. Fix: give labellers clearer rules, or use a score that allows reasonable differences in span boundaries.

## Failure 2 – a rare slot type loses to a common one

| id | request | correct | fine-tuned |
|---|---|---|---|
| 6878 | schedule a meeting with the accounting department on two thirty p. m. on friday | `business_name: accounting department` | `relation: accounting department` |

Why: The training data has only 13 `business_name` examples (4 in calendar), compared with 87 for `relation`. It also labels departments in different ways: "h. r. department" is `person`, "the v. p. of the department" is `relation`, and "police department" is `business_name`. With so few mixed examples, the model has little guidance and chooses the more common label. The intent, event, time, and date are all correct. Fix: add examples of rare slot types, or combine labels that are not used consistently.

## Failure 3 – short, vague requests default to the most common intent

| id | request | correct | fine-tuned |
|---|---|---|---|
| 7569 | plan for tomorrow | `calendar_query` | `calendar_set` |
| 9116 | could you reschedule my meeting | `calendar_query` | `calendar_set` |

Why: These requests are only a few words long and do not clearly say whether the person wants to view or change a calendar. `calendar_set` is more common in training (810 examples versus 566 for `calendar_query`), so the model tends to choose it when the wording is unclear. The slots are correct in both examples. The gold label for 9116 is also open to debate, since rescheduling changes the calendar. Fix: add more short examples, give less common intents more weight during training, and review uncertain labels.

## Note on the base model (for comparison)
The base model put every answer inside ```json fences. None counted as strict JSON, although all 200 answers could still be read. It also sometimes made up details that were not in the request, such as the date "2023-10-05" and time "18:00" for id 6878, instead of using the request's own words. This is why its slot F1 score was only 0.142.
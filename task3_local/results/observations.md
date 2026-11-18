# Task 3 – quality scores and observations

Model: Qwen2.5-1.5B-Instruct in Ollama, CPU only (size_vram = 0).
Settings: temperature 0, seed 42, max 256 new tokens, context 2048.
Rubric: 0 = wrong or ignores the instruction, 1 = partly right, 2 = correct and follows the instruction.

| # | Category | fp16 | q4_K_M | Notes |
|---|---|---|---|---|
| 1 | factual | 2 | 2 | Both "Canberra", one word. |
| 2 | factual | 2 | 2 | Both correct (Jane Austen, 19th century). q4 also gives 1813 but is much longer (71 vs 23 tokens). |
| 3 | factual | 1 | 1 | Asked for two sentences, both gave one. fp16 is more complete (CO2 + water -> glucose + oxygen); q4 leaves out the inputs and oxygen. |
| 4 | reasoning | 1 | 0 | fp16 gets 80 but is not "brief" and hits the 256-token limit mid-sentence. q4 rounds 20/3 to 6.67 first and answers 80.04, which is wrong. |
| 5 | reasoning | 2 | 2 | Both "yes" with a correct one-sentence explanation. |
| 6 | code | 2 | 2 | Both return a working function (code only). Both strip all non-alphanumeric characters, slightly more than "spaces", and the comment says "remove spaces". |
| 7 | code | 0 | 0 | Both say it prints 2, then reason that x is still [1, 2, 3] (length 3). The right answer is 4 because y and x are the same list. Same mistake in both settings. |
| 8 | format | 1 | 1 | Content right, but both wrap the JSON in ```json fences although the prompt says "only the JSON". |
| 9 | format | 2 | 2 | Both give exactly 3 bullets, all under 10 words. |
| 10 | summary | 1 | 1 | One sentence each, both miss the reason (students asked for evening study space). fp16 names the events budget, q4 just says "a budget shift". |
| | **Total** | **14 / 20** | **13 / 20** | |

## Speed and memory (three full runs of the same 10 prompts)

| Run | fp16 mean / median tok/s | q4_K_M mean / median tok/s | Notes |
|---|---|---|---|
| 1 | 9.40 / 9.14 | 18.64 / 18.58 | |
| 2 | 8.58 / 9.16 | 26.34 / 25.10 | |
| 3 (final, metrics.csv) | 8.08 / 8.22 | 18.91 / 20.67 | peak RSS fp16 3.07 GB, q4 1.11 GB |

Memory reported by Ollama (/api/ps): fp16 3.0 GB, q4_K_M 1.04 GB. Measured during run 2 with Get-Process: llama-server 3,069 MB working set while fp16 was loaded.
Prompt processing: fp16 about 93-97 tok/s, q4_K_M about 151-154 tok/s.

## Observations

- q4_K_M generated text about 2× faster than fp16 in every run and used about one third as much memory (1.04 versus 3.0 GB). To create each token, the computer reads the model weights from memory. q4_K_M has about 3× fewer bytes to read, but the speed gain was closer to 2×, likely because the CPU also needs to unpack the 4-bit values.
- The quality scores were close: 13/20 for q4_K_M and 14/20 for fp16. The only score difference was prompt 4: q4_K_M rounded too early and answered 80.04. One example is not enough to say quantisation caused this.
- The largest mistake, on prompt 7, was the same in both settings. This points to a limitation of the 1.5B model rather than quantisation.
- The answers were the same in every run (token counts of 789 and 719) because the temperature was 0 and the seed stayed fixed. Speed varied: q4_K_M ranged from 18.6 to 26.3 tokens per second, likely due to background activity, heat, and laptop power settings.
- Very short answers (3–30 tokens) make speed measurements less steady, so the median is more useful than the mean.

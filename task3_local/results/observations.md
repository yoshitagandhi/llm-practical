# Task 3 – quality scores and observations (DRAFT – review and reword)

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

- q4_K_M was about 2x faster than fp16 at generation in every run, and used about a third of the memory (1.04 vs 3.0 GB). Generating each token means reading all the weights from memory, and q4 has about 3x fewer bytes to read. The gain is closer to 2x than 3x, probably because the CPU also has to unpack the 4-bit values.
- Quality dropped only a little: 13/20 vs 14/20. The only score difference is prompt 4, where q4 rounded too early and got 80.04. One prompt is too little evidence to blame quantisation for it.
- The biggest failure (prompt 7) is identical in both settings, so it is a limit of the 1.5B model, not of quantisation.
- Outputs were the same in every run (same token counts: 789 and 719) because temperature is 0 and the seed is fixed. Speed was not: CPU speed changed by up to about 40% between runs for q4 (18.6 vs 26.3 tok/s), probably because of background load, heat and power settings on a laptop.
- Short answers (3-30 tokens) give noisy tok/s values, so the median is more reliable than the mean.

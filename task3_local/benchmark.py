"""Task 3: benchmark one model in two quantisation settings with Ollama (CPU).

Both settings get the same 10 prompts and the same generation options, so
speed, memory and output quality can be compared directly.

Usage (from the repo root, with Ollama running):
    python task3_local/benchmark.py --limit 2     # quick smoke test
    python task3_local/benchmark.py               # full run

Outputs (task3_local/results/):
    raw_<setting>.jsonl      - per-prompt timings and full output
    metrics.csv              - one summary row per setting
    side_by_side.md          - both outputs per prompt, with blank score columns for manual scoring

How the numbers are measured:
    generation tok/s = eval_count / eval_duration        (reported by Ollama itself)
    prompt tok/s     = prompt_eval_count / prompt_eval_duration
    ollama_size_gb   = memory Ollama reports for the loaded model (/api/ps: weights + KV cache + buffers)
    peak_rss_gb      = largest resident memory of the Ollama processes seen during the run (needs psutil)
"""

import argparse
import csv
import json
import statistics
import time
import urllib.request
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent
RESULTS_DIR = TASK_DIR / "results"
PROMPTS_FILE = TASK_DIR / "prompts.json"
OLLAMA_URL = "http://localhost:11434"

SETTINGS = {
    "fp16": "qwen2.5:1.5b-instruct-fp16",
    "q4_K_M": "qwen2.5:1.5b-instruct-q4_K_M",
}

# Identical for both settings. num_ctx is fixed so the KV cache is the same size.
OPTIONS = {"temperature": 0, "seed": 42, "num_predict": 256, "num_ctx": 2048}
WARMUP_PROMPT = "Reply with the single word: ready."

try:
    import psutil  # optional: only used for the process memory measurement
except ImportError:
    psutil = None


def post_json(path, payload, timeout=600):
    request = urllib.request.Request(
        OLLAMA_URL + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def get_json(path, timeout=30):
    with urllib.request.urlopen(OLLAMA_URL + path, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def generate(model, prompt):
    payload = {"model": model, "prompt": prompt, "stream": False, "options": OPTIONS}
    return post_json("/api/generate", payload)


def unload(model):
    """Ask Ollama to unload a model so the next setting starts from a clean memory state."""
    post_json("/api/generate", {"model": model, "keep_alive": 0})


def loaded_model_info(model):
    """Memory figures Ollama reports for a loaded model (None if not loaded)."""
    for entry in get_json("/api/ps").get("models", []):
        if entry.get("name") == model or entry.get("model") == model:
            return entry
    return None


def ollama_rss_gb():
    """Total resident memory of all Ollama processes, in GB (None without psutil)."""
    if psutil is None:
        return None
    total = 0
    for process in psutil.process_iter(["name", "memory_info"]):
        name = (process.info.get("name") or "").lower()
        # Ollama runs the model in a separate "llama-server" process; "llama" matches both it and
        # "ollama". "ollama app.exe" is the Windows tray icon, not the model, so it is skipped.
        if "llama" in name and "app" not in name and process.info.get("memory_info"):
            total += process.info["memory_info"].rss
    return total / 1024**3


def tokens_per_second(count, duration_ns):
    return count / (duration_ns / 1e9) if count and duration_ns else 0.0


def run_setting(setting, model, prompts):
    print(f"\n=== {setting} ({model}) ===")
    generate(model, WARMUP_PROMPT)  # loads the model; not timed
    info = loaded_model_info(model) or {}
    peak_rss = ollama_rss_gb()

    rows = []
    for item in prompts:
        start = time.time()
        reply = generate(model, item["prompt"])
        wall_seconds = time.time() - start
        rss = ollama_rss_gb()
        if rss is not None:
            peak_rss = max(peak_rss or 0, rss)

        row = {
            "setting": setting,
            "model": model,
            "id": item["id"],
            "category": item["category"],
            "prompt": item["prompt"],
            "output": reply.get("response", ""),
            "eval_count": reply.get("eval_count"),
            "gen_tok_per_s": round(tokens_per_second(reply.get("eval_count"), reply.get("eval_duration")), 2),
            "prompt_tok_per_s": round(tokens_per_second(reply.get("prompt_eval_count"), reply.get("prompt_eval_duration")), 2),
            "wall_seconds": round(wall_seconds, 2),
        }
        rows.append(row)
        print(f"  prompt {item['id']:>2}: {row['eval_count']} tokens, {row['gen_tok_per_s']} tok/s")

    summary = {
        "setting": setting,
        "model": model,
        "prompts": len(rows),
        "mean_gen_tok_per_s": round(statistics.mean(r["gen_tok_per_s"] for r in rows), 2),
        "median_gen_tok_per_s": round(statistics.median(r["gen_tok_per_s"] for r in rows), 2),
        "mean_prompt_tok_per_s": round(statistics.mean(r["prompt_tok_per_s"] for r in rows), 2),
        "total_output_tokens": sum(r["eval_count"] or 0 for r in rows),
        "ollama_size_gb": round(info["size"] / 1024**3, 2) if info.get("size") else None,
        "ollama_size_vram_gb": round(info["size_vram"] / 1024**3, 2) if "size_vram" in info else None,
        "peak_rss_gb": round(peak_rss, 2) if peak_rss is not None else None,
    }
    unload(model)
    return rows, summary


def write_outputs(all_rows, summaries, prompts):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for setting in SETTINGS:
        with open(RESULTS_DIR / f"raw_{setting}.jsonl", "w", encoding="utf-8") as out_file:
            for row in all_rows.get(setting, []):
                out_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(RESULTS_DIR / "metrics.csv", "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    lines = ["# Task 3 – outputs side by side\n",
             "Score each output 0 (wrong / ignores instructions), 1 (partly right), 2 (correct and follows instructions).\n"]
    for item in prompts:
        lines.append(f"\n## Prompt {item['id']} ({item['category']})\n")
        lines.append(f"**Prompt:** {item['prompt']}\n")
        lines.append(f"**Expected:** {item['expected']}\n")
        for setting in SETTINGS:
            match = [r for r in all_rows.get(setting, []) if r["id"] == item["id"]]
            if match:
                lines.append(f"\n**{setting}** ({match[0]['eval_count']} tokens, {match[0]['gen_tok_per_s']} tok/s) – score: __\n")
                lines.append("```\n" + match[0]["output"].strip() + "\n```\n")
    (RESULTS_DIR / "side_by_side.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Ollama in two quantisation settings")
    parser.add_argument("--limit", type=int, default=None, help="only run the first N prompts")
    args = parser.parse_args()

    with open(PROMPTS_FILE, encoding="utf-8") as prompts_file:
        prompts = json.load(prompts_file)[: args.limit]

    try:
        get_json("/api/version")
    except OSError as error:
        raise SystemExit(f"Cannot reach Ollama at {OLLAMA_URL}. Is it running? ({error})")
    if psutil is None:
        print("psutil not installed: peak_rss_gb will be empty (pip install psutil to record it)")

    all_rows, summaries = {}, []
    for setting, model in SETTINGS.items():
        rows, summary = run_setting(setting, model, prompts)
        all_rows[setting] = rows
        summaries.append(summary)

    write_outputs(all_rows, summaries, prompts)
    print("\nSummary:")
    for summary in summaries:
        print(json.dumps(summary))
    print(f"\nWrote results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()

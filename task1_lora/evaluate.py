"""Task 1 evaluation: base model vs LoRA fine-tuned model on the held-out test set.

Runs on a GPU (Colab T4). Both models get the same prompt (prompt.py) and
greedy decoding, so the only difference is the LoRA adapter.

Usage (from the repo root):
    python task1_lora/evaluate.py --adapter task1_lora/outputs/qwen05b-lora
    python task1_lora/evaluate.py --adapter ... --limit 16    # quick smoke test
    python task1_lora/evaluate.py --self-test                 # check scoring code, no GPU needed

Outputs:
    task1_lora/results/predictions.csv  - one row per test example, both models' raw outputs
    task1_lora/results/metrics.json     - metrics for both models

Metrics:
    strict_json_rate - output parses as JSON exactly as generated (no fences, no extra text)
    parsed_rate      - a JSON object could be extracted (fences / extra text stripped)
    intent_accuracy  - predicted intent == gold intent
    slot_precision / slot_recall / slot_f1 - micro-averaged over (type, value) pairs
    exact_match      - intent correct AND the slot pairs match exactly
Unparseable outputs count as wrong on every content metric.
"""

import argparse
import csv
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TASK_DIR))  # so "import prompt" works from the repo root
from prompt import build_messages  # noqa: E402

PROCESSED_DIR = TASK_DIR / "data" / "processed"
RESULTS_DIR = TASK_DIR / "results"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


# ---------- parsing and scoring (pure Python, no GPU) ----------

def parse_strict(text):
    """Return a dict only if the whole output is valid JSON, else None."""
    try:
        parsed = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def parse_lenient(text):
    """Try strict first, then strip ```json fences, then take the outermost {...}."""
    parsed = parse_strict(text)
    if parsed is not None:
        return parsed
    fence_match = FENCE_PATTERN.search(text)
    if fence_match:
        parsed = parse_strict(fence_match.group(1))
        if parsed is not None:
            return parsed
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return parse_strict(text[start:end + 1])
    return None


def slot_pairs(parsed):
    """Turn a parsed prediction's slots into a Counter of (type, value) pairs.

    Values are lower-cased and stripped (MASSIVE utterances are already
    lower-case). Malformed slot entries are ignored.
    """
    pairs = Counter()
    if not parsed or not isinstance(parsed.get("slots"), list):
        return pairs
    for slot in parsed["slots"]:
        if isinstance(slot, dict) and "type" in slot and "value" in slot:
            pairs[(str(slot["type"]).strip(), str(slot["value"]).strip().lower())] += 1
    return pairs


def score_predictions(gold_rows, outputs):
    """Compute all metrics for one model's outputs."""
    n = len(gold_rows)
    strict_ok = parsed_ok = intent_ok = exact_ok = 0
    true_pos = pred_total = gold_total = 0

    for row, output in zip(gold_rows, outputs):
        gold_pairs = slot_pairs({"slots": row["slots"]})
        gold_total += sum(gold_pairs.values())

        if parse_strict(output) is not None:
            strict_ok += 1
        parsed = parse_lenient(output)
        if parsed is None:
            continue
        parsed_ok += 1

        pred_pairs = slot_pairs(parsed)
        pred_total += sum(pred_pairs.values())
        true_pos += sum((pred_pairs & gold_pairs).values())  # multiset intersection

        intent_correct = parsed.get("intent") == row["intent"]
        intent_ok += intent_correct
        exact_ok += intent_correct and pred_pairs == gold_pairs

    precision = true_pos / pred_total if pred_total else 0.0
    recall = true_pos / gold_total if gold_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": n,
        "strict_json_rate": strict_ok / n,
        "parsed_rate": parsed_ok / n,
        "intent_accuracy": intent_ok / n,
        "slot_precision": precision,
        "slot_recall": recall,
        "slot_f1": f1,
        "exact_match": exact_ok / n,
    }


def is_exact(row, output):
    parsed = parse_lenient(output)
    return bool(parsed) and parsed.get("intent") == row["intent"] and slot_pairs(parsed) == slot_pairs({"slots": row["slots"]})


def self_test():
    gold = [
        {"intent": "alarm_set", "slots": [{"type": "time", "value": "five am"}]},
        {"intent": "weather_query", "slots": []},
        {"intent": "alarm_query", "slots": [{"type": "date", "value": "today"}, {"type": "date", "value": "tomorrow"}]},
    ]
    outputs = [
        '{"intent": "alarm_set", "slots": [{"type": "time", "value": "Five AM"}]}',  # exact (case-insensitive value)
        '```json\n{"intent": "weather_query", "slots": []}\n```',                     # fenced: not strict, but parsed
        "I think the intent is alarm_query.",                                         # unparseable
    ]
    assert parse_strict(outputs[1]) is None and parse_lenient(outputs[1]) is not None
    metrics = score_predictions(gold, outputs)
    assert abs(metrics["strict_json_rate"] - 1 / 3) < 1e-9
    assert abs(metrics["parsed_rate"] - 2 / 3) < 1e-9
    assert abs(metrics["intent_accuracy"] - 2 / 3) < 1e-9
    assert abs(metrics["exact_match"] - 2 / 3) < 1e-9
    assert metrics["slot_precision"] == 1.0          # 1 predicted pair, 1 correct
    assert abs(metrics["slot_recall"] - 1 / 3) < 1e-9  # 3 gold pairs, 1 found
    print("Self-test passed:", json.dumps(metrics, indent=2))


# ---------- generation (needs torch + GPU) ----------

def load_jsonl(path):
    with open(path, encoding="utf-8") as in_file:
        return [json.loads(line) for line in in_file if line.strip()]


def generate_outputs(model, tokenizer, rows, labels, batch_size, max_new_tokens):
    import torch

    outputs = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                build_messages(row["utt"], labels["intents"], labels["slot_types"]),
                tokenize=False,
                add_generation_prompt=True,
            )
            for row in batch
        ]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = generated[:, inputs["input_ids"].shape[1]:]  # drop the prompt
        outputs.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
        print(f"  {min(start + batch_size, len(rows))}/{len(rows)}")
    return outputs


def main():
    parser = argparse.ArgumentParser(description="Evaluate base vs LoRA model on the Task 1 test set")
    parser.add_argument("--adapter", help="path to the saved LoRA adapter")
    parser.add_argument("--limit", type=int, default=None, help="only use the first N test rows")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--self-test", action="store_true", help="test the scoring code and exit")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return
    if not args.adapter:
        parser.error("--adapter is required unless --self-test is used")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = load_jsonl(PROCESSED_DIR / "test.jsonl")[: args.limit]
    with open(PROCESSED_DIR / "labels.json", encoding="utf-8") as labels_file:
        labels = json.load(labels_file)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = "left"  # decoder-only models must be left-padded for batched generation
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to("cuda")
    model.eval()

    print(f"Base model on {len(rows)} test rows")
    t0 = time.time()
    base_outputs = generate_outputs(model, tokenizer, rows, labels, args.batch_size, args.max_new_tokens)
    base_seconds = time.time() - t0

    print("Loading LoRA adapter")
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    t0 = time.time()
    tuned_outputs = generate_outputs(model, tokenizer, rows, labels, args.batch_size, args.max_new_tokens)
    tuned_seconds = time.time() - t0

    metrics = {
        "model": MODEL_NAME,
        "adapter": args.adapter,
        "decoding": {"greedy": True, "max_new_tokens": args.max_new_tokens},
        "base": score_predictions(rows, base_outputs),
        "fine_tuned": score_predictions(rows, tuned_outputs),
        "generation_seconds": {"base": round(base_seconds, 1), "fine_tuned": round(tuned_seconds, 1)},
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "metrics.json", "w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)
    with open(RESULTS_DIR / "predictions.csv", "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["id", "scenario", "utt", "gold", "base_output", "base_exact", "ft_output", "ft_exact"])
        for row, base_out, tuned_out in zip(rows, base_outputs, tuned_outputs):
            writer.writerow([
                row["id"], row["scenario"], row["utt"], row["target"],
                base_out, is_exact(row, base_out), tuned_out, is_exact(row, tuned_out),
            ])

    print(f"\n{'metric':<18}{'base':>10}{'fine-tuned':>12}")
    for key in ["strict_json_rate", "parsed_rate", "intent_accuracy", "slot_precision",
                "slot_recall", "slot_f1", "exact_match"]:
        print(f"{key:<18}{metrics['base'][key]:>10.3f}{metrics['fine_tuned'][key]:>12.3f}")
    print(f"\nWrote {RESULTS_DIR / 'metrics.json'} and predictions.csv")


if __name__ == "__main__":
    main()

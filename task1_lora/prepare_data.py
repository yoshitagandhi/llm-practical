"""Task 1 data preparation: MASSIVE (en-US) -> train / val / test JSONL files.

Input  : data/raw/amazon-massive-dataset-1.1/1.1/data/en-US.jsonl
         (the MASSIVE 1.1 download, extracted by hand; never modified)
Outputs: data/processed/train.jsonl, val.jsonl, test.jsonl, labels.json

Usage (from the repo root):
    python task1_lora/prepare_data.py --inspect   # print scenario counts only, write nothing
    python task1_lora/prepare_data.py             # build the processed files

Design decisions (approved):
- Use MASSIVE's own train / dev / test partitions, so no example can leak
  between training and testing.
- Keep only a few scenarios (SCENARIOS below) so the prompt stays short.
- Target JSON = {"intent": ..., "slots": [{"type": ..., "value": ...}, ...]}.
  A list (not a dict) handles the same slot type appearing twice.
- Test set = 200 examples sampled from the official test partition with a fixed seed.
"""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent
RAW_FILE = TASK_DIR / "data" / "raw" / "amazon-massive-dataset-1.1" / "1.1" / "data" / "en-US.jsonl"
PROCESSED_DIR = TASK_DIR / "data" / "processed"

SCENARIOS = ["alarm", "calendar", "weather", "transport"]  # confirm after --inspect
TEST_SAMPLE_SIZE = 200
MIN_TEST_SIZE = 50  # minimum required by the brief
SEED = 42

# MASSIVE marks slots inline as "[slot_type : value]".
SLOT_PATTERN = re.compile(r"\[(\w+) : ([^\]]+)\]")


def read_records(path):
    """Read one MASSIVE locale file (one JSON object per line)."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download and extract MASSIVE 1.1 first (see README)."
        )
    records = []
    with open(path, encoding="utf-8") as in_file:
        for line in in_file:
            if line.strip():
                records.append(json.loads(line))
    return records


def parse_slots(annot_utt):
    """Return slots as a list of {"type", "value"} dicts, in order of appearance."""
    return [{"type": slot_type, "value": value.strip()} for slot_type, value in SLOT_PATTERN.findall(annot_utt)]


def strip_annotation(annot_utt):
    """Remove the [type : value] markup, keeping only the value text.

    Used as a sanity check: the result should equal the plain utterance.
    """
    return SLOT_PATTERN.sub(lambda match: match.group(2).strip(), annot_utt)


def build_target(slots, intent):
    """The exact JSON string the model should learn to produce."""
    return json.dumps({"intent": intent, "slots": slots}, ensure_ascii=False)


def to_row(record):
    slots = parse_slots(record["annot_utt"])
    return {
        "id": record["id"],
        "scenario": record["scenario"],
        "intent": record["intent"],
        "utt": record["utt"],
        "slots": slots,
        "target": build_target(slots, record["intent"]),
    }


def split_by_partition(records, scenarios):
    """Group the chosen scenarios' records by MASSIVE's official partition."""
    splits = {"train": [], "dev": [], "test": []}
    for record in records:
        if record["scenario"] in scenarios:
            splits[record["partition"]].append(record)
    return splits


def sample_test(rows, sample_size, seed):
    """Fixed random sample so the test set is identical on every run."""
    if len(rows) <= sample_size:
        print(f"WARNING: only {len(rows)} test rows available, using all of them")
        return list(rows)
    rng = random.Random(seed)
    return rng.sample(rows, sample_size)


def write_jsonl(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as out_file:
        for row in rows:
            out_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def print_inspection(records):
    """Counts per scenario and partition, to choose SCENARIOS with evidence."""
    counts = Counter((r["scenario"], r["partition"]) for r in records)
    scenarios = sorted({r["scenario"] for r in records})
    print(f"{'scenario':<16}{'train':>7}{'dev':>7}{'test':>7}{'intents':>9}")
    for scenario in scenarios:
        n_intents = len({r["intent"] for r in records if r["scenario"] == scenario})
        print(
            f"{scenario:<16}{counts[(scenario, 'train')]:>7}"
            f"{counts[(scenario, 'dev')]:>7}{counts[(scenario, 'test')]:>7}{n_intents:>9}"
        )
    print(f"\nTotal records: {len(records)}")


def print_summary(train_rows, val_rows, test_rows, intents, slot_types):
    print(f"Scenarios: {SCENARIOS}")
    print(f"Rows -> train: {len(train_rows)}, val: {len(val_rows)}, test: {len(test_rows)}")
    print(f"Intents ({len(intents)}): {intents}")
    print(f"Slot types ({len(slot_types)}): {slot_types}")

    all_rows = train_rows + val_rows + test_rows
    no_slots = sum(1 for row in all_rows if not row["slots"])
    repeated = sum(
        1 for row in all_rows
        if len(row["slots"]) != len({slot["type"] for slot in row["slots"]})
    )
    print(f"Rows with no slots: {no_slots}")
    print(f"Rows where a slot type repeats: {repeated}")

    unseen = [
        row["id"] for row in val_rows + test_rows
        if row["intent"] not in intents or any(s["type"] not in slot_types for s in row["slots"])
    ]
    print(f"Val/test rows using a label not seen in train: {len(unseen)}")

    print("\nExample target:")
    print(f"  utt   : {train_rows[0]['utt']}")
    print(f"  target: {train_rows[0]['target']}")


def run_checks(records, train_rows, val_rows, test_rows):
    # The parser must reproduce the documented example.
    assert parse_slots("wake me up at [time : five am] [date : this week]") == [
        {"type": "time", "value": "five am"},
        {"type": "date", "value": "this week"},
    ]
    # Every target must be valid JSON.
    for row in train_rows + val_rows + test_rows:
        json.loads(row["target"])
    # No example may appear in two splits.
    train_ids = {row["id"] for row in train_rows}
    val_ids = {row["id"] for row in val_rows}
    test_ids = {row["id"] for row in test_rows}
    assert not (train_ids & val_ids), "train/val overlap"
    assert not (train_ids & test_ids), "train/test overlap"
    assert not (val_ids & test_ids), "val/test overlap"
    assert len(test_rows) >= MIN_TEST_SIZE, f"test set has fewer than {MIN_TEST_SIZE} rows"

    # Report (not drop) records where removing the markup does not give back the utterance.
    mismatches = [r["id"] for r in records if strip_annotation(r["annot_utt"]) != r["utt"]]
    print(f"Records where annot_utt does not match utt after stripping markup: {len(mismatches)}")
    if mismatches:
        print(f"  first few ids: {mismatches[:5]}")
    print("All checks passed.")


def main():
    parser = argparse.ArgumentParser(description="Prepare MASSIVE data for Task 1")
    parser.add_argument("--inspect", action="store_true", help="print counts only, write nothing")
    args = parser.parse_args()

    records = read_records(RAW_FILE)

    if args.inspect:
        print_inspection(records)
        return

    splits = split_by_partition(records, SCENARIOS)
    train_rows = [to_row(r) for r in splits["train"]]
    val_rows = [to_row(r) for r in splits["dev"]]
    test_rows = sample_test([to_row(r) for r in splits["test"]], TEST_SAMPLE_SIZE, SEED)

    # The label lists shown in the prompt come from the training data only.
    intents = sorted({row["intent"] for row in train_rows})
    slot_types = sorted({slot["type"] for row in train_rows for slot in row["slots"]})

    chosen_records = splits["train"] + splits["dev"] + splits["test"]
    run_checks(chosen_records, train_rows, val_rows, test_rows)

    write_jsonl(train_rows, PROCESSED_DIR / "train.jsonl")
    write_jsonl(val_rows, PROCESSED_DIR / "val.jsonl")
    write_jsonl(test_rows, PROCESSED_DIR / "test.jsonl")
    labels = {"scenarios": SCENARIOS, "intents": intents, "slot_types": slot_types}
    with open(PROCESSED_DIR / "labels.json", "w", encoding="utf-8") as labels_file:
        json.dump(labels, labels_file, indent=2)

    print_summary(train_rows, val_rows, test_rows, intents, slot_types)
    print(f"\nWrote files to {PROCESSED_DIR}")


if __name__ == "__main__":
    main()

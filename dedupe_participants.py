"""
Removes duplicate trials for any participant who ended up with more than
4 entries in queries.jsonl (e.g. because they restarted/reopened the study
and redid it, producing trial_positions like [1,1,2,2,3,3,4,4]).

For each participant, keeps only the FIRST (earliest timestamp) entry per
trial_position and discards any later duplicates -- in both queries.jsonl
and user_responses.jsonl, matched by participant_id + trial_position.

Backs up both files before making any changes.

Usage:
    python dedupe_participants.py
"""
import json
import shutil
from collections import defaultdict


def load_jsonl(path):
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def write_jsonl(path, entries):
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def dedupe(entries):
    """Keep only the first (by timestamp) entry per (participant_id, trial_position)."""
    entries_sorted = sorted(entries, key=lambda e: e.get("timestamp", ""))
    seen = set()
    kept = []
    dropped = []
    for e in entries_sorted:
        key = (e.get("participant_id"), e.get("trial_position"))
        if key in seen:
            dropped.append(e)
            continue
        seen.add(key)
        kept.append(e)
    return kept, dropped


def main():
    queries_path = "logs/queries.jsonl"
    responses_path = "logs/user_responses.jsonl"

    shutil.copy(queries_path, queries_path + ".bak")
    shutil.copy(responses_path, responses_path + ".bak")
    print("Backed up both files (.bak) before making any changes.\n")

    queries = load_jsonl(queries_path)
    responses = load_jsonl(responses_path)

    kept_queries, dropped_queries = dedupe(queries)
    kept_responses, dropped_responses = dedupe(responses)

    print(f"queries.jsonl: {len(queries)} -> {len(kept_queries)} "
          f"(dropped {len(dropped_queries)} duplicates)")
    for d in dropped_queries:
        print(f"   dropped: participant={d.get('participant_id')} "
              f"trial_position={d.get('trial_position')} "
              f"condition={d.get('condition')} timestamp={d.get('timestamp')}")

    print(f"\nuser_responses.jsonl: {len(responses)} -> {len(kept_responses)} "
          f"(dropped {len(dropped_responses)} duplicates)")
    for d in dropped_responses:
        print(f"   dropped: participant={d.get('participant_id')} "
              f"trial_position={d.get('trial_position')} "
              f"timestamp={d.get('timestamp')}")

    write_jsonl(queries_path, kept_queries)
    write_jsonl(responses_path, kept_responses)
    print("\nBoth files rewritten with duplicates removed. ✅")
    print("Run verify_coverage.py again to confirm everything is clean now.")


if __name__ == "__main__":
    main()
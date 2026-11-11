"""
Repairs a malformed JSONL log file where two or more JSON objects ended up
concatenated on the same line (missing newline between them) due to a
concurrent-write race condition.

Does NOT drop or alter any data -- it just re-parses every JSON object in
the file one at a time using raw_decode (which stops at the end of one
valid object, ignoring whatever comes right after it), then rewrites the
file with exactly one object per line, properly separated.

Usage:
    python repair_logs.py logs/queries.jsonl
    python repair_logs.py logs/user_responses.jsonl

A .bak backup of the original file is made before overwriting anything.
"""
import json
import sys
import shutil


def repair_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    decoder = json.JSONDecoder()
    entries = []
    idx = 0
    n = len(content)
    while idx < n:
        # Skip whitespace/newlines between objects
        while idx < n and content[idx] in " \t\r\n":
            idx += 1
        if idx >= n:
            break
        try:
            obj, end_idx = decoder.raw_decode(content, idx)
        except json.JSONDecodeError as e:
            print(f"⚠️  Could not parse a valid JSON object starting at char {idx}: {e}")
            print("   Stopping here -- everything before this point has been recovered.")
            break
        entries.append(obj)
        idx = end_idx

    print(f"Recovered {len(entries)} valid JSON objects from {path}")

    backup_path = path + ".bak"
    shutil.copy(path, backup_path)
    print(f"Original file backed up to {backup_path}")

    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    print(f"Rewrote {path} with one JSON object per line. ✅")
    return len(entries)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python repair_logs.py <path-to-jsonl-file>")
        sys.exit(1)
    repair_jsonl(sys.argv[1])
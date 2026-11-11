"""
Run this after data collection to verify the study actually ran the way
it was supposed to: every participant should have exactly 4 trials,
covering all 4 distinct conditions AND all 4 distinct questions, with
trial_position 1-4 each used once.
"""
import json
from collections import defaultdict

trials = defaultdict(list)
with open("logs/queries.jsonl") as f:
    for line in f:
        d = json.loads(line)
        trials[d["participant_id"]].append(
            (d.get("trial_position"), d["condition"], d.get("question_slot_id"))
        )

problems = []
for pid, t in trials.items():
    conditions = sorted(c for _, c, _ in t)
    questions = sorted(q for _, _, q in t)
    positions = sorted(p for p, _, _ in t)
    if len(t) != 4:
        problems.append(f"{pid}: has {len(t)} trials, expected 4")
    if len(set(conditions)) != 4:
        problems.append(f"{pid}: conditions = {conditions} (not all 4 distinct)")
    if len(set(questions)) != 4:
        problems.append(f"{pid}: questions = {questions} (not all 4 distinct)")
    if positions != [1, 2, 3, 4]:
        problems.append(f"{pid}: trial_positions = {positions} (expected [1,2,3,4])")

print(f"Checked {len(trials)} participants.")
if problems:
    print(f"\n{len(problems)} PROBLEM(S) FOUND:")
    for p in problems:
        print(" -", p)
else:
    print("All participants completed all 4 distinct conditions AND all 4 distinct questions correctly. ✅")

# Bonus: how evenly did each question land in each condition across the sample?
# (Won't be perfectly even with pure randomization -- just a sanity check that
# nothing is badly skewed, e.g. one question always landing in Condition C.)
pairing_counts = defaultdict(int)
for t in trials.values():
    for _, cond, q in t:
        pairing_counts[(q, cond)] += 1

print("\nQuestion x Condition pairing counts (rough balance check):")
questions_seen = sorted({q for _, _, q in [x for v in trials.values() for x in v]})
conditions_seen = sorted({c for _, c, _ in [x for v in trials.values() for x in v]})
header = "            " + "".join(f"{c[:12]:>14}" for c in conditions_seen)
print(header)
for q in questions_seen:
    row = f"{q:>12}" + "".join(f"{pairing_counts[(q,c)]:>14}" for c in conditions_seen)
    print(row)
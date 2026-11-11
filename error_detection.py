"""
Computes the one number the paper draft is still missing: overall and
per-condition Error Detection Rate (% of error-injected trials where the
participant explicitly reported noticing an error), matching the
definition already used in logger.py's _compute_summary().
"""
import json

queries = {}
with open("logs/queries.jsonl") as f:
    for line in f:
        d = json.loads(line)
        queries[d["question_id"]] = d

responses = []
with open("logs/user_responses.jsonl") as f:
    for line in f:
        responses.append(json.loads(line))

error_responses = [r for r in responses if queries.get(r["question_id"], {}).get("injected_error")]

detected = sum(1 for r in error_responses if r.get("noticed_error") is True)
print(f"Overall Error Detection Rate: {detected}/{len(error_responses)} = {detected/len(error_responses)*100:.1f}%")

for cond in ["C_trace_and_error", "D_error_only"]:
    cond_err = [r for r in error_responses if queries[r["question_id"]]["condition"] == cond]
    cond_detected = sum(1 for r in cond_err if r.get("noticed_error") is True)
    print(f"{cond}: {cond_detected}/{len(cond_err)} = {cond_detected/len(cond_err)*100:.1f}%")
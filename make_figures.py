"""
Regenerates the three result figures used in the paper, directly from
logs/queries.jsonl and logs/user_responses.jsonl, using the same
definitions as analyze_gee.py / error.py / logger.py's _compute_summary().

Run from the project root (same level as the logs/ folder):
    pip install matplotlib --break-system-packages
    python make_figures.py

Produces:
    fig2_bias_rate.png     -> Automation Bias Rate by Condition   (Fig. \ref{bias_rate})
    fig3_confidence.png    -> Average Confidence by Condition     (Fig. \ref{confidence})
    fig4_response_dist.png -> Error Detection Rate by Condition   (Fig. \ref{error})
"""

import json
import matplotlib.pyplot as plt

CONDITIONS = ["A_baseline", "B_trace_only", "C_trace_and_error", "D_error_only"]
CONDITION_LABELS = {
    "A_baseline": "A - Baseline",
    "B_trace_only": "B - Trace Only",
    "C_trace_and_error": "C - Trace + Error",
    "D_error_only": "D - Error Only",
}
ERROR_CONDITIONS = ["C_trace_and_error", "D_error_only"]


def load_data(queries_path="logs/queries.jsonl", responses_path="logs/user_responses.jsonl"):
    queries = {}
    with open(queries_path) as f:
        for line in f:
            d = json.loads(line)
            queries[d["question_id"]] = d

    responses = []
    with open(responses_path) as f:
        for line in f:
            responses.append(json.loads(line))

    # Attach condition / injected_error to each response via its question_id.
    rows = []
    for r in responses:
        q = queries.get(r["question_id"])
        if q is None:
            continue
        rows.append({
            "condition": q["condition"],
            "injected_error": q["injected_error"],
            "accepted": bool(r.get("user_accepted_answer")),
            "noticed_error": bool(r.get("noticed_error")),
            "confidence": r.get("confidence_1_to_5"),
        })
    return rows


def fig_bias_rate(rows):
    """Automation bias rate by condition, error-injected trials only."""
    rates = {}
    for cond in ERROR_CONDITIONS:
        cond_err = [r for r in rows if r["condition"] == cond and r["injected_error"]]
        biased = [r for r in cond_err if r["accepted"] and not r["noticed_error"]]
        rates[cond] = 100 * len(biased) / len(cond_err) if cond_err else 0

    labels = [CONDITION_LABELS[c] for c in ERROR_CONDITIONS]
    values = [rates[c] for c in ERROR_CONDITIONS]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, values, color=["#4C72B0", "#DD8452"])
    ax.set_ylabel("Automation Bias Rate (%)")
    ax.set_ylim(0, 100)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.1f}%", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig("fig2_bias_rate.png", dpi=300)
    plt.close()
    print("Saved fig2_bias_rate.png:", rates)


def fig_confidence(rows):
    """Mean self-reported confidence by condition, all trials."""
    means = {}
    for cond in CONDITIONS:
        vals = [r["confidence"] for r in rows if r["condition"] == cond and r["confidence"]]
        means[cond] = sum(vals) / len(vals) if vals else 0

    labels = [CONDITION_LABELS[c] for c in CONDITIONS]
    values = [means[c] for c in CONDITIONS]
    colors = ["#4C72B0", "#55A868", "#DD8452", "#C44E52"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel("Mean Confidence (1-5)")
    ax.set_ylim(0, 5)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.08, f"{v:.2f}", ha="center", fontweight="bold")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig("fig3_confidence.png", dpi=300)
    plt.close()
    print("Saved fig3_confidence.png:", means)


def fig_error_detection(rows):
    """Error detection rate by condition, error-injected trials only."""
    rates = {}
    for cond in ERROR_CONDITIONS:
        cond_err = [r for r in rows if r["condition"] == cond and r["injected_error"]]
        detected = [r for r in cond_err if r["noticed_error"]]
        rates[cond] = 100 * len(detected) / len(cond_err) if cond_err else 0

    labels = [CONDITION_LABELS[c] for c in ERROR_CONDITIONS]
    values = [rates[c] for c in ERROR_CONDITIONS]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, values, color=["#4C72B0", "#DD8452"])
    ax.set_ylabel("Error Detection Rate (%)")
    ax.set_ylim(0, 100)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.1f}%", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig("fig4_response_dist.png", dpi=300)
    plt.close()
    print("Saved fig4_response_dist.png:", rates)


if __name__ == "__main__":
    rows = load_data()
    fig_bias_rate(rows)
    fig_confidence(rows)
    fig_error_detection(rows)
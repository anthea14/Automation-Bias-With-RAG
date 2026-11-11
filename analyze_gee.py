"""
Analysis script for the 100-participant study.

Run this AFTER all 100 participants have completed sessions using
app.py / logger.py in this package. Expects logs/queries.jsonl and
logs/user_responses.jsonl to contain the full set of trials collected
under the randomized-order, 4-condition, within-subjects protocol.
Run verify_coverage.py first and confirm zero problems before running
this script.

WHY GEE (Generalized Estimating Equations) INSTEAD OF CHI-SQUARE/FISHER:
Every participant contributes one trial to EACH of the 4 conditions, so
trials from the same participant are correlated, not independent. GEE with
a binomial family and participant_id as the clustering variable models
this correctly, using all 4 conditions and all 100 participants in a single
model, rather than running separate pairwise tests (e.g. C vs D only).

WHAT THE MODEL ESTIMATES:
  automation_bias ~ condition + trial_position + question_slot_id
  clustered by participant_id

This directly answers three questions in one model:
  1. Does condition (A/B/C/D) predict automation bias, controlling for
     where in the session the trial fell and which question was asked?
  2. Does trial_position itself predict bias (an order/fatigue/practice
     effect), controlling for condition and question?
  3. Does question_slot_id predict bias (some questions may simply be
     easier or harder to catch an injected error in)? Controlling for
     this matters because the randomization of question-to-condition
     pairing is not perfectly balanced across 100 participants -- e.g.
     one question may land disproportionately often in one condition by
     chance -- and without this term, that imbalance could be mistaken
     for a condition effect.

Requires: pandas, statsmodels
    pip install pandas statsmodels --break-system-packages
"""

import json
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_trials(queries_path="logs/queries.jsonl", responses_path="logs/user_responses.jsonl"):
    queries = {}
    with open(queries_path) as f:
        for line in f:
            d = json.loads(line)
            queries[d["question_id"]] = d

    rows = []
    with open(responses_path) as f:
        for line in f:
            r = json.loads(line)
            q = queries.get(r["question_id"])
            if q is None:
                continue
            # Automation bias is only defined on error-injected trials.
            # (Condition A and B never have injected_error=True, so they
            # naturally drop out of the bias-rate model below, exactly as
            # the paper's own ABR definition intends -- but they DO still
            # matter for error-detection-rate / confidence models if you
            # want to extend this script.)
            rows.append({
                "participant_id": q["participant_id"],
                "condition": q["condition"],
                "trial_position": q["trial_position"],
                "question_slot_id": q.get("question_slot_id"),
                "injected_error": q["injected_error"],
                "accepted": bool(r.get("user_accepted_answer")),
                "noticed_error": bool(r.get("noticed_error")),
                "confidence": r.get("confidence_1_to_5"),
            })
    df = pd.DataFrame(rows)
    df["biased"] = (df["accepted"] & ~df["noticed_error"]).astype(int)
    return df


def summarize(df):
    print("=== Sample sizes ===")
    print(f"Total trials: {len(df)}")
    print(f"Unique participants: {df['participant_id'].nunique()}")
    print(df.groupby("condition").size(), "\n")

    print("=== Automation bias rate by condition (error trials only) ===")
    err = df[df["injected_error"] == True]
    print(err.groupby("condition")["biased"].mean().mul(100).round(1), "\n")

    print("=== Automation bias rate by question (error trials only) ===")
    print(err.groupby("question_slot_id")["biased"].mean().mul(100).round(1), "\n")

    print("=== Question x Condition trial counts (error trials only) ===")
    print(err.groupby(["question_slot_id", "condition"]).size().unstack(fill_value=0), "\n")

    print("=== Mean confidence by condition ===")
    print(df.groupby("condition")["confidence"].mean().round(2), "\n")


def run_gee(df):
    """
    Primary inferential model: GEE logistic regression on error-injected
    trials only (bias is only defined there), clustered by participant,
    with condition, trial_position, AND question_slot_id as predictors.

    question_slot_id is included because the randomization of question
    identity to condition is not perfectly balanced across the sample
    (verified via verify_coverage.py's pairing-count table) -- without
    controlling for it, an uneven pairing (e.g. one question landing
    disproportionately often in one condition) could be misattributed to
    a condition effect rather than a question-difficulty effect.
    """
    err = df[df["injected_error"] == True].copy()
    # Only C and D carry injected errors in this design (A/B are clean by
    # definition), so this model compares C vs D while controlling for
    # trial_position and question_slot_id, using all 100 participants'
    # paired C/D trials at once.
    err["condition"] = err["condition"].astype("category")
    err["question_slot_id"] = err["question_slot_id"].astype("category")
    err["trial_position"] = err["trial_position"].astype(int)

    print("=== GEE logistic regression: biased ~ condition + trial_position + question_slot_id ===\n")

    for struct_name, struct in [
        ("Independence", sm.cov_struct.Independence()),
        ("Exchangeable", sm.cov_struct.Exchangeable()),
    ]:
        model = smf.gee(
            "biased ~ C(condition) + trial_position + C(question_slot_id)",
            groups="participant_id",
            data=err,
            family=sm.families.Binomial(),
            cov_struct=struct,
        )
        result = model.fit()
        print(f"--- Working correlation structure: {struct_name} ---")
        print(result.summary())
        print()

    return result


if __name__ == "__main__":
    df = load_trials()
    summarize(df)
    run_gee(df)
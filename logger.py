"""
Study Logger
Records all interactions for research analysis:
- What the agent retrieved
- What answer it gave
- What the user decided (accepted/rejected)
- Confidence, time, comments
"""

import os
import json
import threading
from datetime import datetime


class StudyLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.query_log_path = os.path.join(log_dir, "queries.jsonl")
        self.response_log_path = os.path.join(log_dir, "user_responses.jsonl")
        # Prevents two concurrent requests (e.g. two participants using the
        # system at the same time) from interleaving their writes to the
        # same log file mid-write, which previously produced malformed
        # lines with two JSON objects concatenated together.
        self._write_lock = threading.Lock()

    def log_query(self, session_id, participant_id, question_id, question, answer,
              condition, injected_error, trace, time_taken, trial_position=None,
              question_slot_id=None):
        if not question_id:
            raise ValueError("log_query: question_id must not be None or empty")
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "participant_id": participant_id,
            "session_id": session_id,
            "question_id": question_id,
            "question": question,
            "answer": answer,
            "condition": condition,
            "injected_error": injected_error,
            "trace_steps": len(trace) if trace else 0,
            "time_taken_sec": time_taken,
            "trial_position": trial_position,  # 1-4: which slot in this participant's sequence
            "question_slot_id": question_slot_id,  # e.g. "Q_BLEU": which of the 4 fixed questions
        }
        self._append(self.query_log_path, entry)

    def log_user_response(self, session_id, participant_id, question_id, 
                      user_accepted, confidence, noticed_error, 
                      time_to_decide, comments, condition, trial_position=None,
                      question_slot_id=None):
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "participant_id": participant_id,
            "session_id": session_id,
            "question_id": question_id,
            "condition": condition,
            "user_accepted_answer": user_accepted,
            "confidence_1_to_5": confidence,
            "noticed_error": noticed_error,
            "time_to_decide_sec": time_to_decide,
            "comments": comments,
            "trial_position": trial_position,
            "question_slot_id": question_slot_id,
        }
        self._append(self.response_log_path, entry)

    def get_all_logs(self):
        """Return all logs as a dict for export."""
        return {
            "queries": self._read_all(self.query_log_path),
            "user_responses": self._read_all(self.response_log_path),
            "summary": self._compute_summary()
        }

    def _compute_summary(self):
        """Compute automation bias metrics correctly.

        Automation bias rate = % of ERROR-injected trials where the user
        accepted the (wrong) answer without noticing.  Counting ALL accepted
        answers (correct + wrong) in the old code inflated this metric and
        made it scientifically meaningless.
        """
        responses = self._read_all(self.response_log_path)
        queries   = self._read_all(self.query_log_path)

        if not responses:
            return {"message": "No data yet"}

        # Build a lookup: question_id -> injected_error flag
        error_flag_by_qid = {
            q["question_id"]: q.get("injected_error", False)
            for q in queries
            if "question_id" in q
        }

        # Split responses into error trials vs. clean trials
        error_responses = [
            r for r in responses
            if error_flag_by_qid.get(r.get("question_id"), False)
        ]
        clean_responses = [
            r for r in responses
            if not error_flag_by_qid.get(r.get("question_id"), False)
        ]

        # ── Core metrics ────────────────────────────────────────────────────

        # Automation bias rate: among error trials, how often did the user
        # accept the wrong answer (and NOT notice the error)?
        biased = sum(
            1 for r in error_responses
            if r.get("user_accepted_answer") is True
            and r.get("noticed_error") is not True
        )
        automation_bias_rate = biased / max(1, len(error_responses))

        # Error detection rate: among error trials, how often did the user
        # notice the injected error?
        detected = sum(1 for r in error_responses if r.get("noticed_error") is True)
        error_detection_rate = detected / max(1, len(error_responses))

        # Average confidence — reported separately for error vs. clean trials
        def avg_conf(subset):
            vals = [r["confidence_1_to_5"] for r in subset if r.get("confidence_1_to_5")]
            return round(sum(vals) / len(vals), 2) if vals else None

        # ── Per-condition breakdown ──────────────────────────────────────────
        conditions = ["A_baseline", "B_trace_only", "C_trace_and_error", "D_error_only"]
        per_condition = {}
        for cond in conditions:
            cond_responses = [r for r in responses if r.get("condition") == cond]
            cond_errors    = [r for r in cond_responses
                              if error_flag_by_qid.get(r.get("question_id"), False)]
            cond_biased    = sum(
                1 for r in cond_errors
                if r.get("user_accepted_answer") is True
                and r.get("noticed_error") is not True
            )
            per_condition[cond] = {
                "n_responses": len(cond_responses),
                "n_error_trials": len(cond_errors),
                "automation_bias_rate": (
                    f"{round(cond_biased / len(cond_errors) * 100, 1)}%"
                    if cond_errors else "n/a"
                ),
                "avg_confidence": avg_conf(cond_responses),
            }

        return {
            "total_responses": len(responses),
            "total_error_trials": len(error_responses),
            "total_clean_trials": len(clean_responses),
            # PRIMARY METRIC for the paper
            "automation_bias_rate": f"{round(automation_bias_rate * 100, 1)}%",
            "automation_bias_note": (
                "% of error-injected trials where user accepted wrong answer "
                "without noticing — this is the correct measure of automation bias"
            ),
            "error_detection_rate": f"{round(error_detection_rate * 100, 1)}%",
            # NEW: flat top-level key covering ALL responses, so a dashboard
            # reading `summary.avg_confidence` gets a real number instead of
            # undefined. The two more specific keys below are unchanged and
            # remain the ones to use in the paper itself.
            "avg_confidence": avg_conf(responses),
            "avg_confidence_on_error_trials": avg_conf(error_responses),
            "avg_confidence_on_clean_trials": avg_conf(clean_responses),
            "per_condition_breakdown": per_condition,
        }

    def _append(self, path, entry):
        # Serializes all writes to a given log file. Without this, two
        # participants submitting a request at nearly the same moment could
        # have their file.write() calls interleave at the OS level,
        # producing a malformed line (two JSON objects concatenated, or a
        # partially-overwritten line) -- exactly the corruption seen in
        # queries.jsonl and user_responses.jsonl previously.
        with self._write_lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
                f.flush()
                os.fsync(f.fileno())

    def _read_all(self, path):
        if not os.path.exists(path):
            return []
        entries = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return entries
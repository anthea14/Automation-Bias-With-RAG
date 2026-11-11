"""
Agentic RAG System - Main Application (FIXED)
For Research: Automation Bias & Over-Reliance Study

FIXES vs original app.py:
1. /assign_condition now actually generates a per-participant randomized
   ORDER of all four conditions (Latin-square balanced across participants),
   rather than relying on manual toggle buttons in the UI.
2. The order is persisted per participant_id, and /ask consumes it one
   step at a time via a `trial_position` index sent from the frontend --
   so the frontend can no longer silently drift into a fixed sequence.
3. Every query log now includes `trial_position` (1-4) so order effects
   can be modeled/controlled for in analysis, instead of being invisible.
4. Condition B is a first-class member of the rotation, so it is no
   longer structurally impossible to collect.
"""

import os
import json
import random
import time
import uuid
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from agent import AgenticRAG
from logger import StudyLogger

app = Flask(__name__, static_folder="static")

rag_agent = AgenticRAG()
study_logger = StudyLogger()

ALL_CONDITIONS = ["A_baseline", "B_trace_only", "C_trace_and_error", "D_error_only"]

# The 4 fixed study questions for the "Attention Is All You Need" document.
# Kept server-side (not typed by hand each time) so that:
#   1. the exact same question wording is used for every participant, and
#   2. the mapping of "which question -> which trial position" can be
#      randomized per participant, independent of condition order.
QUESTIONS = [
    {
        "id": "Q_ARCH",
        "text": "What are the dimensionality of the model (d_model), the number "
                "of attention heads, and the number of encoder/decoder layers "
                "used in the base Transformer configuration?",
    },
    {
        "id": "Q_ATTN",
        "text": "How does scaled dot-product attention work, and why do the "
                "authors scale by the square root of d_k before the softmax?",
    },
    {
        "id": "Q_BLEU",
        "text": "What BLEU scores did the Transformer (big) model achieve on the "
                "WMT 2014 English-to-German and English-to-French translation "
                "tasks, and how does that compare to the previous state of the art?",
    },
    {
        "id": "Q_POS",
        "text": "Why did the authors choose sinusoidal positional encodings "
                "instead of learned positional embeddings?",
    },
]

_ASSIGNMENTS_PATH = os.path.join("logs", "participant_assignments.json")
_COUNTER_PATH = os.path.join("logs", "condition_counter.json")


def _load_assignments():
    if os.path.exists(_ASSIGNMENTS_PATH):
        with open(_ASSIGNMENTS_PATH) as f:
            return json.load(f)
    return {}


def _save_assignments(data):
    os.makedirs("logs", exist_ok=True)
    with open(_ASSIGNMENTS_PATH, "w") as f:
        json.dump(data, f, indent=2)


def _next_participant_number():
    os.makedirs("logs", exist_ok=True)
    if os.path.exists(_COUNTER_PATH):
        with open(_COUNTER_PATH) as f:
            counter = json.load(f).get("counter", 0)
    else:
        counter = 0
    with open(_COUNTER_PATH, "w") as f:
        json.dump({"counter": counter + 1}, f)
    return counter + 1


def _random_order():
    """
    Fully random, independent order of the 4 conditions for this participant.
    Each of the 24 possible permutations is equally likely; no relationship
    is enforced between one participant's order and another's.
    """
    order = ALL_CONDITIONS[:]
    random.shuffle(order)
    return order


def _random_question_order():
    """
    Fully random, independent order of the 4 fixed questions for this
    participant -- shuffled SEPARATELY from the condition order, so
    "which question" and "which condition" land on a given trial_position
    independently of each other. This is what stops one question (e.g. the
    BLEU one) from always ending up paired with the same condition, or
    always ending up last, across the whole sample.
    """
    order = QUESTIONS[:]
    random.shuffle(order)
    return [q["id"] for q in order]


@app.route("/assign_condition", methods=["POST"])
def assign_condition():
    """
    Call this exactly ONCE at the start of a participant's session.
    Generates and stores their full 4-condition order, returns participant_id
    and that order. The frontend then requests conditions by trial_position
    (1-4) rather than deciding show_trace/inject_error itself.

    Accepts an optional participant_id in the POST body (e.g. "P047") so
    experimenters can keep their own numbering; if omitted, one is
    auto-generated. If the participant_id was already assigned (e.g. the
    page was refreshed mid-session), the SAME stored order is returned
    rather than generating a new one, so a refresh can't accidentally
    re-randomize a participant partway through.
    """
    data = request.json or {}
    requested_id = (data.get("participant_id") or "").strip()

    assignments = _load_assignments()

    if requested_id and requested_id in assignments:
        existing = assignments[requested_id]
        return jsonify({
            "participant_id": requested_id,
            "condition_order": existing["order"],
            "question_order": existing["question_order"],
            "questions_by_id": {q["id"]: q["text"] for q in QUESTIONS},
        })

    if requested_id:
        participant_id = requested_id
    else:
        participant_number = _next_participant_number()
        participant_id = f"P{participant_number:03d}"

    order = _random_order()
    question_order = _random_question_order()
    assignments[participant_id] = {
        "order": order,
        "question_order": question_order,
        "created": datetime.utcnow().isoformat(),
    }
    _save_assignments(assignments)

    return jsonify({
        "participant_id": participant_id,
        "condition_order": order,
        "question_order": question_order,
        "questions_by_id": {q["id"]: q["text"] for q in QUESTIONS},
    })


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/ask", methods=["POST"])
def ask():
    """
    Main endpoint: given a trial_position, looks up BOTH the condition and
    the question text from this participant's stored assignment, and
    returns the agent's answer + trace.

    The question is no longer taken from free-typed client text -- it's
    looked up server-side by trial_position, so every participant gets the
    exact same 4 question wordings, just in a randomized order that's
    independent of their condition order.
    """
    data = request.json
    session_id = data.get("session_id", str(uuid.uuid4()))
    participant_id = data.get("participant_id", "unknown")
    trial_position = data.get("trial_position")  # 1-4, REQUIRED

    if trial_position is None:
        return jsonify({"error": "trial_position (1-4) is required"}), 400

    assignments = _load_assignments()
    assignment = assignments.get(participant_id)
    if not assignment:
        return jsonify({"error": f"No assignment found for {participant_id}. Call /assign_condition first."}), 400

    participant_order = assignment["order"]
    question_order = assignment["question_order"]

    idx = int(trial_position) - 1
    if idx < 0 or idx >= len(participant_order):
        return jsonify({"error": "trial_position out of range"}), 400

    condition = participant_order[idx]
    question_slot_id = question_order[idx]
    question_lookup = {q["id"]: q["text"] for q in QUESTIONS}
    question = question_lookup[question_slot_id]

    show_trace = condition in ("B_trace_only", "C_trace_and_error")
    inject_error = condition in ("C_trace_and_error", "D_error_only")

    start_time = time.time()
    result = rag_agent.run(question, inject_error=inject_error)
    elapsed = round(time.time() - start_time, 2)

    response = {
        "answer": result["answer"],
        "session_id": session_id,
        "time_taken": elapsed,
        "question_id": result["question_id"],
        "question_slot_id": question_slot_id,
        "question_text": question,
        "condition": condition,
    }

    if show_trace:
        response["trace"] = result["trace"]
        response["sources"] = result["sources"]
        response["agent_reasoning"] = result["agent_reasoning"]
    else:
        response["trace"] = None
        response["sources"] = None
        response["agent_reasoning"] = None

    study_logger.log_query(
        session_id=session_id,
        participant_id=participant_id,
        question_id=result["question_id"],
        question=question,
        answer=result["answer"],
        condition=condition,
        injected_error=inject_error,
        trace=result["trace"],
        time_taken=elapsed,
        trial_position=trial_position,
        question_slot_id=question_slot_id,  # NEW: which of the 4 fixed questions this was
    )

    return jsonify(response)


@app.route("/submit_response", methods=["POST"])
def submit_response():
    """Records the user's decision and confidence after seeing the agent's answer."""
    data = request.json
    study_logger.log_user_response(
        session_id=data.get("session_id"),
        participant_id=data.get("participant_id", "unknown"),
        question_id=data.get("question_id"),
        user_accepted=data.get("user_accepted"),
        confidence=data.get("confidence"),
        noticed_error=data.get("noticed_error"),
        time_to_decide=data.get("time_to_decide"),
        comments=data.get("comments", ""),
        condition=data.get("condition", "unknown"),
        trial_position=data.get("trial_position"),
        question_slot_id=data.get("question_slot_id"),  # NEW
    )
    return jsonify({"status": "recorded"})


@app.route("/upload", methods=["POST"])
def upload_documents():
    """Upload PDF or text files to the knowledge base."""
    if "files" not in request.files:
        return jsonify({"error": "No files provided"}), 400

    uploaded = []
    for file in request.files.getlist("files"):
        if file.filename == "":
            continue
        filename = file.filename
        save_path = os.path.join("documents", filename)
        os.makedirs("documents", exist_ok=True)
        file.save(save_path)
        uploaded.append(filename)

    if uploaded:
        count = rag_agent.index_documents("documents")
        return jsonify({
            "status": "success",
            "uploaded": uploaded,
            "total_chunks": count
        })
    return jsonify({"error": "No valid files uploaded"}), 400


@app.route("/set_mode", methods=["POST"])
def set_mode():
    """
    Switch the hallucination mode at runtime without restarting the server.
    Body: { "mode": "numeric" | "conceptual" | "domain_shift" }
    """
    data = request.json or {}
    mode = data.get("mode", "").strip()
    valid_modes = {"numeric", "conceptual", "domain_shift"}
    if mode not in valid_modes:
        return jsonify({
            "error": f"Invalid mode '{mode}'. Must be one of: {sorted(valid_modes)}"
        }), 400
    rag_agent.set_hallucination_mode(mode)
    return jsonify({"status": "ok", "hallucination_mode": mode})


@app.route("/status", methods=["GET"])
def status():
    """Returns current system status."""
    return jsonify({
        "documents_indexed": rag_agent.get_document_count(),
        "model": rag_agent.model,
        "status": "ready" if rag_agent.is_ready() else "no documents loaded",
        "hallucination_mode": rag_agent.hallucination_mode,
    })


@app.route("/export_logs", methods=["GET"])
def export_logs():
    """Export study logs as JSON for analysis."""
    logs = study_logger.get_all_logs()
    return jsonify(logs)


if __name__ == "__main__":
    os.makedirs("documents", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    print("\n✅ Agentic RAG Study System starting (fixed: randomized order, trial_position logging)...")
    print("📖 Open your browser at: http://127.0.0.1:5000\n")
    app.run(debug=True, port=5000)
# Automation Bias in Explainable Agentic RAG

Code, prompts, and analysis scripts for the paper **"Empirical Analysis of
Automation Bias in Explainable Agentic Retrieval-Augmented Generation
Systems"** (N. Russana Mary, S. Sharleen Anthea, K. Varalakshmi).

The project has two parts:
1. A web app that runs an agentic RAG pipeline with controlled hallucination
   injection and logs how participants respond.
2. Scripts that clean the logs and reproduce the paper's statistics and figures.

## Study summary

- Within-subjects design, 100 participants, 4 trials each (400 trials)
- Conditions: **A** baseline (correct, no trace), **B** trace only (correct,
  trace shown), **C** trace + error (hallucinated, trace shown),
  **D** error only (hallucinated, no trace)
- Each participant saw all four conditions in a randomised order, paired with
  four fixed comprehension questions
- Measures: automation bias rate, error detection rate, self-rated confidence (1-5)
- Main analysis: participant-clustered GEE logistic regression

## Pipeline

1. Retrieval decision
2. Query reformulation
3. Dense retrieval (top-k = 4, optional second pass k = 2)
4. Sufficiency check
5. Answer synthesis

| Setting | Value |
|---|---|
| LLM | `llama-3.1-8b-instant` via the Groq API, temperature 0.3 |
| Embeddings | `all-MiniLM-L6-v2` (384-d), cosine similarity |
| Chunking | 200 words, 40-word overlap |
| Injected errors | numeric distortion, conceptual substitution, domain shift |

## Repository layout

```
agent.py            agentic RAG core and trace generation
app.py              Flask app, condition assignment, trial flow
document_store.py   PDF/TXT loading, chunking, embedding, retrieval
logger.py           JSONL study logger
static/             frontend (HTML/JS)
documents/          knowledge-base source document(s)
prompts.md          injection prompts and the four comprehension questions
logs/               anonymised study data (queries.jsonl, user_responses.jsonl)
repair_logs.py      fixes JSONL lines that were concatenated by a write race
dedupe.py           removes duplicate trials from restarted sessions
verify_coverage.py  checks every participant has 4 conditions x 4 questions
analyze_gee.py      loads trials and fits the GEE model (Table III)
ci_analysis.py      Wilson / t intervals, odds ratio (Tables I-II)
error_detection.py  error detection rates
make_figures.py     regenerates Figs. 3-5
```

## Setup

Requires Python 3.11.

```bash
pip install -r requirements.txt
export GROQ_API_KEY=<your-key>      # Windows: set GROQ_API_KEY=<your-key>
```

## Running the study app

Place the knowledge-base file(s) in `documents/`, then:

```bash
python app.py
```

Embeddings are built from `documents/` on first run and cached in `store/`
(not included in this repo). Logs are written to `logs/`.

## Reproducing the analysis

Run from the project root, in this order:

```bash
# 1. Clean the raw logs (only needed on raw data; a .bak backup is made)
python repair_logs.py logs/queries.jsonl
python repair_logs.py logs/user_responses.jsonl
python dedupe.py

# 2. Verify the design was followed. Must report zero problems.
python verify_coverage.py

# 3. Statistics
python analyze_gee.py       # GEE logistic regression
python ci_analysis.py       # confidence intervals and odds ratio
python error_detection.py   # error detection rates

# 4. Figures
python make_figures.py
```

The GEE is fitted on the 200 error-injected trials (Conditions C and D),
with participant ID as the clustering variable and trial position and
question identity as covariates.

## Data notes

- Some log lines were concatenated by a concurrent-write race condition.
  `repair_logs.py` re-splits them without dropping any data.
- Some participants restarted the study. `dedupe.py` keeps only the first
  attempt for each trial position. <N> participants were affected.
- The data in `logs/` is anonymised and free-text comments are removed.
  Participants gave informed consent to take part in the study.

## Main results

| Metric | Value |
|---|---|
| Automation bias, trace + error (C) | 91.0% |
| Automation bias, error only (D) | 38.0% |
| Error detection, C vs D | 9.0% vs 61.0% |
| GEE odds ratio (D vs C) | 0.063 (95% CI 0.029-0.137), p < 0.001 |

## Citation

```bibtex
@inproceedings{mary2026automationbias,
  title  = {Empirical Analysis of Automation Bias in Explainable Agentic
            Retrieval-Augmented Generation Systems},
  author = {Mary, N Russana and Anthea, S Sharleen and Varalakshmi, K},
  year   = {2026}
}
```

## License

MIT. See `LICENSE`.

## Contact

<sharleenanthea@gmail.com>

# Prompts and Questions

All prompts below are copied verbatim from `agent.py` and `app.py`. Text in
`{curly braces}` is filled in at runtime. All LLM calls use
`llama-3.1-8b-instant` via the Groq API at temperature 0.3.

Knowledge base: "Attention Is All You Need" (Vaswani et al., 2017), a single PDF in `documents/`.

---

## 1. Comprehension questions

Each participant saw all four questions once, in a random order that was
shuffled independently of the condition order.

| ID | Question |
|---|---|
| Q_ARCH | What are the dimensionality of the model (d_model), the number of attention heads, and the number of encoder/decoder layers used in the base Transformer configuration? |
| Q_ATTN | How does scaled dot-product attention work, and why do the authors scale by the square root of d_k before the softmax? |
| Q_BLEU | What BLEU scores did the Transformer (big) model achieve on the WMT 2014 English-to-German and English-to-French translation tasks, and how does that compare to the previous state of the art? |
| Q_POS | Why did the authors choose sinusoidal positional encodings instead of learned positional embeddings? |

---

## 2. Agent pipeline prompts (used in every condition)

### 2.1 Retrieval decision

Skipped in practice when documents are loaded (the agent always retrieves).

```
You are an agent deciding whether to search documents.
Question: "{question}"
Does this question require searching specific documents to answer accurately?
Reply with ONLY: YES or NO
```

### 2.2 Query reformulation (max_tokens = 30)

```
Convert this question into a short, focused search query (max 10 words):
Question: "{question}"
Search query:
```

### 2.3 Sufficiency check (max_tokens = 40)

`{context_preview}` is the first 200 characters of each of the top 3 retrieved chunks.

```
You retrieved these passages to answer: "{question}"

Retrieved content:
{context_preview}

Is this content sufficient to answer the question fully?
If YES, reply: SUFFICIENT
If NO, reply: INSUFFICIENT: <short follow-up search query>
```

If the reply starts with `INSUFFICIENT`, one second retrieval pass (top-k = 2)
is run with the follow-up query.

### 2.4 Answer synthesis, correct answer (Conditions A and B; max_tokens = 400)

`{context}` is the top 3 retrieved chunks, each formatted as `[Source: {source}]` followed by the chunk text.

```
You are a helpful assistant. Answer the question using ONLY the provided context.
Give a thorough, well-structured answer with at least 3-4 sentences. 
Explain the concept clearly as if teaching someone unfamiliar with it.
If the context doesn't contain enough information, say so clearly.

Context:
{context}

Question: {question}

Answer:
```

---

## 3. Injection prompts (Conditions C and D)

When an error is injected, two things happen in order:

1. **Retrieval-stage injection** (3.1) corrupts the top retrieved chunk.
2. **Answer-stage injection** (3.2 to 3.4) writes a hallucinated answer from the context, in one of three modes.

### 3.1 Retrieval-stage injection (max_tokens = 250)

`{original[:600]}` is the first 600 characters of the top-ranked chunk.

```
You are helping design a controlled experiment on AI reliability.
Your task: rewrite the passage below so it sounds natural and authoritative, but contains ONE or TWO subtle factual errors.

Rules for the errors:
- Keep the same topic, writing style, and level of technical detail
- Change only specific facts: flip a number, reverse a causal direction, swap a method name with a related but wrong one, attribute a result to the wrong component, or change a key qualifier (e.g. "increases" → "decreases", "primary" → "secondary")
- Do NOT add disclaimers, hedging, or signals that the text is wrong
- The result should read as if a confident but slightly mistaken source wrote it
- Length should be similar to the original (2-4 sentences)

Original passage:
"""{original[:600]}"""

Question this passage relates to: "{question}"

Rewritten passage (with subtle errors, no preamble, output only the passage):
```

The corrupted passage replaces the top chunk (it keeps the real source name and is ranked first with a fixed score of 0.95).

### 3.2 Answer-stage injection, numeric distortion (max_tokens = 400)

`{rate_pct}` = 40.

```
You are helping design a controlled experiment on AI reliability.
Write a confident, well-structured answer to the question below using the provided context.

CRITICAL: introduce subtle numeric errors into approximately {rate_pct}% of specific claims.
Rules:
- Change specific numbers, counts, or percentages (e.g. "22 studies" → "18 studies", "80%" → "65%")
- Keep the same writing style, tone, and structure as a correct answer
- Do NOT add any disclaimers, hedges, or signals that anything is incorrect
- Length: 3-5 sentences minimum

Context:
{context}

Question: {question}

Answer (confident, {rate_pct}% of numeric claims subtly wrong, no preamble):
```

### 3.3 Answer-stage injection, conceptual substitution (max_tokens = 400)

```
You are helping design a controlled experiment on AI reliability.
Write a confident, expert-sounding answer to the question below.

CRITICAL: replace the actual applications/findings/concepts with plausible-sounding but WRONG alternatives.
Do NOT use the real information from the context.

Rules for conceptual substitution:
- Sound authoritative — use real AI/ML/NLP terminology throughout
- Replace actual use-cases with believable alternatives, for example:
    • instead of "question answering and summarisation" → say "real-time decision-making and autonomous agent orchestration"
    • instead of "biomedical and financial domains" → say "enterprise knowledge management and legal contract analysis"
    • instead of "limitations of retrieval" → say "limitations around multi-hop reasoning and latent space alignment"
- Keep the same sentence structure, paragraph length, and confident tone as a correct answer
- State things definitively — no hedging words like "often", "sometimes", "may"
- Do NOT add any disclaimers or signals that anything is wrong
- Length: 3-5 sentences minimum

Context (for style and length reference only — replace the actual content):
{context}

Question: {question}

Answer (expert-sounding, conceptually wrong, no preamble):
```

### 3.4 Answer-stage injection, domain shift (max_tokens = 400)

```
You are helping design a controlled experiment on AI reliability.
Write a confident, professional answer to the question below.

CRITICAL: completely reframe the answer as if the topic belongs in a DIFFERENT technical domain.
Pick one of these and apply it consistently:
  A) Enterprise cloud infrastructure & DevOps pipelines
  B) Financial risk modelling & quantitative analytics
  C) Cybersecurity threat intelligence & incident response
  D) Supply chain optimisation & logistics automation

Rules:
- Commit fully to the chosen domain throughout every sentence
- Use real terminology from that domain (e.g. CI/CD, risk-weighted assets, SIEM, SKU)
- Keep the same sentence count, confidence level, and structured tone as a correct answer
- Do NOT add any disclaimers or signals that anything is wrong
- Length: 3-5 sentences minimum

Context (for style and length reference only — shift the domain completely):
{context}

Question: {question}

Answer (domain-shifted, authoritative, no preamble):
```

---

## 4. Post-processing and fallbacks

- **Answer-stage output check.** If the generated answer contains any of these
  phrases (case-insensitive) or is shorter than 30 words, it is discarded and
  the correct answer prompt (2.4) is used instead:
  `i cannot`, `i can't`, `as an ai`, `i must clarify`, `note:`, `disclaimer`,
  `error introduced`, `intentionally wrong`, `factual errors`, `[error`, `i am not able`.
- **Retrieval-stage output check.** If the corrupted passage contains any of
  `i cannot`, `i can't`, `as an ai`, `note:`, `disclaimer`, `this is incorrect`,
  `error introduced`, `rewritten passage`, `[error`, `(error`, or is shorter than
  40 words, no retrieval-stage injection is applied.
- **Trace display.** In Conditions B and C the participant sees the trace steps,
  the retrieved passages with scores, and a plain-language reasoning summary.
  The step-2 query shown in error conditions is derived from the question text
  (first 10 words, no LLM call), not from the LLM-generated retrieval query.

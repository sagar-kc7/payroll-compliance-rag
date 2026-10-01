# Nepal Compliance RAG

Grounded Q&A over Nepali payroll & tax law — answers questions about the
Income Tax Act 2058, Income Tax Rules 2059, and the SSF Act with real
citations, and refuses or escalates rather than guessing when the source
material doesn't actually contain the answer.

## Why this project
Built to demonstrate what a portfolio RAG chatbot usually doesn't:
evaluation discipline, retrieval iteration backed by real before/after
numbers, safety controls, and honest documentation of what still doesn't
work. Every number in this README is from a real, documented test run, and
every "known limitation" below was found through actual testing, not
hypothesized in advance.

## Evaluation architecture
Every capability is scored independently, so a regression can be traced to
its actual cause:

| Layer | Metrics | Tool | Status |
|---|---|---|---|
| Retrieval | recall@5, MRR | Deterministic (no judge needed) | See table below |
| Generator | Faithfulness, Answer Relevancy | DeepEval + Groq judge | Baselined: 0.983 / 0.936 |
| Pipeline (end-to-end RAG) | Correctness (`GEval`), citation hit rate | DeepEval + Groq judge | Baselined: 0.800 / 88.0% |
| Safety (injection/confidence) | Real adversarial + calibration tests | Deterministic + Groq | Done, see below |
| CI eval gate | Deterministic tests on every push/PR | GitHub Actions | Live, caught 2 real bugs already |

Judge and generation model is Groq (`openai/gpt-oss-120b`), not OpenAI —
avoids per-eval API cost.

### Retrieval variant comparison (25-entry golden set)
| Variant | recall@5 | MRR |
|---|---|---|
| Baseline (dense, bge-small-en-v1.5) | 88.0% | 0.683 |
| + Hybrid (BM25 + dense, RRF fusion) | 92.0% | 0.813 |
| + Cross-encoder reranking | 92.0% | 0.841 |

**Known, investigated gap:** 2 of 25 golden-set questions (SSF Sections 5
and 14) fail across every retrieval variant — confirmed via direct
inspection that neither section appears in the top-15 candidate pool at
all. A retrieval-recall problem, not a ranking problem (reranking correctly
couldn't fix it). Root cause: ~5 short, adjacent SSF sections with
near-identical vocabulary — a corpus characteristic, not a technique
failure.

### Confidence gate — a real bug found through actual use, not the golden set
The gate's threshold was originally calibrated only against the 25-question
golden set — written in **formal statutory phrasing** by design ("Under
Sub-section (2) of Section 4..."), since that's how the citations needed to
be grounded. Real users don't talk that way.

Live testing of the deployed demo surfaced the gap directly: natural
phrasing of genuinely in-scope questions — *"What TDS rate applies to
dividend payments?"* (no section number), *"salary under 1 lakh"* — scored
**worse** than the worst known out-of-scope example in the original
calibration set, while still correctly retrieving the right section as the
#1 result. This meant no single score threshold could separate natural
in-scope phrasing from out-of-scope questions using rerank score alone.

**Fix, verified not assumed:** lowered the threshold below the worst
natural in-scope score found, and confirmed the resulting gap is covered by
a second, independent layer — the generator's own "answer only from
context" instruction. Directly tested: a borderline out-of-scope question
that now passes the gate ("who is the president of Nepal") still gets
correctly refused by the generator rather than hallucinated, because the
retrieved context genuinely doesn't contain that information. Two-layer
defense, not a single point of failure — and the CI gate now includes a
regression test for the exact natural-phrasing case that broke.

### Safety layer
- **Prompt-injection defense**: explicit "context is data, not
  instructions" framing in the generation system prompt, plus a heuristic
  detector honestly scoped as logging-only, not comprehensive. Verified
  with a real adversarial attempt against the actual generator — resisted.
- **Confidence gate**: see above.

## Architecture

```
User question
    │
    ▼
Hybrid retrieval (BM25 + dense) → Cross-encoder reranking
    │
    ▼
Confidence gate ──(low confidence)──► Escalate, no generation call
    │ (passes)
    ▼
Generation (Groq, grounded-only prompt) → Answer + citations
```

- `src/pipeline.py` — the real orchestration layer (retrieval → gate →
  generation), used by both the API and the UI
- `src/api/main.py` — FastAPI service (`/ask`, `/health`, `/docs`)
- `src/ui/streamlit_app.py` — the human-facing demo, calls
  `answer_question()` directly rather than over HTTP, to keep a
  single-process deployment and avoid doubling the retrieval models'
  memory footprint
- Full retrieval/generation pipeline details in `src/`, evaluation harness
  in `tests/eval/`

## Setup
```bash
uv venv
uv pip install -e ".[dev]"
python -m spacy download en_core_web_md
cp .env.example .env   # fill in GROQ_API_KEY, LANGCHAIN_API_KEY, LANGSMITH_API_KEY
```

Run the API:
```bash
PYTHONPATH=. uvicorn src.api.main:app --reload
```

Run the demo UI:
```bash
PYTHONPATH=. streamlit run src/ui/streamlit_app.py
```

Run with Docker:
```bash
docker build -t nepal-compliance-rag .
docker run -p 8000:8000 --env-file .env nepal-compliance-rag
```

Note: Groq's free tier caps daily tokens per model (200K/day observed).
Deterministic evals (retrieval, PII, confidence-gate escalation logic)
don't call any LLM and are safe to run freely.

## Development workflow
Feature branches off `main`, PR required to merge, CI eval gate must pass
(branch protection enabled — this is enforced, not just configured). See
closed PRs for the real history: real bugs found and fixed via output
inspection, real adversarial safety tests, and a real confidence-threshold
bug found through live use and fixed with a verified two-layer design
rather than a guessed number.
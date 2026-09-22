# Nepal Compliance RAG

Grounded document-intelligence service for Nepali payroll & tax compliance:
extracts structured data from salary slips and answers compliance questions
with citations, refusing or escalating when confidence is low.

## Why this project
Built to demonstrate what a portfolio RAG chatbot doesn't: evaluation
discipline, structured extraction with validation, safety controls, and
measured cost/latency — the things GenAI/AI Engineer postings (Verisk,
Niural AI, Leapfrog) actually screen for beyond "can you call an LLM API."

## Evaluation architecture
Every capability is scored on its own, independently, so a regression can
be traced to its actual cause rather than a vague "something got worse":

| Layer | Metrics | Tool | Status |
|---|---|---|---|
| Retrieval | recall@5, MRR | Deterministic (no judge needed) | See table below |
| Generator | Faithfulness, Answer Relevancy | DeepEval + Groq judge | Baselined: 0.983 / 0.936 |
| Pipeline (end-to-end RAG) | Correctness (`GEval`), citation hit rate | DeepEval + Groq judge | Baselined: 0.800 / 88.0% |
| Extraction (field-level) | Per-field accuracy vs. golden set | Deterministic comparison | Baselined: 100.0% (8-slip set, see scope note below) |
| Application | p50/p95 latency, cost/query, escalation rate | LangSmith traces | Not started |

Judge model is Groq (`openai/gpt-oss-120b`), not OpenAI — avoids per-eval
cost entirely; AWS credits kept in reserve for a Phase 6 cost-comparison
demo rather than spent here.

### Retrieval variant comparison (25-entry golden set)
| Variant | recall@5 | MRR |
|---|---|---|
| Baseline (dense, bge-small-en-v1.5) | 88.0% | 0.683 |
| + Hybrid (BM25 + dense, RRF fusion) | 92.0% | 0.813 |
| + Cross-encoder reranking | 92.0% | 0.841 |

**Known, investigated gap:** 2 of 25 questions (SSF Sections 5 and 14)
fail across every retrieval variant. Confirmed via direct inspection —
neither section appears anywhere in the top-15 candidate pool from
hybrid retrieval, for either query. Retrieval-recall problem, not a
ranking problem (so reranking correctly couldn't fix it). Root cause:
~5 short, adjacent SSF sections using near-identical vocabulary — a
corpus characteristic, not a technique failure. Not chased further with
contextual retrieval (real token cost for a narrow, understood gap).

### Extraction accuracy — scope note
100% field accuracy (64/64 checks) on 8 hand-verified synthetic salary
slips. These slips use a consistent, clearly-labeled format — this
measures accuracy on well-formatted input, not resilience to genuinely
ambiguous/noisy real-world formatting. A separate robustness check
(`test_extraction.py`) confirms extraction survives messier formatting
without crashing, but that test has no scoreable ground truth by design.

## Status

**Phase 1 — Corpus & structure-aware chunking: done**
- Corpus: Income Tax Act 2058, Income Tax Rules 2059, SSF Act (English
  primary sources only — see `SOURCES.md`)
- Loader handles Section/Rule/Schedule citation with monotonic-numbering
  guards, multi-heading-per-page splitting, and explicit exclusion of a
  ~440-page consolidated amendment-ordinance range with no citable
  heading structure
- 3 real bugs found and fixed via direct output inspection — see closed
  PR history on `main`

**Phase 2 — Golden dataset & evaluation harness: done**
- Golden set finalized at 25 hand-written entries across 6 categories
- Retrieval, generator, and pipeline layers all baselined and gated

**Phase 3 — Retrieval iteration: done (stopped deliberately)**
- Hybrid retrieval + cross-encoder reranking, see comparison table above
- Contextual retrieval not pursued — remaining gap is a corpus
  characteristic, not a chunking/context problem

**Phase 4 — Structured extraction: done**
- `SalarySlip` Pydantic schema, every field grounded in a specific
  statute provision (Section 5/8 income, Section 64 retirement
  contribution, SSF Act Section 7 contributions, Section 87 TDS)
- Extraction via `instructor` + Groq, native validation-repair loop
  (`max_retries`) rather than a hand-rolled one
- Arithmetic cross-check (gross = basic + allowances, net = gross −
  deductions) as a collected-warnings model validator, not a hard
  failure — tested for both false negatives and true positives
- Field-level accuracy: 100% on the 8-slip golden set (scope-noted above)

**Phase 5 — Safety layer: not started (current focus)**
- [ ] PII redaction/detection before data leaves the service
- [ ] Prompt-injection defense on document/slip text (untrusted input)
- [ ] Confidence threshold — escalate to human instead of guessing
- [ ] Grounded citations with clause IDs surfaced to the end user

**Later phases: not started**
- [ ] Observability (LangSmith) + cost/latency-aware model routing
      (candidate use for the reserved AWS credits: Bedrock vs Groq
      cost comparison)
- [ ] CI eval gate (DeepEval in GitHub Actions, blocks merge on regression)
- [ ] FastAPI service + deployment

## Setup
```bash
uv venv
uv pip install -e ".[dev]"
cp .env.example .env   # fill in GROQ_API_KEY, LANGCHAIN_API_KEY
```

Note: Groq's free tier caps daily tokens per model (200K/day observed).
Eval runs involving the judge/extraction model can hit this during heavy
iteration — most `tests/eval/` scripts support `--sample N` or run a
small number of calls by design. Deterministic retrieval evals
(`test_retrieval*.py`) don't call any LLM and are safe to run freely.

## Development workflow
Feature branches off `main`, PR required to merge (branch protection
enabled). See closed PRs for the actual history, including several real
bugs found and fixed via output inspection rather than trusting summary
metrics alone — that process is as much the point of this repo as the
final numbers.
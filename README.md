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
Four independent layers, each scored separately so a regression can be
traced to its cause:

| Layer | Metrics | Tool | Status |
|---|---|---|---|
| Retrieval | recall@5, MRR | Deterministic (no judge needed) | See table below |
| Generator | Faithfulness, Answer Relevancy | DeepEval + Groq judge | Baselined: 0.983 / 0.936 |
| Pipeline (end-to-end) | Correctness (`GEval`), citation hit rate | DeepEval + Groq judge | Baselined: 0.800 / 88.0% |
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
fail across every variant. Confirmed via direct inspection — neither
section appears anywhere in the top-15 candidate pool from hybrid
retrieval, for either query. This is a retrieval-recall problem, not a
ranking problem, so reranking correctly couldn't fix it (and didn't).
Root cause: the SSF Act has ~5 short, adjacent sections (4/5/8/9/14) using
near-identical "contribution/deposit/employer" vocabulary — a corpus
characteristic, not a technique failure. Deliberately not chased further
with contextual retrieval (would cost real LLM-generation tokens per
chunk for a narrow, well-understood 8% gap) — logged as a known
limitation instead of silently accepted.

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
- All three eval layers (retrieval, generator, pipeline) baselined and
  regression-gated

**Phase 3 — Retrieval iteration: done (stopped deliberately, see table above)**
- [x] Hybrid (BM25 + dense) retrieval
- [x] Cross-encoder reranking
- [ ] Contextual retrieval — not pursued; remaining gap is a corpus
      characteristic (near-duplicate SSF sections), not a chunking/context
      problem, and the fix would cost real tokens for narrow gain

**Phase 4 — Structured extraction: not started (current focus)**
- [ ] Salary slip → Pydantic schema extraction
- [ ] Validation + bounded repair loop on schema/arithmetic failures
- [ ] Field-level accuracy eval against a labelled set

**Later phases: not started**
- [ ] Safety layer: PII redaction, prompt-injection defense, confidence
      gate for escalation
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
Eval runs involving the judge model can hit this during heavy iteration —
`tests/eval/` scripts support `--sample N` to test cheaply before running
the full golden set. Deterministic retrieval evals (`test_retrieval*.py`)
don't call the judge at all and are safe to run freely.

## Development workflow
Feature branches off `main`, PR required to merge (branch protection
enabled). See closed PRs for the actual history, including several real
bugs found and fixed via output inspection rather than trusting summary
metrics alone — that process is as much the point of this repo as the
final numbers.
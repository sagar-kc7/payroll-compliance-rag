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
| Retrieval | recall@5, MRR | Deterministic (no judge needed) | Baselined: 80.0% / 0.661 |
| Generator | Faithfulness, Answer Relevancy | DeepEval + Groq judge | Baselined: 0.983 / 0.936 |
| Pipeline (end-to-end) | Correctness (`GEval`), citation hit rate | DeepEval + Groq judge | Baselined: 0.800 / 88.0% |
| Application | p50/p95 latency, cost/query, escalation rate | LangSmith traces | Not started |

Judge model is Groq (`openai/gpt-oss-120b`), not OpenAI — avoids per-eval
cost entirely; AWS credits kept in reserve for a Phase 6 cost-comparison
demo rather than spent here. Offline eval (DeepEval) runs against the
golden set; CI gate (blocks merge on regression) not wired up yet —
planned for Phase 6.

## Status

**Phase 1 — Corpus & structure-aware chunking: done**
- Corpus: Income Tax Act 2058, Income Tax Rules 2059, SSF Act (English
  primary sources only — see `SOURCES.md`)
- Loader handles Section/Rule/Schedule citation with monotonic-numbering
  guards, multi-heading-per-page splitting, and explicit exclusion of a
  ~440-page consolidated amendment-ordinance range (pages 190-626 of the
  Income Tax Act) that has no citable heading structure
- 3 real bugs found and fixed via direct output inspection, not assumed
  from summary metrics — see closed PR history on `main`

**Phase 2 — Golden dataset & evaluation harness: done**
- Golden set finalized at **25 hand-written entries** (deliberate scope
  decision, not a partial toward a larger target): tax_computation 8,
  ssf 5, exemptions 4, tax_rates 4, tds 3, definitions 1
- Baseline dense retriever (bge-small-en-v1.5 + Chroma) — recall@5 80.0%,
  MRR 0.661
- Basic grounded generator (Groq `openai/gpt-oss-120b`) — faithfulness
  0.983, relevancy 0.936 (measured on known-correct context, isolating
  generator quality from retrieval quality)
- Full pipeline eval — correctness 0.800, citation hit rate 88.0%. Of 6
  low-scoring questions: 3 were genuine retrieval misses where the
  generator correctly refused rather than hallucinating (a behavior
  worth preserving, not "fixing"); 2 were real generator completeness
  gaps (correct headline fact, dropped stated caveats); 1 exposed a
  blind spot in the citation_hit metric itself (section-level match
  doesn't guarantee the specific needed sub-chunk was retrieved, for
  any section split across many chunks — e.g. Section 2's 17 sub-chunks)
- All three layers gated with regression-floor assertions in `tests/eval/`

**Phase 3 — Retrieval iteration: not started**
- [ ] Hybrid (BM25 + dense) retrieval
- [ ] Contextual retrieval
- [ ] Reranking
- [ ] Re-run `test_retrieval.py` after each change, keep a before/after
      results table — this table is the actual portfolio deliverable

**Later phases: not started**
- [ ] Structured extraction (salary slip → Pydantic schema) with
      validation repair loops
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
the full golden set.

## Development workflow
Feature branches off `main`, PR required to merge (branch protection
enabled). See closed PRs for the actual history, including several real
bugs found and fixed via output inspection rather than trusting summary
metrics alone — that process is as much the point of this repo as the
final numbers.
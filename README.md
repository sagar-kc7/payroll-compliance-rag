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

| Layer | Metrics | Tool |
|---|---|---|
| Retrieval | Recall@k, MRR, Contextual Precision/Recall | DeepEval |
| Generator | Faithfulness, Answer Relevancy, Hallucination | DeepEval |
| Pipeline (end-to-end) | Correctness vs. golden answer, citation accuracy | DeepEval `GEval` |
| Application | p50/p95 latency, cost/query, escalation rate | LangSmith traces |

Offline eval (DeepEval) runs in CI against the golden set and gates merges.
Online eval (LangSmith) traces live/demo traffic for latency and cost.

## Status
- [x] Project scaffolded
- [x] Corpus sources identified and documented (`data/SOURCES.md`) —
      **action needed: you download the actual PDFs, I can't fetch them**
- [x] PDF loader written (`src/ingestion/loader.py`) — page-level
      extraction with best-effort clause detection for citations
- [ ] **Next: run the loader against real PDFs, confirm clause detection
      works, then build the structure-aware chunker** (reusing your
      `~/Chunk` work)
- [ ] Golden set (0/60 written — schema defined in `data/golden_set/schema.md`)
- [ ] Baseline retriever
- [ ] DeepEval test suite (retrieval / generator / pipeline)
- [ ] Extraction schema + repair loop
- [ ] Safety layer (PII redaction, injection defense, confidence gate)
- [ ] Observability (LangSmith) + CI gate
- [ ] FastAPI service + deploy

## Known scope decision pending
`data/SOURCES.md` flags that IRD's directives are Nepali-script only, and
the loader's clause detector currently only handles English "Section N"
patterns. Decide explicitly: English-only corpus for v1 (documented
limitation), or add Devanagari clause detection now. Don't let this get
decided by default.

## Setup
```bash
uv venv
uv pip install -e ".[dev]"
cp .env.example .env   # fill in GROQ_API_KEY, LANGCHAIN_API_KEY
```

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
be traced to its actual cause:

| Layer | Metrics | Tool | Status |
|---|---|---|---|
| Retrieval | recall@5, MRR | Deterministic (no judge needed) | See table below |
| Generator | Faithfulness, Answer Relevancy | DeepEval + Groq judge | Baselined: 0.983 / 0.936 |
| Pipeline (end-to-end RAG) | Correctness (`GEval`), citation hit rate | DeepEval + Groq judge | Baselined: 0.800 / 88.0% |
| Extraction (field-level) | Per-field accuracy vs. golden set | Deterministic comparison | Baselined: 100.0% (8-slip set, scoped) |
| Safety (PII/injection/confidence) | Real adversarial + calibration tests | Deterministic + Groq | Done, see below |
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
fail across every retrieval variant — confirmed via direct inspection
that neither section appears in the top-15 candidate pool at all, a
retrieval-recall problem, not a ranking problem. Root cause: ~5 short,
adjacent SSF sections with near-identical vocabulary. Not chased further
with contextual retrieval (real token cost for a narrow, understood gap).

### Extraction accuracy — scope note
100% field accuracy (64/64 checks) on 8 hand-verified synthetic salary
slips, all using a consistent, clearly-labeled format — measures accuracy
on well-formatted input, not resilience to ambiguous/noisy formatting
(covered separately, without a scoreable ground truth, by a messy-format
robustness test).

### Safety layer
- **PII redaction** (Presidio + spacy `en_core_web_md`, with a custom
  Nepali PAN recognizer). Model choice tested empirically — the smaller
  `en_core_web_sm` missed a real name entirely; medium caught it.
  4 tests: positive detection, actual redaction, no false positives on
  real statute text, no false positives on currency amounts.
- **Prompt-injection defense**: explicit "context is data, not
  instructions" framing in both system prompts, plus a heuristic
  detector honestly scoped as a logging/flagging layer, not a
  comprehensive defense. Verified with real adversarial attempts against
  the actual extractor and generator (a hijack instruction embedded in a
  slip and in a retrieved context chunk) — both resisted. One tested
  attack phrasing; documented as evidence, not a general guarantee.
- **Confidence gate**: threshold (-2.0 on cross-encoder rerank score)
  calibrated from real data — 25 in-scope golden-set questions scored
  1.340-7.650, 5 out-of-scope questions scored -11.166 to -4.904, clean
  separation. Out-of-scope questions escalate before the generator is
  even called (cost-saving, confirmed). Noted honestly: a rephrased
  in-scope question scored 0.367 — correctly above threshold but well
  below the golden set's own minimum, since calibration only covered
  literal golden-set phrasings, not paraphrase variation.
- This also fixed a real gap: `src/pipeline.py` is the first module
  that actually uses the hybrid+reranked retriever Phase 3 found best —
  until now it only existed in eval scripts.

## Status

**Phase 1 — Corpus & structure-aware chunking: done**
**Phase 2 — Golden dataset & evaluation harness: done**
**Phase 3 — Retrieval iteration: done (stopped deliberately)**
**Phase 4 — Structured extraction: done**
**Phase 5 — Safety layer: done** (PII redaction, injection defense,
calibrated confidence gate — details above)

**Phase 6 — Observability & cost engineering: not started (current focus)**
- [ ] LangSmith tracing wired into the real pipeline (not just eval scripts)
- [ ] Cost/latency measurement per query
- [ ] Model routing (candidate use for the reserved AWS credits: Bedrock
      vs Groq cost comparison — real $ numbers from actual usage)
- [ ] CI eval gate (DeepEval in GitHub Actions, blocks merge on regression)

**Phase 7 — Ship it: not started**
- [ ] FastAPI service wrapping `src/pipeline.py` and the extraction path
- [ ] Docker
- [ ] Deploy

## Setup
```bash
uv venv
uv pip install -e ".[dev]"
python -m spacy download en_core_web_md
cp .env.example .env   # fill in GROQ_API_KEY, LANGCHAIN_API_KEY
```

Note: Groq's free tier caps daily tokens per model (200K/day observed).
Eval runs involving the judge/extraction model can hit this during heavy
iteration — most `tests/eval/` scripts support `--sample N` or run a
small number of calls by design. Deterministic evals (retrieval, PII,
confidence-gate calibration) don't call any LLM and are safe to run freely.

## Development workflow
Feature branches off `main`, PR required to merge (branch protection
enabled). See closed PRs for the actual history, including several real
bugs found and fixed via output inspection, and real adversarial/
calibration tests rather than trusted-by-assumption safety claims —
that process is as much the point of this repo as the final numbers.
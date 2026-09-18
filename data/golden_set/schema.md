# Golden set format

One JSON object per line (`golden_set.jsonl`). Each entry is a real
question a payroll/tax user would ask, with the ground-truth answer and
the exact source clause it must be grounded in.

```json
{
  "id": "gs_001",
  "question": "What is the employee TDS rate on interest income sourced in Nepal?",
  "expected_answer": "15%",
  "source_file": "tds_rates_2077_78.pdf",
  "source_clause": "Sub-section (1) of Section 88",
  "category": "tds",
  "difficulty": "easy"
}
```

Field notes:
- `expected_answer` — keep it short and checkable, not a paragraph. DeepEval's
  `GEval` will compare the generated answer's *meaning* against this, so a
  precise short answer is more useful than prose.
- `source_clause` — must match what your chunker's metadata will actually
  produce (see loader.py's `detected_clause`). If it can't match, your
  citation-accuracy metric can't score it — keep this field honest to what
  the pipeline can technically cite, not to what you wish it could cite.
- `category` — lets you slice results later (e.g. "TDS questions score 0.91
  faithfulness, SSF questions score 0.74" — that's a finding, not just a
  number).
- `difficulty` — `easy` (single-clause lookup), `medium` (requires combining
  2 clauses), `hard` (requires a calculation, e.g. computing net salary).

## Target size
60–100 entries. Write these yourself, by hand, after reading the actual
source PDFs — don't have an LLM generate them from the documents; a
model-generated golden set can't catch the model's own blind spots, and an
interviewer asking "how was this validated" deserves a real answer ("I wrote
and checked each one against the primary source").

## Status
0/60 written. This is Phase 2 — don't start it until the corpus (Phase 1) is
loaded and you've confirmed clause detection is working on real pages.

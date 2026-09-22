"""
Pipeline-layer evaluation: the real end-to-end path — actual retriever,
actual generator, no known-correct-context shortcut. This is where
retrieval misses (test_retrieval.py) and generator weaknesses
(test_generator.py) compound into a single visible failure, which is
the point: a user never gets known-correct context, they get whatever
the retriever actually finds.

Two things measured per question:
- GEval correctness: does the final answer substantively match the
  golden expected_answer (allowing phrasing differences)?
- citation_hit: did the retriever's top-k actually include the true
  source section? (same check as test_retrieval.py, repeated here so a
  low correctness score can be attributed to "retrieval missed" vs
  "generation went wrong despite having the right context.")

Run standalone for a human-readable report (supports a small sample
size to sanity-check before spending a full quota's worth of judge
calls):
    PYTHONPATH=. python -m tests.eval.test_pipeline --sample 3
    PYTHONPATH=. python -m tests.eval.test_pipeline

Run under pytest:
    pytest tests/eval/test_pipeline.py -m eval
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from src.eval.judge import GroqJudge
from src.generation.answer import generate_answer
from src.retrieval.baseline import get_collection, retrieve

ROOT = Path(__file__).parent.parent.parent
GOLDEN_SET_PATH = ROOT / "data" / "golden_set" / "golden_set.jsonl"


def load_golden_set() -> list[dict]:
    with GOLDEN_SET_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _correctness_metric(judge: GroqJudge) -> GEval:
    return GEval(
        name="Correctness",
        criteria=(
            "Determine whether 'actual output' substantively answers the "
            "question correctly compared to 'expected output'. Different "
            "phrasing, wording, or added explanation is fine — but the "
            "concrete facts that matter (rates, percentages, section "
            "numbers, named conditions) must match. Missing or wrong "
            "concrete facts should be penalized even if the overall tone "
            "sounds confident and correct."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        model=judge,
        threshold=0.5,
    )


def evaluate_pipeline(k: int = 5, sample_size: int | None = None) -> dict:
    golden = load_golden_set()
    if sample_size:
        golden = golden[:sample_size]

    collection = get_collection()
    judge = GroqJudge()
    correctness_metric = _correctness_metric(judge)

    results = []
    for entry in golden:
        retrieved = retrieve(entry["question"], k=k, collection=collection)
        citation_hit = any(
            r["source_file"] == entry["source_file"] and r["section"] == entry["source_clause"]
            for r in retrieved
        )

        answer = generate_answer(entry["question"], retrieved)

        test_case = LLMTestCase(
            input=entry["question"],
            actual_output=answer,
            expected_output=entry["expected_answer"],
            retrieval_context=[r["text"] for r in retrieved],
        )
        correctness_metric.measure(test_case)

        results.append(
            {
                "id": entry["id"],
                "question": entry["question"],
                "answer": answer,
                "expected": entry["expected_answer"],
                "citation_hit": citation_hit,
                "correctness": correctness_metric.score,
                "correctness_reason": correctness_metric.reason,
            }
        )

    avg_correctness = sum(r["correctness"] for r in results) / len(results)
    citation_hit_rate = sum(r["citation_hit"] for r in results) / len(results)

    return {
        "total": len(golden),
        "avg_correctness": avg_correctness,
        "citation_hit_rate": citation_hit_rate,
        "per_question": results,
    }


@pytest.mark.eval
def test_pipeline_correctness():
    """
    Baseline established 2026-09 against the finalized 25-entry golden
    set: avg correctness = 0.800, citation hit rate = 88.0%.

    6 low-scoring questions break into two distinct causes:
    - gs_004, gs_012, gs_013: genuine retrieval misses (same 3 found by
      test_retrieval.py standalone, recall@5=80%). Generator correctly
      refused to answer rather than hallucinating when given wrong
      context — GEval still scores this 0.0 against the expected fact
      (correctly), but this refusal behavior is worth preserving in
      Phase 5, not "fixing."
    - gs_016, gs_023: real generator completeness gaps — correct
      headline fact, but caveats/cross-references from the source text
      were dropped. Candidate prompt fix: explicitly instruct the
      generator to include stated conditions/exceptions, not just the
      primary number.
    - gs_025: citation_hit=True but correctness=0.0 — a blind spot in
      citation_hit itself. Section 2 spans 17 sub-chunks all labeled
      "Section 2"; citation_hit only checks section-level match, not
      whether the SPECIFIC sub-chunk containing the needed clause was
      retrieved. Any section split across many chunks can produce this
      false-positive citation_hit. Not fixed here — would need
      chunk_id-level (not section-level) ground truth in the golden
      set to catch it properly.

    Thresholds set below today's real numbers as a regression floor.
    """
    result = evaluate_pipeline()
    print(f"\navg correctness: {result['avg_correctness']:.3f}")
    print(f"citation hit rate: {result['citation_hit_rate']:.2%}")
    assert result["avg_correctness"] >= 0.65, f"correctness dropped to {result['avg_correctness']:.3f}"
    assert result["citation_hit_rate"] >= 0.75, f"citation hit rate dropped to {result['citation_hit_rate']:.2%}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None, help="Only evaluate the first N questions")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    result = evaluate_pipeline(k=args.k, sample_size=args.sample)

    print(f"Questions evaluated: {result['total']}")
    print(f"avg correctness: {result['avg_correctness']:.3f}")
    print(f"citation hit rate: {result['citation_hit_rate']:.2%}\n")

    low = [r for r in result["per_question"] if r["correctness"] < 0.7 or not r["citation_hit"]]
    if low:
        print(f"{len(low)} question(s) with correctness < 0.7 or a citation miss:\n")
        for r in low:
            print(f"  [{r['id']}] {r['question']}")
            print(f"    citation_hit: {r['citation_hit']}")
            print(f"    correctness: {r['correctness']:.2f} — {r['correctness_reason']}")
            print(f"    answer: {r['answer'][:200]}")
            print()
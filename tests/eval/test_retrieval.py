"""
Retrieval-layer evaluation: recall@k and MRR against the golden set.

Deliberately does NOT use DeepEval's LLM-judged metrics here — whether
the correct section appears in the top-k results is a fact we can check
by exact match against the golden set's source_clause, not a judgment
call. Faithfulness/GEval (which genuinely need a judge) come in at the
generator and pipeline layers, not here.

Run standalone for a human-readable report:
    python -m tests.eval.test_retrieval

Run under pytest for the pass/fail version once a baseline number exists
to gate against:
    pytest tests/eval/test_retrieval.py -m eval
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.retrieval.baseline import get_collection, retrieve

ROOT = Path(__file__).parent.parent.parent
GOLDEN_SET_PATH = ROOT / "data" / "golden_set" / "golden_set.jsonl"


def load_golden_set() -> list[dict]:
    with GOLDEN_SET_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def evaluate_retrieval(k: int = 5) -> dict:
    """
    For each golden question, retrieve top-k and check whether a chunk
    matching (source_file, source_clause) appears among them.

    recall@k: fraction of questions where the correct section appears
              anywhere in the top-k.
    MRR:      mean reciprocal rank of the correct section's first
              appearance (1.0 if it's the top result, 0 if absent).
    """
    golden = load_golden_set()
    collection = get_collection()

    hits_at_k = 0
    reciprocal_ranks = []
    per_question = []

    for entry in golden:
        results = retrieve(entry["question"], k=k, collection=collection)

        rank = None
        for i, r in enumerate(results, start=1):
            if r["source_file"] == entry["source_file"] and r["section"] == entry["source_clause"]:
                rank = i
                break

        found = rank is not None
        hits_at_k += int(found)
        reciprocal_ranks.append(1.0 / rank if found else 0.0)

        per_question.append(
            {
                "id": entry["id"],
                "question": entry["question"],
                "expected": f"{entry['source_file']} :: {entry['source_clause']}",
                "found_at_rank": rank,
                "top_result": f"{results[0]['source_file']} :: {results[0]['section']}" if results else None,
            }
        )

    return {
        "k": k,
        "recall_at_k": hits_at_k / len(golden),
        "mrr": sum(reciprocal_ranks) / len(golden),
        "total_questions": len(golden),
        "per_question": per_question,
    }


@pytest.mark.eval
def test_retrieval_recall_meets_baseline():
    """
    Baseline established 2026-09 with the bge-small-en-v1.5 embedding
    model against the 15-question pilot golden set: recall@5 = 80.0%,
    MRR = 0.661. Threshold set below that as a regression floor, not a
    target — Phase 4 (hybrid retrieval, reranking) should clear it by a
    wide margin; this assertion exists to catch accidental regressions
    (e.g. a bad re-index), not to represent "good" retrieval.
    """
    result = evaluate_retrieval(k=5)
    print(f"\nrecall@5: {result['recall_at_k']:.2%}")
    print(f"MRR: {result['mrr']:.3f}")
    assert result["recall_at_k"] >= 0.70, (
        f"recall@5 dropped to {result['recall_at_k']:.2%}, below the 70% floor "
        f"(baseline was 80.0%) — check for an index/corpus mismatch before "
        f"assuming this is a real retrieval regression"
    )


if __name__ == "__main__":
    result = evaluate_retrieval(k=5)

    print(f"Questions evaluated: {result['total_questions']}")
    print(f"recall@{result['k']}: {result['recall_at_k']:.2%}")
    print(f"MRR: {result['mrr']:.3f}\n")

    misses = [q for q in result["per_question"] if q["found_at_rank"] is None]
    if misses:
        print(f"{len(misses)} question(s) where the correct section was NOT in top-{result['k']}:\n")
        for m in misses:
            print(f"  [{m['id']}] {m['question']}")
            print(f"    expected: {m['expected']}")
            print(f"    top result instead: {m['top_result']}")
            print()
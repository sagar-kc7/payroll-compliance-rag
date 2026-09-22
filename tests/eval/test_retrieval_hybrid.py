"""
Hybrid retrieval evaluation — same golden set, same recall@k/MRR
harness as the baseline (test_retrieval.py), so numbers are directly
comparable. Baseline: recall@5 = 80.0%, MRR = 0.661.

Run standalone:
    PYTHONPATH=. python -m tests.eval.test_retrieval_hybrid

Run under pytest:
    pytest tests/eval/test_retrieval_hybrid.py -m eval
"""

from __future__ import annotations

import pytest

from src.retrieval.hybrid import retrieve as hybrid_retrieve
from tests.eval.test_retrieval import evaluate_retrieval


def evaluate_hybrid_retrieval(k: int = 5) -> dict:
    return evaluate_retrieval(k=k, retrieve_fn=lambda q, kk: hybrid_retrieve(q, k=kk))


@pytest.mark.eval
def test_hybrid_retrieval_recall():
    """
    First run — no threshold yet. Run standalone, record the real
    recall@5/MRR, compare against baseline (80.0% / 0.661), THEN add a
    real assertion — ideally one that requires hybrid to be at least as
    good as baseline, not just "not terrible."
    """
    result = evaluate_hybrid_retrieval(k=5)
    print(f"\n[hybrid] recall@5: {result['recall_at_k']:.2%}")
    print(f"[hybrid] MRR: {result['mrr']:.3f}")


if __name__ == "__main__":
    result = evaluate_hybrid_retrieval(k=5)
    print(f"Questions evaluated: {result['total_questions']}")
    print(f"[hybrid] recall@5: {result['recall_at_k']:.2%}")
    print(f"[hybrid] MRR: {result['mrr']:.3f}\n")

    misses = [q for q in result["per_question"] if q["found_at_rank"] is None]
    if misses:
        print(f"{len(misses)} question(s) where the correct section was NOT in top-5:\n")
        for m in misses:
            print(f"  [{m['id']}] {m['question']}")
            print(f"    expected: {m['expected']}")
            print(f"    top result instead: {m['top_result']}")
            print()
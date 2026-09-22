"""
Reranked retrieval evaluation — hybrid candidates reranked with a
cross-encoder. Same golden set and harness as baseline/hybrid, for
direct comparison.

Baseline (dense only): recall@5 = 88.0%, MRR = 0.683
Hybrid (BM25 + dense):  recall@5 = 92.0%, MRR = 0.813

Run standalone:
    PYTHONPATH=. python -m tests.eval.test_retrieval_reranked

Run under pytest:
    pytest tests/eval/test_retrieval_reranked.py -m eval
"""

from __future__ import annotations

import pytest

from src.retrieval.reranker import retrieve as reranked_retrieve
from tests.eval.test_retrieval import evaluate_retrieval


def evaluate_reranked_retrieval(k: int = 5) -> dict:
    return evaluate_retrieval(k=k, retrieve_fn=lambda q, kk: reranked_retrieve(q, k=kk))


@pytest.mark.eval
def test_reranked_retrieval_recall():
    """
    First run — no threshold yet. Compare against hybrid (92.0%/0.813)
    before setting a gate, and specifically check whether gs_012/gs_013
    (the SSF Section confusion that survived hybrid retrieval) are
    fixed — that's the actual point of adding this layer.
    """
    result = evaluate_reranked_retrieval(k=5)
    print(f"\n[reranked] recall@5: {result['recall_at_k']:.2%}")
    print(f"[reranked] MRR: {result['mrr']:.3f}")


if __name__ == "__main__":
    result = evaluate_reranked_retrieval(k=5)
    print(f"Questions evaluated: {result['total_questions']}")
    print(f"[reranked] recall@5: {result['recall_at_k']:.2%}")
    print(f"[reranked] MRR: {result['mrr']:.3f}\n")

    misses = [q for q in result["per_question"] if q["found_at_rank"] is None]
    if misses:
        print(f"{len(misses)} question(s) where the correct section was NOT in top-5:\n")
        for m in misses:
            print(f"  [{m['id']}] {m['question']}")
            print(f"    expected: {m['expected']}")
            print(f"    top result instead: {m['top_result']}")
            print()
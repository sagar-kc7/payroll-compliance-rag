"""
Production RAG pipeline: retrieve (best variant from Phase 3: hybrid +
reranking) -> confidence gate -> generate, or escalate.

This is also the fix for a real gap: until now, hybrid/reranked
retrieval only existed in eval scripts (test_retrieval_hybrid.py,
test_retrieval_reranked.py) — the actual answer-generation path
(test_pipeline.py) was still calling the plain baseline retriever
directly. This module is the first to actually use the retrieval
variant Phase 3 found best.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.generation.answer import generate_answer
from src.retrieval.reranker import retrieve

# Calibrated against real data (tests/eval/calibrate_confidence_threshold.py),
# not guessed. Measured 2026-09: in-scope (25-question golden set) top-1
# rerank scores ranged 1.340-7.650 (mean 4.544); 5 deliberately
# out-of-scope questions ranged -11.166 to -4.904 (highest: "Who is the
# current president of Nepal?" — topically Nepal-adjacent but not
# payroll/tax-relevant, still correctly scored low). Clean ~6.2-point
# gap between the two groups. -2.0 sits at the midpoint with balanced
# margin on both sides. Re-run the calibration script if the corpus,
# embedding model, or reranker model ever changes — this number is
# specific to today's configuration, not a universal constant.
LOW_CONFIDENCE_RERANK_THRESHOLD = -2.0

ESCALATION_MESSAGE = (
    "I don't have enough confidence in the available source material to "
    "answer this accurately. Please consult a tax professional or the "
    "original statute directly, or rephrase your question."
)


@dataclass
class AnswerResult:
    answer: str
    citations: list[str]
    confidence_score: float | None
    escalated: bool
    escalation_reason: str | None = None


def answer_question(question: str, k: int = 5) -> AnswerResult:
    results = retrieve(question, k=k)

    if not results:
        return AnswerResult(
            answer=ESCALATION_MESSAGE,
            citations=[],
            confidence_score=None,
            escalated=True,
            escalation_reason="no_results_retrieved",
        )

    top_score = results[0].get("rerank_score")
    if top_score is not None and top_score < LOW_CONFIDENCE_RERANK_THRESHOLD:
        return AnswerResult(
            answer=ESCALATION_MESSAGE,
            citations=[],
            confidence_score=top_score,
            escalated=True,
            escalation_reason=f"low_confidence_retrieval (top rerank_score={top_score:.2f})",
        )

    answer = generate_answer(question, results)
    citations = sorted(
        {f"{r['source_file']} :: {r['section']}" for r in results if r.get("section")}
    )

    return AnswerResult(
        answer=answer,
        citations=citations,
        confidence_score=top_score,
        escalated=False,
    )


if __name__ == "__main__":
    print("=== Normal, in-scope question ===")
    result = answer_question("What TDS rate applies to dividend payments under Section 88?")
    print(f"Escalated: {result.escalated}")
    print(f"Confidence: {result.confidence_score}")
    print(f"Citations: {result.citations}")
    print(f"Answer: {result.answer}\n")

    print("=== Out-of-scope question ===")
    result = answer_question("What is the capital gains tax rate in the United States?")
    print(f"Escalated: {result.escalated}")
    print(f"Confidence: {result.confidence_score}")
    print(f"Answer: {result.answer}")
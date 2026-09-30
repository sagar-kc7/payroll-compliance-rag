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

from dotenv import load_dotenv

load_dotenv()

from dataclasses import dataclass

from langsmith import traceable

from src.generation.answer import generate_answer
from src.retrieval.reranker import retrieve

# REVISED 2026-09 after a real calibration failure found in live UI
# testing, not the golden set. Original threshold (-2.0) was calibrated
# ONLY against the 25-question golden set's phrasing — which is
# uniformly FORMAL statutory language ("Under Sub-section (2) of
# Section 4...", written that way deliberately for citation-grounding
# in Phase 2). Real users don't talk like that.
#
# Tested 8 natural-phrased questions covering existing golden-set
# topics (no section numbers, casual wording). Two scored WORSE than
# our worst known out-of-scope example (-4.904, "who is the president
# of Nepal"): "What TDS rate applies to dividend payments?" scored
# -5.119, "salary under 1 lakh" scored -7.282 — both while correctly
# retrieving the right section as the #1 result. This means NO single
# threshold can perfectly separate natural in-scope phrasing from
# out-of-scope questions using rerank score alone; formal-vs-any-
# out-of-scope "clean separation" from the original calibration was an
# artifact of the golden set's unnaturally formal wording, not a real
# property of the retrieval system.
#
# Threshold lowered to -7.5 (below the worst natural in-scope score
# found) and the gap is now covered by a SECOND, independent layer:
# the generator's own "answer only from context" instruction (Phase 5).
# Verified directly, not assumed: with this threshold, "who is the
# president of Nepal" now passes the gate (score -4.90) but the
# generator correctly responded "the provided context does not contain
# any information about the current president of Nepal" rather than
# hallucinating an answer. Two-layer defense: gate catches the clearly
# irrelevant cases cheaply (no generation cost), generator catches
# what's left. A single-layer score-only gate could not have handled
# both known failure modes simultaneously — this is a design
# consequence of that finding, not a preference.
LOW_CONFIDENCE_RERANK_THRESHOLD = -7.5

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


@traceable(run_type="chain", name="answer_question")
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
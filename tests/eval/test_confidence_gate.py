"""
Confidence gate tests. Retrieval/reranking cost nothing (local); only
the in-scope case costs a Groq call (the out-of-scope case should
escalate BEFORE reaching the generator — that's itself part of what's
being tested).

Run under pytest:
    pytest tests/eval/test_confidence_gate.py -m eval
"""

from __future__ import annotations

import pytest

from src.pipeline import LOW_CONFIDENCE_RERANK_THRESHOLD, answer_question


@pytest.mark.eval
def test_in_scope_question_answers_without_escalating():
    result = answer_question("What TDS rate applies to dividend payments under Section 88?")
    assert not result.escalated
    assert result.citations, "Expected real citations on a non-escalated answer"
    assert result.confidence_score is not None
    assert result.confidence_score >= LOW_CONFIDENCE_RERANK_THRESHOLD


@pytest.mark.eval
def test_out_of_scope_question_escalates():
    result = answer_question("What is the capital gains tax rate in the United States?")
    assert result.escalated
    assert result.citations == []
    assert result.confidence_score is not None
    assert result.confidence_score < LOW_CONFIDENCE_RERANK_THRESHOLD
    assert "capital gains" not in result.answer.lower(), (
        "Escalated response should be the generic escalation message, "
        "not an attempted answer"
    )
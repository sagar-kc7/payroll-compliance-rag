"""
Generator-layer evaluation: Faithfulness and Answer Relevancy, using the
KNOWN-CORRECT context for each golden question (not the retriever's
output). This isolates generator quality from retrieval quality — a low
score here means the model hallucinates or wanders even when given the
right source text, a different problem than retrieval missing the right
chunk (see test_retrieval.py).

Run standalone for a human-readable report:
    PYTHONPATH=. python -m tests.eval.test_generator

Run under pytest:
    pytest tests/eval/test_generator.py -m eval
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from src.eval.judge import GroqJudge
from src.generation.answer import generate_answer

ROOT = Path(__file__).parent.parent.parent
GOLDEN_SET_PATH = ROOT / "data" / "golden_set" / "golden_set.jsonl"
CHUNKS_PATH = ROOT / "data" / "processed" / "chunks.jsonl"


def load_golden_set() -> list[dict]:
    with GOLDEN_SET_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_chunks() -> list[dict]:
    with CHUNKS_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def gold_context_for(entry: dict, chunks: list[dict]) -> list[dict]:
    """All chunks matching this golden entry's true (source_file, section) —
    the known-correct context, bypassing the retriever entirely."""
    return [
        c
        for c in chunks
        if c["source_file"] == entry["source_file"] and c["section"] == entry["source_clause"]
    ]


def evaluate_generator(sample_size: int | None = None) -> dict:
    golden = load_golden_set()
    chunks = load_chunks()
    judge = GroqJudge()

    if sample_size:
        golden = golden[:sample_size]

    results = []
    for entry in golden:
        context_chunks = gold_context_for(entry, chunks)
        if not context_chunks:
            results.append(
                {"id": entry["id"], "error": "no matching context chunk — check golden set citation"}
            )
            continue

        answer = generate_answer(entry["question"], context_chunks)
        context_texts = [c["text"] for c in context_chunks]

        test_case = LLMTestCase(
            input=entry["question"],
            actual_output=answer,
            retrieval_context=context_texts,
        )

        faithfulness = FaithfulnessMetric(model=judge, threshold=0.5)
        relevancy = AnswerRelevancyMetric(model=judge, threshold=0.5)
        faithfulness.measure(test_case)
        relevancy.measure(test_case)

        results.append(
            {
                "id": entry["id"],
                "question": entry["question"],
                "answer": answer,
                "expected": entry["expected_answer"],
                "faithfulness": faithfulness.score,
                "faithfulness_reason": faithfulness.reason,
                "relevancy": relevancy.score,
                "relevancy_reason": relevancy.reason,
            }
        )

    valid = [r for r in results if "error" not in r]
    avg_faithfulness = sum(r["faithfulness"] for r in valid) / len(valid) if valid else 0.0
    avg_relevancy = sum(r["relevancy"] for r in valid) / len(valid) if valid else 0.0

    return {
        "avg_faithfulness": avg_faithfulness,
        "avg_relevancy": avg_relevancy,
        "total": len(golden),
        "errors": [r for r in results if "error" in r],
        "per_question": results,
    }


@pytest.mark.eval
@pytest.mark.eval
def test_generator_faithfulness_and_relevancy():
    """
    Baseline established 2026-09 against the 15-question pilot golden
    set, using known-correct context (not the retriever): avg
    faithfulness = 0.983, avg relevancy = 0.936. The two sub-0.7
    relevancy scores (gs_004, gs_005) were inspected directly — both
    answers were factually correct and fully grounded, just penalized
    for extra explanatory framing around the direct answer. Judge
    calibration note, not a generator defect. Threshold set below the
    real baseline as a regression floor.
    """
    result = evaluate_generator()
    print(f"\navg faithfulness: {result['avg_faithfulness']:.3f}")
    print(f"avg relevancy: {result['avg_relevancy']:.3f}")
    assert not result["errors"]
    assert result["avg_faithfulness"] >= 0.90, f"faithfulness dropped to {result['avg_faithfulness']:.3f}"
    assert result["avg_relevancy"] >= 0.80, f"relevancy dropped to {result['avg_relevancy']:.3f}"


if __name__ == "__main__":
    result = evaluate_generator()
    print(f"Questions evaluated: {result['total']}")
    print(f"avg faithfulness: {result['avg_faithfulness']:.3f}")
    print(f"avg relevancy: {result['avg_relevancy']:.3f}\n")

    if result["errors"]:
        print(f"{len(result['errors'])} error(s):")
        for e in result["errors"]:
            print(f"  [{e['id']}] {e['error']}")
        print()

    low_scores = [
        r
        for r in result["per_question"]
        if "faithfulness" in r and (r["faithfulness"] < 0.7 or r["relevancy"] < 0.7)
    ]
    if low_scores:
        print(f"{len(low_scores)} question(s) with faithfulness or relevancy < 0.7:\n")
        for r in low_scores:
            print(f"  [{r['id']}] {r['question']}")
            print(f"    answer: {r['answer'][:150]}")
            print(f"    faithfulness: {r['faithfulness']:.2f} — {r['faithfulness_reason']}")
            print(f"    relevancy: {r['relevancy']:.2f} — {r['relevancy_reason']}")
            print()